"""
src/core/orchestrator.py – AsyncIO event loop managing the full pipeline.

Pipeline stages
---------------
AudioChunk → Transcriber → FallacyAgent → ResearcherAgent
                                ↓               ↓
                          StateManager ← ← ← ← ←
                                ↓
                         OBSClient / OverlayServer

The Orchestrator wires these stages together with asyncio.Queue objects
so each stage runs concurrently, keeping end-to-end latency well below
the <2 second budget specified in the architecture document.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from config.settings import settings
from src.core.state_manager import Alert, StateManager, TranscriptSegment
from src.core.stream_handler import AudioChunk, StreamHandler

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    The central async controller for the DebateWatcher pipeline.

    Example::

        orch = Orchestrator()
        await orch.start()          # begin capturing + processing
        await asyncio.sleep(3600)   # run for one hour
        await orch.stop()
    """

    def __init__(self, session_id: str = "default") -> None:
        self._session_id = session_id

        # Queues linking pipeline stages
        self._audio_q: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=32)
        self._transcript_q: asyncio.Queue[TranscriptSegment] = asyncio.Queue(
            maxsize=64
        )
        self._alert_q: asyncio.Queue[Alert] = asyncio.Queue(maxsize=64)

        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Components – initialised lazily in start()
        self._stream_handler: Optional[StreamHandler] = None
        self._state_manager: Optional[StateManager] = None
        self._transcriber = None
        self._fallacy_agent = None
        self._researcher_agent = None
        self._obs_client = None
        self._overlay_server = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all components and begin processing."""
        logger.info("Orchestrator starting (session=%s)…", self._session_id)
        self._running = True

        await self._init_components()

        self._tasks = [
            asyncio.create_task(self._run_audio_ingestion(), name="audio_ingestion"),
            asyncio.create_task(self._run_transcription(), name="transcription"),
            asyncio.create_task(self._run_analysis(), name="analysis"),
            asyncio.create_task(self._run_alert_dispatch(), name="alert_dispatch"),
        ]
        logger.info("Orchestrator pipeline started with %d tasks.", len(self._tasks))

    async def stop(self) -> None:
        """Gracefully shut down all pipeline stages."""
        logger.info("Orchestrator stopping…")
        self._running = False

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._stream_handler:
            await self._stream_handler.stop()
        if self._state_manager:
            await self._state_manager.close()
        logger.info("Orchestrator stopped.")

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _run_audio_ingestion(self) -> None:
        """Stage 1: Consume AudioChunks from StreamHandler into the queue."""
        assert self._stream_handler is not None
        await self._stream_handler.start()
        async for chunk in self._stream_handler.stream():
            if not self._running:
                break
            await self._audio_q.put(chunk)

    async def _run_transcription(self) -> None:
        """Stage 2: Transcribe each AudioChunk → TranscriptSegment."""
        assert self._transcriber is not None
        assert self._state_manager is not None

        while self._running:
            try:
                chunk: AudioChunk = await asyncio.wait_for(
                    self._audio_q.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            t0 = time.monotonic()
            try:
                async for segment in self._transcriber.transcribe(chunk):
                    await self._state_manager.save_segment(segment)
                    if not segment.is_partial:
                        await self._transcript_q.put(segment)
                    latency_ms = (time.monotonic() - t0) * 1000
                    logger.debug(
                        "Transcription latency=%.0fms partial=%s text=%r",
                        latency_ms,
                        segment.is_partial,
                        segment.text[:60],
                    )
            except Exception as exc:
                logger.error("Transcription error: %s", exc, exc_info=True)

    async def _run_analysis(self) -> None:
        """Stage 3: Run fallacy detection and fact-checking concurrently."""
        assert self._fallacy_agent is not None
        assert self._researcher_agent is not None

        while self._running:
            try:
                segment: TranscriptSegment = await asyncio.wait_for(
                    self._transcript_q.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            t0 = time.monotonic()
            try:
                # Run both agents in parallel for lower latency
                fallacy_task = asyncio.create_task(
                    self._fallacy_agent.analyse(segment)
                )
                research_task = asyncio.create_task(
                    self._researcher_agent.check(segment)
                )
                fallacy_alerts, research_alerts = await asyncio.gather(
                    fallacy_task, research_task, return_exceptions=True
                )

                for result in (fallacy_alerts, research_alerts):
                    if isinstance(result, Exception):
                        logger.error("Analysis error: %s", result, exc_info=False)
                        continue
                    for alert in result:
                        if alert.confidence >= settings.pipeline.fallacy_confidence_threshold:
                            await self._alert_q.put(alert)

                logger.debug(
                    "Analysis latency=%.0fms for segment %r",
                    (time.monotonic() - t0) * 1000,
                    segment.text[:40],
                )
            except Exception as exc:
                logger.error("Analysis stage error: %s", exc, exc_info=True)

    async def _run_alert_dispatch(self) -> None:
        """Stage 4: Persist alerts and push high-confidence ones to OBS/overlay."""
        assert self._state_manager is not None

        while self._running:
            try:
                alert: Alert = await asyncio.wait_for(
                    self._alert_q.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                await self._state_manager.save_alert(alert)

                # Auto-approve alerts above the OBS push threshold
                if alert.confidence >= settings.pipeline.obs_push_confidence_threshold:
                    await self._push_to_obs(alert)
                    alert.pushed_to_obs = True
                    await self._state_manager.save_alert(alert)

                logger.info(
                    "Alert dispatched: type=%s fallacy=%s confidence=%.2f pushed=%s",
                    alert.alert_type,
                    alert.fallacy_type,
                    alert.confidence,
                    alert.pushed_to_obs,
                )
            except Exception as exc:
                logger.error("Alert dispatch error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _push_to_obs(self, alert: Alert) -> None:
        payload = {
            "type": alert.alert_type,
            "fallacy_type": alert.fallacy_type,
            "confidence": alert.confidence,
            "explanation": alert.explanation,
        }
        if self._obs_client:
            try:
                await self._obs_client.push_alert(payload)
            except Exception as exc:
                logger.warning("OBS push failed: %s", exc)
        if self._overlay_server:
            try:
                await self._overlay_server.broadcast(payload)
            except Exception as exc:
                logger.warning("Overlay broadcast failed: %s", exc)

    async def _init_components(self) -> None:
        """Lazily import and initialise all pipeline components."""
        from src.agents.fallacy_agent import FallacyAgent
        from src.agents.researcher_agent import ResearcherAgent
        from src.agents.transcriber import Transcriber
        from src.interface.obs_client import OBSClient
        from src.interface.overlay_server import OverlayServer

        self._state_manager = StateManager(session_id=self._session_id)
        await self._state_manager.initialise()

        self._stream_handler = StreamHandler()
        self._transcriber = Transcriber()
        self._fallacy_agent = FallacyAgent(self._state_manager)
        self._researcher_agent = ResearcherAgent(self._state_manager)

        self._obs_client = OBSClient()
        self._overlay_server = OverlayServer()

        # Start overlay server in background
        asyncio.create_task(self._overlay_server.serve(), name="overlay_server")
