"""
src/interface/obs_client.py – OBS WebSocket 5.0 controller.

Communicates with OBS Studio via the obs-websocket-py library to:
* Update ``Text (GDI+)`` source text fields with live transcripts.
* Toggle Browser Source scene items for animated fallacy overlays.
* Push structured JSON payloads to the overlay via ``SetInputSettings``.

All operations are async-safe via a threading lock around the synchronous
obs-websocket-py client.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# OBS scene item / source names expected in the OBS scene collection.
# These can be overridden via environment variables in a future release.
SOURCE_TRANSCRIPT = "DW_Transcript"
SOURCE_ALERT_BANNER = "DW_AlertBanner"
SCENE_ALERTS = "DebateWatcher_Alerts"


class OBSClient:
    """
    Async-friendly wrapper around the OBS WebSocket 5.0 API.

    Example::

        client = OBSClient()
        await client.connect()
        await client.push_alert({"fallacy_type": "Ad Hominem", "confidence": 0.9})
        await client.disconnect()
    """

    def __init__(self) -> None:
        self._host = settings.obs.host
        self._port = settings.obs.port
        self._password = settings.obs.password
        self._ws = None
        self._lock = asyncio.Lock()
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """
        Attempt to connect to OBS.  Returns ``True`` on success.

        Failures are logged as warnings so the rest of the pipeline
        continues without OBS.
        """
        async with self._lock:
            if self._connected:
                return True
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._blocking_connect)
                self._connected = True
                logger.info(
                    "OBSClient connected to %s:%d", self._host, self._port
                )
                return True
            except Exception as exc:
                logger.warning(
                    "OBS connection failed (OBS may not be running): %s", exc
                )
                return False

    async def disconnect(self) -> None:
        async with self._lock:
            if self._ws and self._connected:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._ws.disconnect)
                except Exception:
                    pass
            self._connected = False
            logger.info("OBSClient disconnected.")

    # ------------------------------------------------------------------
    # Alert / overlay API
    # ------------------------------------------------------------------

    async def push_alert(self, payload: dict[str, Any]) -> None:
        """
        Push a structured alert payload to OBS.

        Serialises *payload* as JSON and writes it to the
        ``DW_AlertBanner`` Browser Source ``url`` parameter so the
        overlay JavaScript can render the alert.
        """
        if not self._connected:
            await self.connect()
        if not self._connected:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._blocking_push_alert, payload)

    async def update_transcript(self, text: str) -> None:
        """Update the live transcript text source in OBS."""
        if not self._connected:
            await self.connect()
        if not self._connected:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._blocking_set_text, text)

    # ------------------------------------------------------------------
    # Synchronous helpers (run in executor threads)
    # ------------------------------------------------------------------

    def _blocking_connect(self) -> None:
        try:
            from obswebsocket import obsws  # type: ignore

            self._ws = obsws(self._host, self._port, self._password)
            self._ws.connect()
        except ImportError:
            raise RuntimeError(
                "obs-websocket-py not installed.  pip install obs-websocket-py"
            )

    def _blocking_push_alert(self, payload: dict) -> None:
        if self._ws is None:
            return
        try:
            from obswebsocket import requests as obs_requests  # type: ignore

            json_payload = json.dumps(payload)
            # Embed the payload in the Browser Source URL as a fragment
            # so the overlay JS can read it via location.hash
            overlay_url = (
                f"http://{settings.overlay.host}:{settings.overlay.port}"
                f"/overlay#alert={json_payload}"
            )
            req = obs_requests.SetInputSettings(
                inputName=SOURCE_ALERT_BANNER,
                inputSettings={"url": overlay_url},
            )
            self._ws.call(req)
            logger.debug("OBS alert pushed: %s", payload.get("fallacy_type"))
        except Exception as exc:
            logger.warning("OBS push_alert failed: %s", exc)

    def _blocking_set_text(self, text: str) -> None:
        if self._ws is None:
            return
        try:
            from obswebsocket import requests as obs_requests  # type: ignore

            req = obs_requests.SetInputSettings(
                inputName=SOURCE_TRANSCRIPT,
                inputSettings={"text": text[-500:]},  # cap to 500 chars
            )
            self._ws.call(req)
        except Exception as exc:
            logger.warning("OBS update_transcript failed: %s", exc)
