"""
src/core/stream_handler.py – Real-time audio ingestion and chunking.

Uses a circular (ring) buffer to capture audio from a system device or a
Virtual Audio Cable routed from OBS.  Overlapping 3-5 second windows are
emitted as NumPy arrays so the transcription pipeline always has enough
context across segment boundaries.

Voice Activity Detection (VAD) is applied to each chunk before forwarding
to the transcriber, which reduces idle-CPU load during silent periods.

Speaker diarisation (pyannote-audio) is optional and toggled via config.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A single, time-stamped audio segment ready for transcription."""

    data: np.ndarray          # float32, mono, sample_rate=16000
    start_time: float         # seconds since stream start
    end_time: float
    speaker_id: Optional[str] = None   # populated if diarization is enabled


@dataclass
class StreamHandlerConfig:
    sample_rate: int = field(default_factory=lambda: settings.audio.sample_rate)
    chunk_seconds: float = field(default_factory=lambda: settings.audio.chunk_seconds)
    chunk_overlap: float = field(default_factory=lambda: settings.audio.chunk_overlap)
    device: Optional[str] = field(default_factory=lambda: settings.audio.device or None)
    diarization_enabled: bool = field(
        default_factory=lambda: settings.diarization.enabled
    )


class CircularAudioBuffer:
    """Thread-safe ring buffer that accumulates raw audio samples."""

    def __init__(self, capacity_seconds: float, sample_rate: int) -> None:
        capacity = int(capacity_seconds * sample_rate)
        self._buf: collections.deque[float] = collections.deque(maxlen=capacity)
        self._sample_rate = sample_rate
        self._total_samples: int = 0

    def write(self, samples: np.ndarray) -> None:
        self._buf.extend(samples.tolist())
        self._total_samples += len(samples)

    def read(self, n_samples: int) -> np.ndarray:
        available = min(n_samples, len(self._buf))
        data = list(self._buf)[-available:]
        return np.array(data, dtype=np.float32)

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def __len__(self) -> int:
        return len(self._buf)


def _simple_vad(chunk: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True when the RMS energy of *chunk* exceeds *threshold*.

    This is a lightweight heuristic.  For production workloads, replace
    with silero-VAD or webrtcvad for higher accuracy.
    """
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    return rms > threshold


class StreamHandler:
    """
    Ingests live audio via ``sounddevice`` and emits :class:`AudioChunk`
    objects through an ``asyncio.Queue``.

    Usage::

        handler = StreamHandler()
        async for chunk in handler.stream():
            await process(chunk)
    """

    def __init__(self, config: Optional[StreamHandlerConfig] = None) -> None:
        self._cfg = config or StreamHandlerConfig()
        self._sr = self._cfg.sample_rate
        self._chunk_samples = int(self._cfg.chunk_seconds * self._sr)
        self._overlap_samples = int(self._cfg.chunk_overlap * self._sr)
        self._step_samples = self._chunk_samples - self._overlap_samples

        capacity_seconds = max(self._cfg.chunk_seconds * 3, 30.0)
        self._ring = CircularAudioBuffer(capacity_seconds, self._sr)
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=32)
        self._running = False
        self._stream_start: float = 0.0
        self._diarizer: Optional[Callable] = None

        # Diarizer is loaded lazily during start() to avoid blocking __init__
        if self._cfg.diarization_enabled:
            self._diarizer = None  # populated in start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the audio device and begin background capture."""
        import time

        self._running = True
        self._stream_start = time.monotonic()

        if self._cfg.diarization_enabled and self._diarizer is None:
            loop = asyncio.get_event_loop()
            self._diarizer = await loop.run_in_executor(None, self._load_diarizer)

        logger.info(
            "StreamHandler starting: device=%s, sr=%d, chunk=%.1fs, overlap=%.1fs",
            self._cfg.device or "<default>",
            self._sr,
            self._cfg.chunk_seconds,
            self._cfg.chunk_overlap,
        )
        asyncio.get_event_loop().run_in_executor(None, self._blocking_capture)

    async def stop(self) -> None:
        self._running = False
        logger.info("StreamHandler stopped.")

    async def stream(self) -> AsyncIterator[AudioChunk]:
        """Async generator that yields :class:`AudioChunk` objects."""
        while self._running or not self._queue.empty():
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield chunk
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blocking_capture(self) -> None:
        """Run in a thread executor – blocks on sounddevice.InputStream."""
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.error(
                "sounddevice is not installed.  Audio capture unavailable.  "
                "Install it with: pip install sounddevice"
            )
            return

        import time

        accumulated: list[np.ndarray] = []
        accumulated_samples = 0

        def _callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            nonlocal accumulated_samples
            if status:
                logger.warning("sounddevice status: %s", status)
            mono = indata[:, 0].astype(np.float32)
            self._ring.write(mono)
            accumulated.append(mono.copy())
            accumulated_samples += len(mono)

            if accumulated_samples >= self._step_samples:
                segment = np.concatenate(accumulated)[-self._chunk_samples :]
                elapsed = self._ring.total_samples / self._sr
                start_t = max(0.0, elapsed - self._cfg.chunk_seconds)

                if _simple_vad(segment):
                    ac = AudioChunk(
                        data=segment,
                        start_time=start_t,
                        end_time=elapsed,
                    )
                    asyncio.get_event_loop().call_soon_threadsafe(
                        self._queue.put_nowait, ac
                    )

                # retain the overlap tail
                tail_samples = min(self._overlap_samples, len(segment))
                accumulated.clear()
                accumulated.append(segment[-tail_samples:])
                accumulated_samples = tail_samples

        device_arg = self._cfg.device if self._cfg.device else None
        with sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            device=device_arg,
            callback=_callback,
        ):
            while self._running:
                time.sleep(0.05)

    def _load_diarizer(self) -> Optional[Callable]:
        """Lazily load pyannote Pipeline for speaker diarization."""
        try:
            from pyannote.audio import Pipeline  # type: ignore

            hf_token = settings.diarization.hf_auth_token or None
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            logger.info("pyannote diarization pipeline loaded.")
            return pipeline
        except Exception as exc:
            logger.warning("Could not load pyannote diarizer: %s", exc)
            return None
