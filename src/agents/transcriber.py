"""
src/agents/transcriber.py – Low-latency speech-to-text wrapper.

Supports two backends:
* **faster-whisper** (default) – Python-native, GPU-accelerated via CTranslate2.
* **whisper-cpp** – Subprocess call to a pre-compiled whisper.cpp binary for
  maximum CPU performance.

The transcriber emits two types of segments:
* **Partial** – speculative, low-latency text for UI responsiveness.
* **Final** – confirmed text forwarded to the semantic analysis stage.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from config.settings import settings
from src.core.state_manager import TranscriptSegment
from src.core.stream_handler import AudioChunk

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Async wrapper around a speech-recognition backend.

    The transcriber is backend-agnostic: set ``WHISPER_BACKEND`` in ``.env``
    to switch between ``faster-whisper`` (default) and ``whisper-cpp``.

    Example::

        t = Transcriber()
        async for seg in t.transcribe(chunk):
            print(seg.is_partial, seg.text)
    """

    def __init__(self) -> None:
        self._backend = settings.whisper.backend.lower()
        self._model_name = settings.whisper.model
        self._device = settings.whisper.device
        self._compute_type = settings.whisper.compute_type
        self._model = None
        self._lock = asyncio.Lock()
        logger.info(
            "Transcriber init: backend=%s model=%s device=%s",
            self._backend,
            self._model_name,
            self._device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(self, chunk: AudioChunk) -> AsyncIterator[TranscriptSegment]:
        """Yield partial and/or final :class:`TranscriptSegment` for *chunk*."""
        if self._backend == "faster-whisper":
            async for seg in self._transcribe_faster_whisper(chunk):
                yield seg
        elif self._backend == "whisper-cpp":
            async for seg in self._transcribe_whisper_cpp(chunk):
                yield seg
        else:
            logger.error("Unknown Whisper backend: %s", self._backend)

    # ------------------------------------------------------------------
    # faster-whisper backend
    # ------------------------------------------------------------------

    async def _transcribe_faster_whisper(
        self, chunk: AudioChunk
    ) -> AsyncIterator[TranscriptSegment]:
        model = await self._get_faster_whisper_model()
        if model is None:
            return

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_faster_whisper, model, chunk.data
        )
        for seg in result:
            yield TranscriptSegment(
                text=seg["text"].strip(),
                start_time=chunk.start_time + seg["start"],
                end_time=chunk.start_time + seg["end"],
                is_partial=False,
                session_id="default",
            )

    def _run_faster_whisper(self, model, audio: np.ndarray) -> list[dict]:
        """Blocking call – runs in executor thread."""
        try:
            segments, _ = model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
            )
            return [
                {"text": s.text, "start": s.start, "end": s.end}
                for s in segments
            ]
        except Exception as exc:
            logger.error("faster-whisper transcription failed: %s", exc)
            return []

    async def _get_faster_whisper_model(self):
        """Lazy-load the faster-whisper model (thread-safe)."""
        async with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel  # type: ignore

                device = self._device
                if device == "auto":
                    try:
                        import torch  # type: ignore
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"

                logger.info(
                    "Loading faster-whisper model %s on %s (%s)…",
                    self._model_name,
                    device,
                    self._compute_type,
                )
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(
                        self._model_name,
                        device=device,
                        compute_type=self._compute_type,
                    ),
                )
                logger.info("faster-whisper model loaded.")
                return self._model
            except ImportError:
                logger.error(
                    "faster-whisper is not installed.  "
                    "Install with: pip install faster-whisper"
                )
                return None

    # ------------------------------------------------------------------
    # whisper-cpp backend
    # ------------------------------------------------------------------

    async def _transcribe_whisper_cpp(
        self, chunk: AudioChunk
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Calls the whisper.cpp command-line binary.

        Requires ``WHISPER_CPP_BIN`` in the environment pointing to the
        compiled ``./main`` executable and ``WHISPER_CPP_MODEL`` pointing
        to the GGML model file.
        """
        import os
        import wave

        binary = os.environ.get("WHISPER_CPP_BIN", "./whisper.cpp/main")
        model_path = os.environ.get(
            "WHISPER_CPP_MODEL", f"./models/{self._model_name}.bin"
        )

        if not Path(binary).exists():
            logger.error("whisper.cpp binary not found at %s", binary)
            return

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_whisper_cpp,
            chunk.data,
            binary,
            model_path,
        )
        if result:
            yield TranscriptSegment(
                text=result.strip(),
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                is_partial=False,
                session_id="default",
            )

    def _run_whisper_cpp(
        self, audio: np.ndarray, binary: str, model_path: str
    ) -> str:
        """Write audio to a temp WAV file and run the whisper.cpp binary."""
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(settings.audio.sample_rate)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        try:
            proc = subprocess.run(
                [binary, "-m", model_path, "-f", wav_path, "--output-txt"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return proc.stdout
        except subprocess.TimeoutExpired:
            logger.error("whisper.cpp timed out.")
            return ""
        except Exception as exc:
            logger.error("whisper.cpp error: %s", exc)
            return ""
        finally:
            Path(wav_path).unlink(missing_ok=True)
