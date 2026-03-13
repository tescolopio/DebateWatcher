"""
tests/test_transcriber.py – Unit tests for the Whisper transcription wrapper.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.transcriber import Transcriber
from src.core.stream_handler import AudioChunk


@pytest.fixture
def chunk():
    """A 3-second silent audio chunk."""
    data = np.zeros(int(3 * 16000), dtype=np.float32)
    return AudioChunk(data=data, start_time=0.0, end_time=3.0)


class TestTranscriber:
    def test_default_backend_is_faster_whisper(self):
        t = Transcriber()
        assert t._backend == "faster-whisper"

    @pytest.mark.asyncio
    async def test_transcribe_faster_whisper_yields_segments(self, chunk):
        t = Transcriber()
        t._backend = "faster-whisper"

        # Mock the run_faster_whisper result
        mock_result = [
            {"text": "Hello world.", "start": 0.0, "end": 2.0}
        ]

        async def mock_transcribe(audio_chunk):
            from src.core.state_manager import TranscriptSegment
            yield TranscriptSegment(
                text="Hello world.",
                start_time=0.0,
                end_time=2.0,
                is_partial=False,
            )

        with patch.object(t, "_transcribe_faster_whisper", side_effect=mock_transcribe):
            segments = []
            async for seg in t.transcribe(chunk):
                segments.append(seg)

        assert len(segments) == 1
        assert segments[0].text == "Hello world."

    @pytest.mark.asyncio
    async def test_unknown_backend_yields_nothing(self, chunk):
        t = Transcriber()
        t._backend = "unknown-backend"
        segments = []
        async for seg in t.transcribe(chunk):
            segments.append(seg)
        assert segments == []

    def test_run_faster_whisper_returns_segments(self):
        t = Transcriber()

        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = " Test transcript."
        mock_seg.start = 0.0
        mock_seg.end = 2.5
        mock_info = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        audio = np.zeros(16000, dtype=np.float32)
        result = t._run_faster_whisper(mock_model, audio)

        assert len(result) == 1
        assert result[0]["text"] == " Test transcript."
        assert result[0]["start"] == 0.0

    def test_run_faster_whisper_handles_exception(self):
        t = Transcriber()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")
        audio = np.zeros(16000, dtype=np.float32)
        result = t._run_faster_whisper(mock_model, audio)
        assert result == []
