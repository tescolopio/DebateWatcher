"""
tests/test_stream_handler.py – Unit tests for the audio ingestion layer.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.core.stream_handler import AudioChunk, CircularAudioBuffer, _simple_vad


class TestCircularAudioBuffer:
    def test_write_and_read(self):
        buf = CircularAudioBuffer(capacity_seconds=5.0, sample_rate=16000)
        samples = np.ones(8000, dtype=np.float32)
        buf.write(samples)
        result = buf.read(8000)
        assert len(result) == 8000
        assert float(np.mean(result)) == pytest.approx(1.0)

    def test_capacity_limit(self):
        """Buffer should not grow beyond capacity."""
        buf = CircularAudioBuffer(capacity_seconds=1.0, sample_rate=16000)
        # Write 3× the capacity
        for _ in range(3):
            buf.write(np.ones(16000, dtype=np.float32))
        assert len(buf) <= 16000

    def test_total_samples_counter(self):
        buf = CircularAudioBuffer(capacity_seconds=10.0, sample_rate=16000)
        buf.write(np.zeros(1000, dtype=np.float32))
        buf.write(np.zeros(500, dtype=np.float32))
        assert buf.total_samples == 1500

    def test_read_partial(self):
        """Reading more samples than available returns what is in the buffer."""
        buf = CircularAudioBuffer(capacity_seconds=5.0, sample_rate=16000)
        buf.write(np.ones(100, dtype=np.float32))
        result = buf.read(500)
        assert len(result) == 100


class TestSimpleVAD:
    def test_silent_audio_returns_false(self):
        silence = np.zeros(4000, dtype=np.float32)
        assert _simple_vad(silence) is False

    def test_loud_audio_returns_true(self):
        loud = np.ones(4000, dtype=np.float32)
        assert _simple_vad(loud) is True

    def test_threshold_boundary(self):
        # RMS of a constant 0.05 array is 0.05 → above default threshold (0.01)
        signal = np.full(4000, 0.05, dtype=np.float32)
        assert _simple_vad(signal, threshold=0.01) is True
        assert _simple_vad(signal, threshold=0.1) is False


class TestAudioChunk:
    def test_dataclass_creation(self):
        data = np.zeros(4000, dtype=np.float32)
        chunk = AudioChunk(data=data, start_time=0.0, end_time=0.25)
        assert chunk.start_time == 0.0
        assert chunk.speaker_id is None

    def test_speaker_id_assignment(self):
        data = np.zeros(4000, dtype=np.float32)
        chunk = AudioChunk(data=data, start_time=1.0, end_time=2.0, speaker_id="SPEAKER_A")
        assert chunk.speaker_id == "SPEAKER_A"
