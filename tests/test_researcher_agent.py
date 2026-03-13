"""
tests/test_researcher_agent.py – Unit tests for the RAG fact-checking agent.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.researcher_agent import ResearcherAgent, _chunk_text, _format_timestamp
from src.core.state_manager import StateManager, TranscriptSegment


@pytest.fixture
def mock_state():
    state = MagicMock(spec=StateManager)
    state.search_similar = AsyncMock(return_value=[])
    return state


@pytest.fixture
def agent(mock_state):
    return ResearcherAgent(state_manager=mock_state)


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = _chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        words = ["word"] * 200
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_chunks_contain_original_words(self):
        text = "The quick brown fox jumps over the lazy dog."
        chunks = _chunk_text(text, chunk_size=20, overlap=5)
        reconstructed = " ".join(chunks)
        for word in text.split():
            assert word in reconstructed


class TestFormatTimestamp:
    def test_zero_seconds(self):
        assert _format_timestamp(0.0) == "00:00"

    def test_one_minute(self):
        assert _format_timestamp(60.0) == "01:00"

    def test_five_twelve(self):
        assert _format_timestamp(312.0) == "05:12"

    def test_fractional_seconds_truncated(self):
        assert _format_timestamp(61.9) == "01:01"


class TestResearcherAgent:
    @pytest.mark.asyncio
    async def test_check_no_similar_returns_empty(self, agent, mock_state):
        mock_state.search_similar.return_value = []
        segment = TranscriptSegment(
            text="Some claim.",
            start_time=10.0,
            end_time=13.0,
        )
        agent._kb_loaded = True
        alerts = await agent.check(segment)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_check_no_contradiction_returns_empty(self, agent, mock_state):
        # Similar segment exists but happened < 30s ago → skipped
        mock_state.search_similar.return_value = [
            {
                "text": "Recent similar claim.",
                "start_time": 5.0,  # < 30s before current segment at 10.0
                "speaker_id": "SPEAKER_A",
                "distance": 0.1,
            }
        ]
        segment = TranscriptSegment(
            text="New similar claim.",
            start_time=10.0,
            end_time=13.0,
        )
        agent._kb_loaded = True
        alerts = await agent.check(segment)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_check_detects_contradiction(self, agent, mock_state):
        mock_state.search_similar.return_value = [
            {
                "text": "I have never raised taxes.",
                "start_time": 0.0,   # > 30s before current at 60.0
                "speaker_id": "SPEAKER_A",
                "distance": 0.15,
            }
        ]
        segment = TranscriptSegment(
            text="We raised taxes three times in the last term.",
            start_time=60.0,
            end_time=65.0,
        )
        agent._kb_loaded = True

        contradiction_response = json.dumps({
            "contradiction_detected": True,
            "confidence": 0.88,
            "explanation": "Earlier denial contradicts current admission of tax increases.",
        })

        with patch.object(
            agent, "_check_contradiction_pair", return_value=json.loads(contradiction_response)
        ):
            alerts = await agent.check(segment)

        assert len(alerts) == 1
        assert alerts[0].alert_type == "contradiction"
        assert alerts[0].confidence == pytest.approx(0.88)
        assert "00:00" in alerts[0].explanation

    @pytest.mark.asyncio
    async def test_kb_load_skipped_when_dir_missing(self, agent, tmp_path):
        """No error should be raised if knowledge_base/ doesn't exist."""
        with patch("src.agents.researcher_agent._get_knowledge_base_dir", return_value=tmp_path / "nonexistent"):
            agent._kb_loaded = False
            await agent._ensure_kb_loaded()
        assert agent._kb_loaded is True
        assert agent._kb_docs == []
