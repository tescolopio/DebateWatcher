"""
tests/test_fallacy_agent.py – Unit tests for fallacy detection.

The Ollama LLM is mocked so these tests run without a local model.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.fallacy_agent import FallacyAgent, FallacyResult, FALLACY_LIBRARY
from src.core.state_manager import StateManager, TranscriptSegment


@pytest.fixture
def mock_state():
    return MagicMock(spec=StateManager)


@pytest.fixture
def agent(mock_state):
    return FallacyAgent(state_manager=mock_state)


class TestFallacyLibrary:
    def test_library_has_24_entries(self):
        assert len(FALLACY_LIBRARY) == 24

    def test_ad_hominem_in_library(self):
        assert "Ad Hominem" in FALLACY_LIBRARY

    def test_straw_man_in_library(self):
        assert "Straw Man" in FALLACY_LIBRARY


class TestParseResponse:
    def test_parse_valid_fallacy_response(self):
        payload = {
            "fallacy_detected": True,
            "fallacy_type": "Ad Hominem",
            "confidence": 0.88,
            "explanation": "Speaker dismisses argument based on opponent's character.",
            "quote": "you're too young to understand",
        }
        result = FallacyAgent._parse_response(json.dumps(payload))
        assert result is not None
        assert result.fallacy_detected is True
        assert result.fallacy_type == "Ad Hominem"
        assert result.confidence == pytest.approx(0.88)

    def test_parse_no_fallacy_response(self):
        payload = {"fallacy_detected": False}
        result = FallacyAgent._parse_response(json.dumps(payload))
        assert result is not None
        assert result.fallacy_detected is False

    def test_parse_response_with_markdown_fence(self):
        raw = "```json\n" + json.dumps({"fallacy_detected": False}) + "\n```"
        result = FallacyAgent._parse_response(raw)
        assert result is not None
        assert result.fallacy_detected is False

    def test_parse_invalid_json_returns_none(self):
        result = FallacyAgent._parse_response("This is not JSON at all.")
        assert result is None

    def test_parse_json_embedded_in_text(self):
        raw = 'Sure! Here is the result: {"fallacy_detected": false} Thanks!'
        result = FallacyAgent._parse_response(raw)
        assert result is not None
        assert result.fallacy_detected is False


class TestFallacyAgent:
    @pytest.mark.asyncio
    async def test_analyse_returns_empty_when_no_fallacy(self, agent):
        segment = TranscriptSegment(
            text="The tax rate should be reduced to stimulate growth.",
            start_time=10.0,
            end_time=15.0,
        )
        no_fallacy_json = json.dumps({"fallacy_detected": False})

        with patch.object(agent, "_blocking_ollama_call", return_value=no_fallacy_json):
            alerts = await agent.analyse(segment)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_analyse_returns_alert_on_fallacy(self, agent):
        segment = TranscriptSegment(
            text="You can't trust him, he's a liar.",
            start_time=30.0,
            end_time=35.0,
        )
        fallacy_json = json.dumps({
            "fallacy_detected": True,
            "fallacy_type": "Ad Hominem",
            "confidence": 0.92,
            "explanation": "Attacks person rather than the argument.",
        })

        with patch.object(agent, "_blocking_ollama_call", return_value=fallacy_json):
            alerts = await agent.analyse(segment)
        assert len(alerts) == 1
        assert alerts[0].fallacy_type == "Ad Hominem"
        assert alerts[0].confidence == pytest.approx(0.92)
        assert alerts[0].alert_type == "fallacy"

    @pytest.mark.asyncio
    async def test_context_window_maintained(self, agent):
        """Recent segments should accumulate in the context window."""
        no_fallacy_json = json.dumps({"fallacy_detected": False})
        with patch.object(agent, "_blocking_ollama_call", return_value=no_fallacy_json):
            for i in range(7):
                seg = TranscriptSegment(
                    text=f"Statement number {i}.",
                    start_time=float(i * 5),
                    end_time=float(i * 5 + 4),
                )
                await agent.analyse(seg)

        # Context window is 5, so max 5 segments should be retained
        assert len(agent._recent_segments) <= agent._context_window

    @pytest.mark.asyncio
    async def test_analyse_handles_ollama_failure_gracefully(self, agent):
        segment = TranscriptSegment(
            text="Some statement.",
            start_time=0.0,
            end_time=3.0,
        )
        with patch.object(agent, "_blocking_ollama_call", return_value=None):
            alerts = await agent.analyse(segment)
        assert alerts == []
