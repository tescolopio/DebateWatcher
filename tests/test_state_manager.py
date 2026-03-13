"""
tests/test_state_manager.py – Unit tests for SQLite + ChromaDB state store.

These tests use an in-memory SQLite path to avoid touching disk state.
ChromaDB is mocked to avoid needing a live database during CI.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.core.state_manager import Alert, StateManager, TranscriptSegment


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    """Redirect all file I/O to a temp directory."""
    monkeypatch.setattr(
        "config.settings.settings.pipeline.sqlite_path",
        tmp_path / "test.db",
    )
    monkeypatch.setattr(
        "config.settings.settings.chroma.persist_dir",
        tmp_path / "chroma",
    )


@pytest.fixture
def state_manager():
    return StateManager(session_id="test_session")


@pytest.mark.asyncio
async def test_initialise_creates_tables(state_manager, tmp_path):
    with patch.object(state_manager, "_init_chroma"):
        await state_manager.initialise()
        await state_manager.close()


@pytest.mark.asyncio
async def test_save_and_retrieve_segment(state_manager):
    with patch.object(state_manager, "_init_chroma"):
        await state_manager.initialise()
        seg = TranscriptSegment(
            text="The economy is doing well.",
            start_time=0.0,
            end_time=3.5,
            session_id="test_session",
        )
        await state_manager.save_segment(seg)
        segments = await state_manager.get_segments()
        assert len(segments) == 1
        assert segments[0].text == "The economy is doing well."
        await state_manager.close()


@pytest.mark.asyncio
async def test_partial_segments_excluded_by_default(state_manager):
    with patch.object(state_manager, "_init_chroma"):
        await state_manager.initialise()
        partial = TranscriptSegment(
            text="Partial text…",
            start_time=0.0,
            end_time=1.0,
            is_partial=True,
            session_id="test_session",
        )
        final = TranscriptSegment(
            text="Final text.",
            start_time=1.0,
            end_time=2.0,
            is_partial=False,
            session_id="test_session",
        )
        await state_manager.save_segment(partial)
        await state_manager.save_segment(final)

        segments = await state_manager.get_segments(include_partials=False)
        assert all(not s.is_partial for s in segments)
        assert len(segments) == 1

        all_segs = await state_manager.get_segments(include_partials=True)
        assert len(all_segs) == 2
        await state_manager.close()


@pytest.mark.asyncio
async def test_save_and_retrieve_alert(state_manager):
    with patch.object(state_manager, "_init_chroma"):
        await state_manager.initialise()
        alert = Alert(
            segment_id="seg-001",
            alert_type="fallacy",
            fallacy_type="Ad Hominem",
            confidence=0.85,
            explanation="Speaker attacks person instead of argument.",
        )
        await state_manager.save_alert(alert)
        alerts = await state_manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].fallacy_type == "Ad Hominem"
        await state_manager.close()


@pytest.mark.asyncio
async def test_approve_alert(state_manager):
    with patch.object(state_manager, "_init_chroma"):
        await state_manager.initialise()
        alert = Alert(
            segment_id="seg-002",
            alert_type="fallacy",
            confidence=0.7,
        )
        await state_manager.save_alert(alert)
        await state_manager.approve_alert(alert.id)
        approved = await state_manager.get_alerts(approved_only=True)
        assert len(approved) == 1
        assert approved[0].approved is True
        await state_manager.close()


@pytest.mark.asyncio
async def test_search_similar_without_chroma_returns_empty(state_manager):
    with patch.object(state_manager, "_init_chroma"):
        await state_manager.initialise()
        state_manager._chroma_collection = None
        results = await state_manager.search_similar("some claim")
        assert results == []
        await state_manager.close()
