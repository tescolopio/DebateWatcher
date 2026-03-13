"""
src/core/state_manager.py – Persistent debate state using SQLite + ChromaDB.

Responsibilities
----------------
* Persist every transcript segment to a local SQLite database.
* Index segment text into a ChromaDB vector collection so the RAG agent
  can retrieve semantically similar earlier statements in sub-second time.
* Expose helpers for contradiction detection (e.g., "speaker X claimed Y
  at 05:12 but now claims Z").
* Provide a simple read API consumed by the web dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_CREATE_SEGMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS segments (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    speaker_id  TEXT,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    text        TEXT NOT NULL,
    is_partial  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
"""

_CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id              TEXT PRIMARY KEY,
    segment_id      TEXT NOT NULL,
    alert_type      TEXT NOT NULL,
    fallacy_type    TEXT,
    confidence      REAL NOT NULL,
    explanation     TEXT,
    approved        INTEGER NOT NULL DEFAULT 0,
    pushed_to_obs   INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    FOREIGN KEY (segment_id) REFERENCES segments(id)
);
"""


@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    speaker_id: Optional[str] = None
    is_partial: bool = False
    session_id: str = field(default_factory=lambda: "default")
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


@dataclass
class Alert:
    segment_id: str
    alert_type: str             # "fallacy" | "contradiction" | "fact_check"
    confidence: float
    fallacy_type: Optional[str] = None
    explanation: Optional[str] = None
    approved: bool = False
    pushed_to_obs: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


class StateManager:
    """
    Thread-safe, async-friendly state store for the live debate session.

    Example::

        state = StateManager()
        await state.initialise()
        await state.save_segment(segment)
        results = await state.search_similar("the economy grew last year", n=5)
    """

    def __init__(self, session_id: str = "default") -> None:
        self._session_id = session_id
        self._db: Optional[object] = None          # aiosqlite connection
        self._chroma_collection: Optional[object] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """Open SQLite connection and initialise ChromaDB collection."""
        settings.ensure_data_dirs()
        await self._init_sqlite()
        await asyncio.get_event_loop().run_in_executor(None, self._init_chroma)
        logger.info("StateManager initialised (session=%s)", self._session_id)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            logger.info("StateManager closed SQLite connection.")

    # ------------------------------------------------------------------
    # Segment persistence
    # ------------------------------------------------------------------

    async def save_segment(self, segment: TranscriptSegment) -> None:
        """Persist *segment* to SQLite and index its text in ChromaDB."""
        async with self._lock:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO segments
                    (id, session_id, speaker_id, start_time, end_time, text,
                     is_partial, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    segment.id,
                    self._session_id,
                    segment.speaker_id,
                    segment.start_time,
                    segment.end_time,
                    segment.text,
                    int(segment.is_partial),
                    segment.created_at,
                ),
            )
            await self._db.commit()

        if not segment.is_partial and self._chroma_collection is not None:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._chroma_upsert,
                segment,
            )

    async def get_segments(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
        include_partials: bool = False,
    ) -> list[TranscriptSegment]:
        """Return recent segments, newest-last."""
        sid = session_id or self._session_id
        partial_filter = "" if include_partials else "AND is_partial = 0"
        async with self._lock:
            cursor = await self._db.execute(
                f"""
                SELECT id, session_id, speaker_id, start_time, end_time, text,
                       is_partial, created_at
                FROM segments
                WHERE session_id = ? {partial_filter}
                ORDER BY start_time ASC
                LIMIT ?
                """,
                (sid, limit),
            )
            rows = await cursor.fetchall()
        return [
            TranscriptSegment(
                id=r[0],
                session_id=r[1],
                speaker_id=r[2],
                start_time=r[3],
                end_time=r[4],
                text=r[5],
                is_partial=bool(r[6]),
                created_at=r[7],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Alert persistence
    # ------------------------------------------------------------------

    async def save_alert(self, alert: Alert) -> None:
        async with self._lock:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO alerts
                    (id, segment_id, alert_type, fallacy_type, confidence,
                     explanation, approved, pushed_to_obs, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    alert.id,
                    alert.segment_id,
                    alert.alert_type,
                    alert.fallacy_type,
                    alert.confidence,
                    alert.explanation,
                    int(alert.approved),
                    int(alert.pushed_to_obs),
                    alert.created_at,
                ),
            )
            await self._db.commit()

    async def get_alerts(
        self,
        approved_only: bool = False,
        limit: int = 50,
    ) -> list[Alert]:
        filter_clause = "WHERE approved = 1" if approved_only else ""
        async with self._lock:
            cursor = await self._db.execute(
                f"""
                SELECT id, segment_id, alert_type, fallacy_type, confidence,
                       explanation, approved, pushed_to_obs, created_at
                FROM alerts
                {filter_clause}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
        return [
            Alert(
                id=r[0],
                segment_id=r[1],
                alert_type=r[2],
                fallacy_type=r[3],
                confidence=r[4],
                explanation=r[5],
                approved=bool(r[6]),
                pushed_to_obs=bool(r[7]),
                created_at=r[8],
            )
            for r in rows
        ]

    async def approve_alert(self, alert_id: str) -> None:
        async with self._lock:
            await self._db.execute(
                "UPDATE alerts SET approved = 1 WHERE id = ?", (alert_id,)
            )
            await self._db.commit()

    # ------------------------------------------------------------------
    # Semantic search (RAG)
    # ------------------------------------------------------------------

    async def search_similar(
        self, query: str, n: int = 5, session_id: Optional[str] = None
    ) -> list[dict]:
        """
        Return the *n* most semantically similar segments to *query*.

        Each result dict contains ``text``, ``start_time``, ``speaker_id``,
        ``distance``.
        """
        if self._chroma_collection is None:
            return []
        sid = session_id or self._session_id
        results = await asyncio.get_event_loop().run_in_executor(
            None, self._chroma_query, query, n, sid
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _init_sqlite(self) -> None:
        try:
            import aiosqlite  # type: ignore

            self._db = await aiosqlite.connect(str(settings.pipeline.sqlite_path))
            await self._db.execute(_CREATE_SEGMENTS_TABLE)
            await self._db.execute(_CREATE_ALERTS_TABLE)
            await self._db.commit()
            logger.info("SQLite initialised at %s", settings.pipeline.sqlite_path)
        except ImportError:
            logger.error("aiosqlite not installed.  State persistence disabled.")

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings as ChromaInternalSettings  # type: ignore

            client = chromadb.PersistentClient(
                path=str(settings.chroma.persist_dir),
                settings=ChromaInternalSettings(anonymized_telemetry=False),
            )
            self._chroma_collection = client.get_or_create_collection(
                name=settings.chroma.collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB initialised (collection=%s)", settings.chroma.collection
            )
        except ImportError:
            logger.warning("chromadb not installed.  Semantic search disabled.")
        except Exception as exc:
            logger.warning("ChromaDB init failed: %s", exc)

    def _chroma_upsert(self, segment: TranscriptSegment) -> None:
        if self._chroma_collection is None:
            return
        self._chroma_collection.upsert(
            ids=[segment.id],
            documents=[segment.text],
            metadatas=[
                {
                    "session_id": self._session_id,
                    "speaker_id": segment.speaker_id or "",
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                }
            ],
        )

    def _chroma_query(
        self, query: str, n: int, session_id: str
    ) -> list[dict]:
        if self._chroma_collection is None:
            return []
        try:
            res = self._chroma_collection.query(
                query_texts=[query],
                n_results=n,
                where={"session_id": session_id},
            )
            results = []
            if res and res.get("documents"):
                docs = res["documents"][0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                for doc, meta, dist in zip(docs, metas, dists):
                    results.append(
                        {
                            "text": doc,
                            "start_time": meta.get("start_time", 0.0),
                            "speaker_id": meta.get("speaker_id", ""),
                            "distance": dist,
                        }
                    )
            return results
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return []
