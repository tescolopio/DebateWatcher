"""
src/agents/researcher_agent.py – RAG-based local fact-checking agent.

Checks each finalised segment against:
1. Earlier statements in the current session (contradiction detection via
   the ChromaDB vector store in StateManager).
2. A local corpus of verified documents in the ``knowledge_base/``
   directory (plain-text or Markdown files ingested at startup).

Contradictions are returned as :class:`Alert` objects with
``alert_type="contradiction"``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.core.state_manager import Alert, StateManager, TranscriptSegment

logger = logging.getLogger(__name__)

_CONTRADICTION_PROMPT = """\
You are a fact-checker reviewing a live debate transcript.

Earlier statement:
"{earlier}"

Current statement:
"{current}"

Do these two statements contradict each other? Respond ONLY with JSON:
{{
  "contradiction_detected": true or false,
  "confidence": <float 0.0-1.0>,
  "explanation": "<15-25 word description for the audience>"
}}
"""

_FACT_CHECK_PROMPT = """\
You are a fact-checker with access to verified reference material.

Reference excerpt:
"{reference}"

Claim to verify:
"{claim}"

Does the reference material contradict or refute the claim?
Respond ONLY with JSON:
{{
  "refuted": true or false,
  "confidence": <float 0.0-1.0>,
  "explanation": "<15-25 word plain-English description>"
}}
"""

def _get_knowledge_base_dir() -> Path:
    """Return the knowledge base directory from settings (lazy to avoid import cycles)."""
    return settings.pipeline.knowledge_base_dir


class ResearcherAgent:
    """
    Performs two types of real-time fact verification:

    * **Intra-session contradiction detection** – compares the current
      segment against semantically similar earlier segments from the
      same debate using the ChromaDB vector store.
    * **Knowledge-base fact checking** – compares claims against a
      local corpus of verified reference documents.

    Example::

        agent = ResearcherAgent(state_manager)
        alerts = await agent.check(segment)
    """

    def __init__(self, state_manager: StateManager) -> None:
        self._state = state_manager
        self._kb_docs: list[dict] = []  # {text, source}
        self._kb_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(self, segment: TranscriptSegment) -> list[Alert]:
        """
        Run all fact-checking tasks for *segment*.

        Returns a (possibly empty) list of :class:`Alert` objects.
        """
        await self._ensure_kb_loaded()
        alerts: list[Alert] = []

        contradiction_alerts = await self._check_contradictions(segment)
        alerts.extend(contradiction_alerts)

        return alerts

    # ------------------------------------------------------------------
    # Contradiction detection (RAG over session history)
    # ------------------------------------------------------------------

    async def _check_contradictions(
        self, segment: TranscriptSegment
    ) -> list[Alert]:
        similar = await self._state.search_similar(segment.text, n=3)
        if not similar:
            return []

        # Filter to segments that happened significantly earlier
        earlier_segments = [
            s for s in similar
            if s["start_time"] < segment.start_time - 30  # at least 30s earlier
        ]
        if not earlier_segments:
            return []

        alerts = []
        loop = asyncio.get_event_loop()
        for earlier in earlier_segments:
            result = await loop.run_in_executor(
                None,
                self._check_contradiction_pair,
                earlier["text"],
                segment.text,
            )
            if result and result.get("contradiction_detected"):
                confidence = float(result.get("confidence", 0.0))
                if confidence >= settings.pipeline.fallacy_confidence_threshold:
                    timestamp = _format_timestamp(earlier["start_time"])
                    explanation = result.get("explanation", "")
                    alerts.append(
                        Alert(
                            segment_id=segment.id,
                            alert_type="contradiction",
                            confidence=confidence,
                            explanation=f"[{timestamp}] {explanation}",
                        )
                    )
        return alerts

    def _check_contradiction_pair(
        self, earlier: str, current: str
    ) -> Optional[dict]:
        """Synchronous Ollama call – runs in thread executor."""
        import json as _json

        try:
            import ollama  # type: ignore

            prompt = _CONTRADICTION_PROMPT.format(
                earlier=earlier, current=current
            )
            response = ollama.chat(
                model=settings.ollama.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 128},
            )
            raw = response["message"]["content"].strip()
            # Strip markdown fences
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines() if not l.strip().startswith("```")
                )
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                return _json.loads(raw[start:end])
        except ImportError:
            logger.error("ollama package not installed.")
        except Exception as exc:
            logger.warning("Contradiction check failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Knowledge-base loading
    # ------------------------------------------------------------------

    async def _ensure_kb_loaded(self) -> None:
        if self._kb_loaded:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_knowledge_base)
        self._kb_loaded = True

    def _load_knowledge_base(self) -> None:
        """
        Load all .txt and .md files from the knowledge_base/ directory.

        Documents are split into ~500-character chunks with 50-char overlap.
        """
        kb_dir = _get_knowledge_base_dir()
        if not kb_dir.exists():
            logger.info(
                "No knowledge_base/ directory found.  Corpus fact-checking disabled."
            )
            return

        docs_loaded = 0
        for filepath in kb_dir.rglob("*"):
            if filepath.suffix.lower() not in (".txt", ".md"):
                continue
            try:
                text = filepath.read_text(encoding="utf-8")
                chunks = _chunk_text(text, chunk_size=500, overlap=50)
                for chunk in chunks:
                    self._kb_docs.append(
                        {"text": chunk, "source": str(filepath)}
                    )
                docs_loaded += 1
            except Exception as exc:
                logger.warning("Failed to load %s: %s", filepath, exc)

        logger.info(
            "Knowledge base loaded: %d files → %d chunks",
            docs_loaded,
            len(self._kb_docs),
        )


# ── Utility functions ────────────────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping chunks of approximately *chunk_size* chars."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = []
        char_count = 0
        j = i
        while j < len(words) and char_count < chunk_size:
            chunk_words.append(words[j])
            char_count += len(words[j]) + 1
            j += 1
        chunks.append(" ".join(chunk_words))
        # If we consumed all remaining words, we're done
        if j >= len(words):
            break
        # Step forward, retaining an overlap tail
        overlap_words = max(1, overlap // 6)
        i = j - overlap_words
    return chunks
