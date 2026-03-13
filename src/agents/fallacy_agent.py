"""
src/agents/fallacy_agent.py – LLM-based logical fallacy detection.

Uses a local Ollama instance to analyse finalized transcript segments and
return structured JSON identifying rhetorical fallacies.

The agent is grounded in a library of 24 logical fallacies, maintains a
rolling context window of recent statements, and returns results as
:class:`FallacyResult` objects consumed by the Orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings
from src.core.state_manager import Alert, StateManager, TranscriptSegment

logger = logging.getLogger(__name__)

# ── Fallacy library ──────────────────────────────────────────────────────────
FALLACY_LIBRARY: list[str] = [
    "Ad Hominem",
    "Straw Man",
    "Appeal to Authority",
    "False Dichotomy",
    "Slippery Slope",
    "Circular Reasoning",
    "Hasty Generalization",
    "Red Herring",
    "Appeal to Emotion",
    "Bandwagon Fallacy",
    "Post Hoc Ergo Propter Hoc",
    "False Equivalence",
    "Moving the Goalposts",
    "Burden of Proof Reversal",
    "Anecdotal Evidence",
    "Texas Sharpshooter",
    "Survivorship Bias",
    "Whataboutism",
    "Gish Gallop",
    "Motte and Bailey",
    "No True Scotsman",
    "Appeal to Nature",
    "Genetic Fallacy",
    "Sunk Cost Fallacy",
]

_SYSTEM_PROMPT = """\
You are a formal logic referee monitoring a live debate.
Your job is to detect logical fallacies in real time.

You will receive a transcript segment and recent context.
Analyse the segment for logical fallacies from this exhaustive list:
{fallacy_list}

If you detect a fallacy, respond ONLY with valid JSON in this exact format:
{{
  "fallacy_detected": true,
  "fallacy_type": "<name from the list above>",
  "confidence": <float 0.0-1.0>,
  "explanation": "<10-20 word plain-English description for the audience>",
  "quote": "<the exact words that constitute the fallacy>"
}}

If no fallacy is detected, respond ONLY with:
{{
  "fallacy_detected": false
}}

Rules:
- Respond with JSON only. No preamble, no commentary.
- Use a HIGH confidence threshold – only flag clear, unambiguous fallacies.
- Do not flag rhetorical questions or strong opinions as fallacies.
""".format(
    fallacy_list="\n".join(f"- {f}" for f in FALLACY_LIBRARY)
)


@dataclass
class FallacyResult:
    fallacy_detected: bool
    fallacy_type: Optional[str] = None
    confidence: float = 0.0
    explanation: Optional[str] = None
    quote: Optional[str] = None


class FallacyAgent:
    """
    Sends transcript segments to a local Ollama LLM and parses the
    structured JSON response for logical fallacies.

    Example::

        agent = FallacyAgent(state_manager)
        alerts = await agent.analyse(segment)
    """

    def __init__(
        self,
        state_manager: StateManager,
        context_window: int = 5,
    ) -> None:
        self._state = state_manager
        self._context_window = context_window
        self._recent_segments: list[str] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyse(self, segment: TranscriptSegment) -> list[Alert]:
        """
        Analyse *segment* for logical fallacies.

        Returns a (possibly empty) list of :class:`Alert` objects.
        """
        async with self._lock:
            self._recent_segments.append(segment.text)
            if len(self._recent_segments) > self._context_window:
                self._recent_segments.pop(0)

        context = "\n".join(self._recent_segments[:-1])
        current = self._recent_segments[-1]

        result = await self._call_ollama(context, current)
        if result is None or not result.fallacy_detected:
            return []

        alert = Alert(
            segment_id=segment.id,
            alert_type="fallacy",
            fallacy_type=result.fallacy_type,
            confidence=result.confidence,
            explanation=result.explanation,
        )
        return [alert]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_ollama(
        self, context: str, current_text: str
    ) -> Optional[FallacyResult]:
        """Call the local Ollama API and parse the JSON response."""
        user_message = (
            f"Recent context:\n{context}\n\nCurrent statement:\n{current_text}"
            if context
            else f"Statement:\n{current_text}"
        )

        loop = asyncio.get_event_loop()
        raw_response = await loop.run_in_executor(
            None, self._blocking_ollama_call, user_message
        )
        if raw_response is None:
            return None

        return self._parse_response(raw_response)

    def _blocking_ollama_call(self, user_message: str) -> Optional[str]:
        """Synchronous Ollama call – runs in a thread executor."""
        try:
            import ollama  # type: ignore

            response = ollama.chat(
                model=settings.ollama.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={
                    "temperature": settings.ollama.analysis_temperature,
                    "num_predict": 256,
                },
            )
            return response["message"]["content"]
        except ImportError:
            logger.error("ollama package not installed.  pip install ollama")
            return None
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            return None

    @staticmethod
    def _parse_response(raw: str) -> Optional[FallacyResult]:
        """Extract the JSON payload from the LLM response."""
        raw = raw.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to find a JSON object within the text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    logger.warning("Could not parse Ollama response as JSON: %r", raw[:200])
                    return None
            else:
                logger.warning("No JSON found in Ollama response: %r", raw[:200])
                return None

        fallacy_detected = bool(data.get("fallacy_detected", False))
        if not fallacy_detected:
            return FallacyResult(fallacy_detected=False)

        return FallacyResult(
            fallacy_detected=True,
            fallacy_type=data.get("fallacy_type"),
            confidence=float(data.get("confidence", 0.0)),
            explanation=data.get("explanation"),
            quote=data.get("quote"),
        )
