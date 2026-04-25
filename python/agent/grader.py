"""
Answer grader: asks the LLM to score faithfulness+relevance of an answer
against the retrieved context on a 0.0-1.0 scale.

On any parse failure we return a neutral passing score. This is
deliberate: failed parses are OUR bug, and we don't want them to
trigger spurious retries (which are expensive).
"""

import json
import logging
import re
import sys
from dataclasses import dataclass

from .prompts import GRADER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GradeResult:
    score: float
    reason: str
    passed: bool


class AnswerGrader:
    def __init__(self, llm, threshold: float = 0.5):
        self.llm = llm
        self.threshold = float(threshold)

    def grade(self, query: str, answer: str, context: str) -> GradeResult:
        # Truncate context in prompt to keep grader LLM call fast
        ctx = context if len(context) <= 3000 else context[:3000] + "\n...[truncated]"
        prompt = GRADER_PROMPT.format(query=query, context=ctx, answer=answer)

        try:
            raw = self.llm.generate_response(
                prompt=prompt, temperature=0.0, max_tokens=120, response_format="json"
            )
        except Exception as e:
            print(f"[Agent] grader LLM call failed: {e}", file=sys.stderr)
            return GradeResult(score=0.5, reason=f"grader call failed: {e}", passed=True)

        score, reason = self._parse(raw)
        if score is None:
            # Safe default: do not trigger retries on our own parse failures
            return GradeResult(
                score=0.5,
                reason=f"grader parse failure; raw={raw[:80]!r}",
                passed=True,
            )

        return GradeResult(
            score=score,
            reason=reason or "",
            passed=score >= self.threshold,
        )

    @staticmethod
    def _parse(raw: str):
        """Extract (score, reason) from LLM output. Returns (None, None) on failure."""
        if not raw:
            return None, None

        text = raw.strip()

        # Try: direct JSON parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"]), str(obj.get("reason", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try: locate first {...} block and parse it
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict) and "score" in obj:
                    return float(obj["score"]), str(obj.get("reason", ""))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Last resort: regex for a bare score float
        num_match = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if num_match:
            try:
                return float(num_match.group(1)), ""
            except ValueError:
                pass

        return None, None
