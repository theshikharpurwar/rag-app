"""
LLM-based query decomposition for RAG-Fusion.

Skips decomposition for short, simple questions (heuristic) to avoid extra
retrieval cost. On any failure, returns a single-query list so retrieval
never regresses below the baseline.
"""

import logging
import re
import sys
from typing import List

from .prompts import DECOMPOSITION_PROMPT

logger = logging.getLogger(__name__)

_TRIGGER_TOKENS = frozenset({
    "and", "or", "vs", "versus", "compare", "both",
})


class QueryDecomposer:
    def __init__(self, llm, max_subqueries: int = 3, min_words: int = 8):
        self.llm = llm
        self.max_subqueries = int(max_subqueries)
        self.min_words = int(min_words)

    def _is_simple(self, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return True
        words = q.split()
        if len(words) >= self.min_words:
            return False
        lower = q.lower()
        tokens = set(re.findall(r"[a-zA-Z]+", lower))
        if tokens & _TRIGGER_TOKENS:
            return False
        if q.count("?") > 1:
            return False
        return True

    @staticmethod
    def _parse_numbered_lines(raw: str) -> List[str]:
        if not raw:
            return []
        numbered = re.compile(r"^\s*\d+[\).\s]+\s*(.+?)\s*$")
        lines: List[str] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = numbered.match(line)
            if m:
                lines.append(m.group(1).strip())
        if lines:
            return lines
        return [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]

    def decompose(self, query: str) -> List[str]:
        q = (query or "").strip()
        if not q:
            return [q]

        if self._is_simple(q):
            return [q]

        prompt = DECOMPOSITION_PROMPT.format(n=self.max_subqueries, query=q)
        try:
            raw = self.llm.generate_response(
                prompt=prompt, temperature=0.2, max_tokens=200,
            )
        except Exception as e:
            print(f"[Agent] decomposer LLM call failed: {e}", file=sys.stderr)
            return [q]

        if not raw:
            return [q]

        parts = self._parse_numbered_lines(raw)
        if not parts:
            return [q]

        seen = set()
        out: List[str] = []
        for p in parts:
            p = p.strip()
            if not p or p.lower() in seen:
                continue
            seen.add(p.lower())
            out.append(p)
            if len(out) >= self.max_subqueries:
                break

        if not out:
            return [q]

        # Always include the original query first so cross-query RRF cannot omit the user's wording.
        full = [q]
        for p in out:
            if p.lower() != q.lower() and p not in full:
                full.append(p)

        if len(full) == 1:
            return [q]

        return full
