"""
Query router: classifies a user query as "specific" (keyword-heavy) or
"broad" (relationship/summary) and produces adjusted RRF weights that
softly bias retrieval towards the better-suited path.

The bias is a *soft nudge* (default 0.15): a misclassification still
leaves all three retrieval paths (vector / BM25 / graph) contributing.
"""

import logging
import sys
from typing import Literal, Tuple

from .prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)

Route = Literal["specific", "broad"]


class QueryRouter:
    def __init__(self, llm, weight_boost: float = 0.15):
        """
        Args:
            llm: object with .generate_response(prompt, temperature, max_tokens)
            weight_boost: fraction of weight to shift between BM25 and graph
                based on the classification. Must be < min(BM25_WEIGHT, GRAPH_WEIGHT)
                to keep weights non-negative under typical configs.
        """
        self.llm = llm
        self.weight_boost = float(weight_boost)

    def classify(self, query: str) -> Route:
        """Return 'specific' or 'broad'. Falls back to 'broad' on any failure."""
        prompt = ROUTER_PROMPT.format(query=query)
        try:
            raw = self.llm.generate_response(
                prompt=prompt, temperature=0.1, max_tokens=8
            )
        except Exception as e:
            print(f"[Agent] router LLM call failed: {e}", file=sys.stderr)
            return "broad"

        token = (raw or "").strip().upper()
        # Take first non-empty token, strip punctuation
        first = token.split()[0] if token.split() else ""
        first = first.strip(".,:;!?-_\"'`")

        if first.startswith("SPECIFIC"):
            return "specific"
        if first.startswith("BROAD"):
            return "broad"

        logger.info(f"[Agent] router fallback: unparseable response {raw!r} -> broad")
        return "broad"

    def adjust_weights(
        self,
        route: Route,
        vector_w: float,
        bm25_w: float,
        graph_w: float,
    ) -> Tuple[float, float, float]:
        """
        Shift `weight_boost` from one path to another based on route, then
        renormalize so the three weights sum to the same total as the input.

        - specific: shift from graph -> bm25
        - broad:    shift from bm25  -> graph
        """
        v, b, g = float(vector_w), float(bm25_w), float(graph_w)
        shift = self.weight_boost

        if route == "specific":
            b += shift
            g -= shift
        elif route == "broad":
            g += shift
            b -= shift

        # Clamp to non-negative to guard against overly aggressive boosts
        v = max(v, 0.0)
        b = max(b, 0.0)
        g = max(g, 0.0)

        total = v + b + g
        original_total = float(vector_w) + float(bm25_w) + float(graph_w)
        if total <= 0:
            return float(vector_w), float(bm25_w), float(graph_w)

        scale = original_total / total
        return v * scale, b * scale, g * scale
