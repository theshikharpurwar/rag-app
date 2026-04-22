"""
Agentic control loop: route -> retrieve -> generate -> grade -> maybe retry.

Takes injected callables for retrieve_fn and generate_fn so this module
is unit-testable without Qdrant/Ollama, and so local_llm.py's wiring
stays thin.
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .decomposer import QueryDecomposer
from .grader import AnswerGrader, GradeResult
from .prompts import REWRITE_PROMPT
from .query_router import QueryRouter
from .rag_fusion import RAGFusion

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    answer: str
    sources: List[Any]
    trace: dict = field(default_factory=dict)


class AgenticRAG:
    """
    Orchestrates the self-correcting RAG loop.

    retrieve_fn signature: (query: str, weights: tuple[float,float,float]) -> (context_str, sources)
    generate_fn signature: (query: str, context_str: str, chat_history: list) -> answer_str
    """

    def __init__(
        self,
        llm,
        router: QueryRouter,
        grader: AnswerGrader,
        retrieve_fn: Callable[[str, Tuple[float, float, float]], Tuple[str, List[Any]]],
        generate_fn: Callable[[str, str, list], str],
        max_retries: int = 2,
        base_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        decomposer: Optional[QueryDecomposer] = None,
        fusion: Optional[RAGFusion] = None,
    ):
        self.llm = llm
        self.router = router
        self.grader = grader
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.max_retries = int(max_retries)
        self.base_weights = tuple(base_weights)
        self.decomposer = decomposer
        self.fusion = fusion

    def run(self, query: str, chat_history: list) -> AgentResult:
        original_query = (query or "").strip()
        route = self.router.classify(original_query)
        weights = self.router.adjust_weights(route, *self.base_weights)

        attempts = []
        current_query = original_query
        last_answer = ""
        last_sources: List[Any] = []
        last_grade: GradeResult = GradeResult(score=0.0, reason="no attempt", passed=False)

        # Total attempts = 1 (initial) + max_retries
        for attempt_idx in range(self.max_retries + 1):
            sub_trace: List[str] = []
            if self.decomposer and self.fusion:
                sub_queries = self.decomposer.decompose(current_query)
                if len(sub_queries) > 1:
                    context_str, sources = self.fusion.retrieve(sub_queries, weights)
                    sub_trace = list(sub_queries)
                else:
                    context_str, sources = self.retrieve_fn(current_query, weights)
            else:
                context_str, sources = self.retrieve_fn(current_query, weights)

            answer = self.generate_fn(current_query, context_str, chat_history)
            grade = self.grader.grade(current_query, answer, context_str)

            attempts.append({
                "attempt": attempt_idx,
                "query": current_query,
                "score": grade.score,
                "passed": grade.passed,
                "reason": grade.reason,
                "sub_queries": sub_trace,
            })

            last_answer = answer
            last_sources = sources
            last_grade = grade

            if grade.passed:
                break

            if attempt_idx >= self.max_retries:
                break

            # Rewrite query for the next attempt
            try:
                rewritten = self.llm.generate_response(
                    prompt=REWRITE_PROMPT.format(
                        query=original_query,
                        reason=grade.reason or "low confidence",
                    ),
                    temperature=0.3,
                    max_tokens=80,
                )
                rewritten = (rewritten or "").strip().splitlines()[0].strip() if rewritten else ""
                if rewritten and rewritten != current_query:
                    current_query = rewritten
                else:
                    # Rewrite failed to produce something new; no point retrying
                    break
            except Exception as e:
                print(f"[Agent] rewrite LLM call failed: {e}", file=sys.stderr)
                break

        trace = {
            "route": route,
            "weights": list(weights),
            "attempts": attempts,
            "retries_used": max(0, len(attempts) - 1),
            "final_score": last_grade.score,
            "final_passed": last_grade.passed,
        }

        return AgentResult(answer=last_answer, sources=last_sources, trace=trace)
