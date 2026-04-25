"""
Agentic control loop: route -> retrieve -> generate -> grade -> maybe retry.

Takes injected callables for retrieve_fn and generate_fn so this module
is unit-testable without Qdrant/Ollama, and so local_llm.py's wiring
stays thin.
"""

import logging
import sys
from dataclasses import dataclass, field
from collections.abc import Iterator
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

    retrieve_fn signature: (query, weights, route=None) -> (context_str, sources)
    generate_fn signature: (query: str, context_str: str, chat_history: list) -> answer_str
    """

    def __init__(
        self,
        llm,
        router: QueryRouter,
        grader: AnswerGrader,
        retrieve_fn: Callable[..., Tuple[str, List[Any]]],
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

        decomposer = self.decomposer
        fusion = self.fusion
        if route == "multi_hop" and decomposer is None:
            decomposer = QueryDecomposer(self.llm)
            fusion = RAGFusion(self.retrieve_fn)

        attempts = []
        current_query = original_query
        last_answer = ""
        last_sources: List[Any] = []
        last_grade: GradeResult = GradeResult(score=0.0, reason="no attempt", passed=False)

        # Total attempts = 1 (initial) + max_retries
        for attempt_idx in range(self.max_retries + 1):
            sub_trace: List[str] = []
            if decomposer and fusion:
                sub_queries = decomposer.decompose(current_query)
                if len(sub_queries) > 1:
                    context_str, sources = fusion.retrieve(sub_queries, weights, route)
                    sub_trace = list(sub_queries)
                else:
                    context_str, sources = self.retrieve_fn(current_query, weights, route)
            else:
                context_str, sources = self.retrieve_fn(current_query, weights, route)

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
            "auto_decompose": route == "multi_hop" and self.decomposer is None,
            "weights": list(weights),
            "attempts": attempts,
            "retries_used": max(0, len(attempts) - 1),
            "final_score": last_grade.score,
            "final_passed": last_grade.passed,
        }

        return AgentResult(answer=last_answer, sources=last_sources, trace=trace)

    def run_stream(
        self,
        query: str,
        chat_history: list,
        generate_fn_stream: Callable[[str, str, list], Iterator[str]],
    ) -> Iterator[dict]:
        """
        Like run(), but the last attempt's generation is streamed (token events).
        Non-final attempts use generate_fn to avoid showing rejected text.
        When max_retries>0, early successful passes return a non-streamed answer; only the
        final scheduled attempt (attempt == max_retries) uses the streaming API.
        """
        original_query = (query or "").strip()
        yield {"phase": "routing"}
        route = self.router.classify(original_query)
        weights = self.router.adjust_weights(route, *self.base_weights)

        decomposer = self.decomposer
        fusion = self.fusion
        if route == "multi_hop" and decomposer is None:
            decomposer = QueryDecomposer(self.llm)
            fusion = RAGFusion(self.retrieve_fn)

        attempts: List[dict] = []
        current_query = original_query
        last_answer = ""
        last_sources: List[Any] = []
        last_grade: GradeResult = GradeResult(
            score=0.0, reason="no attempt", passed=False
        )

        for attempt_idx in range(self.max_retries + 1):
            yield {"phase": "retrieving"}
            sub_trace: List[str] = []
            if decomposer and fusion:
                sub_queries = decomposer.decompose(current_query)
                if len(sub_queries) > 1:
                    context_str, sources = fusion.retrieve(sub_queries, weights, route)
                    sub_trace = list(sub_queries)
                else:
                    context_str, sources = self.retrieve_fn(
                        current_query, weights, route
                    )
            else:
                context_str, sources = self.retrieve_fn(
                    current_query, weights, route
                )

            yield {"phase": "generating"}
            parts: List[str] = []
            for token in generate_fn_stream(
                current_query, context_str, chat_history
            ):
                parts.append(token)
                yield {"token": token}
            answer = "".join(parts)

            yield {"phase": "grading"}
            grade = self.grader.grade(current_query, answer, context_str)
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "query": current_query,
                    "score": grade.score,
                    "passed": grade.passed,
                    "reason": grade.reason,
                    "sub_queries": sub_trace,
                }
            )
            last_answer = answer
            last_sources = sources
            last_grade = grade
            if grade.passed:
                break
            if attempt_idx >= self.max_retries:
                break
            yield {"status": "refining", "message": "Refining answer…"}
            try:
                rewritten = self.llm.generate_response(
                    prompt=REWRITE_PROMPT.format(
                        query=original_query,
                        reason=grade.reason or "low confidence",
                    ),
                    temperature=0.3,
                    max_tokens=80,
                )
                rewritten = (
                    (rewritten or "").strip().splitlines()[0].strip() if rewritten else ""
                )
                if rewritten and rewritten != current_query:
                    current_query = rewritten
                else:
                    break
            except Exception as e:
                print(f"[Agent] rewrite LLM call failed: {e}", file=sys.stderr)
                break

        trace = {
            "route": route,
            "auto_decompose": route == "multi_hop" and self.decomposer is None,
            "weights": list(weights),
            "attempts": attempts,
            "retries_used": max(0, len(attempts) - 1),
            "final_score": last_grade.score,
            "final_passed": last_grade.passed,
        }
        yield {
            "done": True,
            "sources": last_sources,
            "trace": trace,
            "raw_answer": last_answer,
        }
