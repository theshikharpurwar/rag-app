"""LLM-as-Judge metrics for Phase 4 evaluation."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.grader import AnswerGrader

from config import (
    EVAL_JUDGE_TEMPERATURE,
    EVAL_MAX_CONCURRENCY,
    JUDGE_LLM_MODEL,
    OLLAMA_API_BASE,
)
from llm.ollama_llm import OllamaLLM

_FAITH_PROMPT = """You are an evaluator. Rate how well the ANSWER is grounded in the CONTEXT (faithfulness).
Return ONLY valid JSON: {{"score": <0.0-1.0>, "reason": "<brief>"}}
Question: {query}
CONTEXT:
{context}
ANSWER:
{answer}
"""

_RELEVANCY_PROMPT = """You are an evaluator. Rate how well the ANSWER addresses the QUESTION (relevancy).
Return ONLY valid JSON: {{"score": <0.0-1.0>, "reason": "<brief>"}}
Question: {query}
ANSWER:
{answer}
"""

_RECALL_PROMPT = """You are an evaluator. The CONTEXT was retrieved to answer the question.
Rate how well the CONTEXT contains the information needed to support the GOLD reference answer (recall).
Return ONLY valid JSON: {{"score": <0.0-1.0>, "reason": "<brief>"}}
Question: {query}
GOLD reference answer:
{gold}
CONTEXT:
{context}
"""


@dataclass
class MetricScores:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    conciseness: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "conciseness": self.conciseness,
        }


class EvaluationJudge:
    def __init__(self, llm: Optional[OllamaLLM] = None):
        self.llm = llm or OllamaLLM(model_name=JUDGE_LLM_MODEL, api_base=OLLAMA_API_BASE)
        self._temperature = EVAL_JUDGE_TEMPERATURE

    def score_all(
        self,
        query: str,
        answer: str,
        context: str,
        gold_answer: str,
        gold_pages: Optional[List[int]] = None,
        sources: Optional[List[Any]] = None,
    ) -> MetricScores:
        """Score faithfulness, relevancy, and context recall.

        When retrieval is non-empty, the three LLM judge calls run under
        :class:`~concurrent.futures.ThreadPoolExecutor` with at most
        ``min(3, EVAL_MAX_CONCURRENCY)`` workers. Each call catches its own
        failures and returns a neutral score (see ``_call_json_score``), so
        overlapping work does not leave partially applied side effects.
        """
        ctx = context if context else ""
        has_sources = bool(sources)
        has_context = bool(ctx.strip())

        # If the retriever returned nothing, faithfulness and context_recall are
        # structurally 0.0 — the answer is necessarily ungrounded and the context
        # cannot support the gold answer. Avoid letting a noisy LLM judge return
        # arbitrary scores for empty context.
        empty_retrieval = not has_sources and not has_context
        if empty_retrieval:
            f = 0.0
            cr = 0.0
            r = self._relevancy(query, answer)
        else:
            workers = max(1, min(3, EVAL_MAX_CONCURRENCY))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_f = ex.submit(self._faithfulness, query, answer, ctx)
                fut_r = ex.submit(self._relevancy, query, answer)
                fut_cr = ex.submit(
                    self._context_recall,
                    query,
                    gold_answer,
                    ctx,
                    gold_pages,
                    sources,
                )
                f = fut_f.result()
                r = fut_r.result()
                cr = fut_cr.result()
        c = _conciseness(gold_answer, answer)
        return MetricScores(
            faithfulness=f, answer_relevancy=r, context_recall=cr, conciseness=c
        )

    def _call_json_score(self, prompt: str) -> float:
        try:
            raw = self.llm.generate_response(
                prompt=prompt,
                temperature=self._temperature,
                max_tokens=200,
            )
        except Exception as e:
            print(f"[Eval] judge LLM call failed: {e}", file=sys.stderr)
            return 0.5
        score, _ = AnswerGrader._parse(raw)
        if score is None:
            return 0.5
        return _clamp01(float(score))

    def _faithfulness(self, query: str, answer: str, context: str) -> float:
        ctx = _truncate(context, 3500)
        p = _FAITH_PROMPT.format(query=query, context=ctx, answer=answer)
        return self._call_json_score(p)

    def _relevancy(self, query: str, answer: str) -> float:
        p = _RELEVANCY_PROMPT.format(query=query, answer=answer)
        return self._call_json_score(p)

    def _context_recall(
        self,
        query: str,
        gold: str,
        context: str,
        gold_pages: Optional[List[int]],
        sources: Optional[List[Any]],
    ) -> float:
        ctx = _truncate(context, 3500)
        p = _RECALL_PROMPT.format(query=query, gold=gold, context=ctx)
        llm_score = self._call_json_score(p)
        if gold_pages and sources:
            overlap = _page_overlap_score(sources, gold_pages)
            if overlap is not None:
                return _clamp01(max(llm_score, overlap))
        return llm_score


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + "\n...[truncated]"


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _conciseness(gold_answer: str, answer: str) -> float:
    g = (gold_answer or "").split()
    a = (answer or "").split()
    if not a:
        return 0.0 if g else 1.0
    ratio = min(1.0, len(g) / max(1, len(a)))
    return _clamp01(ratio)


def _page_int(val: Any) -> Optional[int]:
    try:
        if val is None:
            return None
        if isinstance(val, int):
            return val
        s = str(val).strip()
        if s.isdigit():
            return int(s)
    except (TypeError, ValueError):
        pass
    return None


def _page_overlap_score(sources: List[Any], gold_pages: List[int]) -> Optional[float]:
    if not gold_pages:
        return None
    want = set(gold_pages)
    found: set = set()
    for s in sources or []:
        if isinstance(s, dict):
            p = _page_int(s.get("page"))
            if p is not None and p in want:
                found.add(p)
    if not found:
        return 0.0
    return len(found) / max(1, len(want))


def sources_to_context_string(sources: List[Any]) -> str:
    parts = []
    for i, s in enumerate(sources or []):
        if not isinstance(s, dict):
            continue
        text = str(s.get("text", ""))
        page = s.get("page", "")
        parts.append(f"SOURCE {i+1} (Page {page}): {text}")
    return "\n\n".join(parts)
