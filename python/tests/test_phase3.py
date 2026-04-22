# python/tests/test_phase3.py
"""
Unit tests for Phase 3: Agentic self-correction and Phase 3.5 decomposition
/ RAG-Fusion.

Covers QueryRouter (classification + weight adjustment), AnswerGrader
(JSON parsing + threshold behaviour), AgenticRAG (happy / retry /
exhaust), QueryDecomposer, RAGFusion, and nested decomposition + retry.

Run: cd python && python -m pytest tests/test_phase3.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# Shared mock LLM
# =============================================================================

class ScriptedLLM:
    """
    Minimal fake LLM. Returns pre-programmed responses by detecting which
    prompt is being sent. Also records the full prompt log for assertions.
    """

    def __init__(self, router_response="BROAD",
                 grader_responses=None,
                 rewrite_response="rewritten query",
                 decompose_response=None,
                 decompose_responses=None):
        self.router_response = router_response
        self.grader_responses = list(grader_responses or [])
        self.rewrite_response = rewrite_response
        self.decompose_response = decompose_response
        self._decompose_queue = (
            list(decompose_responses) if decompose_responses is not None else None
        )
        self.calls = []  # list of dicts: {"kind": ..., "prompt": ...}

    def _classify_prompt(self, prompt):
        if "You break a complex question" in prompt:
            return "decompose"
        if "SPECIFIC" in prompt and "BROAD" in prompt and "Question:" in prompt:
            return "router"
        if "JSON:" in prompt and '"score"' in prompt:
            return "grader"
        if "Rewritten question:" in prompt:
            return "rewrite"
        return "unknown"

    def generate_response(self, prompt, context=None, max_tokens=1000,
                          temperature=0.7, messages=None):
        kind = self._classify_prompt(prompt)
        self.calls.append({"kind": kind, "prompt": prompt})

        if kind == "decompose":
            if self._decompose_queue:
                return self._decompose_queue.pop(0)
            if self.decompose_response is not None:
                return self.decompose_response
            return "1. sub one\n2. sub two"
        if kind == "router":
            return self.router_response
        if kind == "grader":
            if not self.grader_responses:
                return '{"score": 1.0, "reason": "default pass"}'
            return self.grader_responses.pop(0)
        if kind == "rewrite":
            return self.rewrite_response
        return ""


# =============================================================================
# QueryRouter
# =============================================================================

class TestQueryRouter:
    def test_classify_specific(self):
        from agent.query_router import QueryRouter
        llm = ScriptedLLM(router_response="SPECIFIC")
        r = QueryRouter(llm)
        assert r.classify("What year was this founded?") == "specific"

    def test_classify_broad(self):
        from agent.query_router import QueryRouter
        llm = ScriptedLLM(router_response="BROAD")
        r = QueryRouter(llm)
        assert r.classify("Summarise the main argument.") == "broad"

    def test_classify_with_punctuation(self):
        from agent.query_router import QueryRouter
        llm = ScriptedLLM(router_response=" specific.\nsome trailing text")
        r = QueryRouter(llm)
        assert r.classify("q") == "specific"

    def test_classify_unparseable_falls_back_to_broad(self):
        from agent.query_router import QueryRouter
        llm = ScriptedLLM(router_response="maybe kinda probably")
        r = QueryRouter(llm)
        assert r.classify("q") == "broad"

    def test_classify_llm_exception_falls_back(self):
        from agent.query_router import QueryRouter

        class BoomLLM:
            def generate_response(self, **kw):
                raise RuntimeError("network down")

        r = QueryRouter(BoomLLM())
        assert r.classify("q") == "broad"

    def test_adjust_weights_specific_shifts_to_bm25(self):
        from agent.query_router import QueryRouter
        r = QueryRouter(ScriptedLLM(), weight_boost=0.1)
        v, b, g = r.adjust_weights("specific", 0.4, 0.3, 0.3)
        assert b > 0.3
        assert g < 0.3
        assert abs((v + b + g) - 1.0) < 1e-9  # renormalised to original total

    def test_adjust_weights_broad_shifts_to_graph(self):
        from agent.query_router import QueryRouter
        r = QueryRouter(ScriptedLLM(), weight_boost=0.1)
        v, b, g = r.adjust_weights("broad", 0.4, 0.3, 0.3)
        assert g > 0.3
        assert b < 0.3
        assert abs((v + b + g) - 1.0) < 1e-9

    def test_adjust_weights_preserves_total(self):
        from agent.query_router import QueryRouter
        r = QueryRouter(ScriptedLLM(), weight_boost=0.5)  # aggressive; would push bm25 negative
        v, b, g = r.adjust_weights("broad", 0.4, 0.3, 0.3)
        assert b >= 0.0
        assert g >= 0.0
        assert abs((v + b + g) - 1.0) < 1e-9


# =============================================================================
# AnswerGrader
# =============================================================================

class TestAnswerGrader:
    def test_valid_json_pass(self):
        from agent.grader import AnswerGrader
        llm = ScriptedLLM(grader_responses=['{"score": 0.9, "reason": "all good"}'])
        g = AnswerGrader(llm, threshold=0.5)
        res = g.grade("q", "a", "ctx")
        assert res.score == pytest.approx(0.9)
        assert res.passed is True
        assert res.reason == "all good"

    def test_valid_json_fail(self):
        from agent.grader import AnswerGrader
        llm = ScriptedLLM(grader_responses=['{"score": 0.2, "reason": "hallucinated"}'])
        g = AnswerGrader(llm, threshold=0.5)
        res = g.grade("q", "a", "ctx")
        assert res.score == pytest.approx(0.2)
        assert res.passed is False

    def test_threshold_boundary_inclusive(self):
        from agent.grader import AnswerGrader
        llm = ScriptedLLM(grader_responses=['{"score": 0.5, "reason": "borderline"}'])
        g = AnswerGrader(llm, threshold=0.5)
        res = g.grade("q", "a", "ctx")
        assert res.passed is True

    def test_json_with_surrounding_text(self):
        from agent.grader import AnswerGrader
        llm = ScriptedLLM(grader_responses=[
            'Here is my grade: {"score": 0.7, "reason": "ok"} end'
        ])
        g = AnswerGrader(llm, threshold=0.5)
        res = g.grade("q", "a", "ctx")
        assert res.score == pytest.approx(0.7)
        assert res.passed is True

    def test_malformed_response_defaults_to_pass(self):
        from agent.grader import AnswerGrader
        llm = ScriptedLLM(grader_responses=["i have no idea what format to use"])
        g = AnswerGrader(llm, threshold=0.5)
        res = g.grade("q", "a", "ctx")
        assert res.passed is True  # safe default
        assert res.score == pytest.approx(0.5)
        assert "parse failure" in res.reason

    def test_llm_exception_defaults_to_pass(self):
        from agent.grader import AnswerGrader

        class BoomLLM:
            def generate_response(self, **kw):
                raise RuntimeError("oops")

        g = AnswerGrader(BoomLLM(), threshold=0.5)
        res = g.grade("q", "a", "ctx")
        assert res.passed is True
        assert res.score == pytest.approx(0.5)


# =============================================================================
# AgenticRAG control loop
# =============================================================================

def _make_retrieve_fn(contexts_by_query=None, default_ctx="default context"):
    """Build a retrieve_fn that returns scripted contexts based on the query."""
    mapping = dict(contexts_by_query or {})
    call_log = []

    def _fn(query, weights):
        call_log.append({"query": query, "weights": weights})
        ctx = mapping.get(query, default_ctx)
        sources = [{"id": 1, "text": ctx}]
        return ctx, sources

    return _fn, call_log


def _make_generate_fn(answers_by_query=None, default_answer="default answer"):
    mapping = dict(answers_by_query or {})
    call_log = []

    def _fn(query, context_str, chat_history):
        call_log.append({"query": query, "context": context_str})
        return mapping.get(query, default_answer)

    return _fn, call_log


class TestAgenticRAG:
    def test_happy_path_no_retries(self):
        """First attempt passes → exactly one route/retrieve/generate/grade call."""
        from agent import AgenticRAG, AnswerGrader, QueryRouter

        llm = ScriptedLLM(
            router_response="SPECIFIC",
            grader_responses=['{"score": 0.9, "reason": "good"}'],
        )
        retrieve_fn, r_log = _make_retrieve_fn()
        generate_fn, g_log = _make_generate_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
        )

        res = agent.run("What year?", chat_history=[])

        assert res.answer == "default answer"
        assert res.trace["route"] == "specific"
        assert res.trace["retries_used"] == 0
        assert res.trace["final_passed"] is True
        assert len(res.trace["attempts"]) == 1
        assert res.trace["attempts"][0]["sub_queries"] == []
        assert len(r_log) == 1
        assert len(g_log) == 1
        # No rewrite should have been requested
        assert not any(c["kind"] == "rewrite" for c in llm.calls)

    def test_retry_then_pass(self):
        """First grade fails, rewrite issued, second grade passes."""
        from agent import AgenticRAG, AnswerGrader, QueryRouter

        llm = ScriptedLLM(
            router_response="BROAD",
            grader_responses=[
                '{"score": 0.1, "reason": "irrelevant"}',
                '{"score": 0.8, "reason": "better"}',
            ],
            rewrite_response="a better phrased question",
        )
        retrieve_fn, r_log = _make_retrieve_fn()
        generate_fn, g_log = _make_generate_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
        )

        res = agent.run("original question", chat_history=[])

        assert res.trace["retries_used"] == 1
        assert res.trace["final_passed"] is True
        assert len(res.trace["attempts"]) == 2
        assert res.trace["attempts"][0]["query"] == "original question"
        assert res.trace["attempts"][1]["query"] == "a better phrased question"
        assert len(r_log) == 2
        assert len(g_log) == 2
        assert sum(1 for c in llm.calls if c["kind"] == "rewrite") == 1

    def test_exhaust_retries_returns_last_answer(self):
        """All attempts fail → trace shows retries_used == max_retries."""
        from agent import AgenticRAG, AnswerGrader, QueryRouter

        llm = ScriptedLLM(
            router_response="BROAD",
            grader_responses=[
                '{"score": 0.1, "reason": "a"}',
                '{"score": 0.2, "reason": "b"}',
                '{"score": 0.0, "reason": "c"}',
            ],
            rewrite_response="rewrite-1",
        )
        # Make every rewrite produce a different query so the loop doesn't short-circuit
        rewrites = iter(["rewrite-1", "rewrite-2"])
        original_generate = llm.generate_response

        def _patched(prompt, **kw):
            kind = llm._classify_prompt(prompt)
            if kind == "rewrite":
                llm.calls.append({"kind": "rewrite", "prompt": prompt})
                try:
                    return next(rewrites)
                except StopIteration:
                    return "rewrite-fallback"
            return original_generate(prompt, **kw)

        llm.generate_response = _patched

        retrieve_fn, r_log = _make_retrieve_fn()
        generate_fn, g_log = _make_generate_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
        )

        res = agent.run("q0", chat_history=[])

        assert res.trace["retries_used"] == 2
        assert res.trace["final_passed"] is False
        assert len(res.trace["attempts"]) == 3
        assert [a["query"] for a in res.trace["attempts"]] == [
            "q0", "rewrite-1", "rewrite-2",
        ]
        assert len(r_log) == 3
        assert len(g_log) == 3

    def test_rewrite_identical_breaks_loop(self):
        """If rewrite returns the same query, loop stops to avoid wasted work."""
        from agent import AgenticRAG, AnswerGrader, QueryRouter

        llm = ScriptedLLM(
            router_response="BROAD",
            grader_responses=['{"score": 0.1, "reason": "bad"}'],
            rewrite_response="q0",  # identical to original
        )
        retrieve_fn, r_log = _make_retrieve_fn()
        generate_fn, g_log = _make_generate_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            max_retries=3,
            base_weights=(0.4, 0.3, 0.3),
        )

        res = agent.run("q0", chat_history=[])

        assert len(res.trace["attempts"]) == 1
        assert res.trace["retries_used"] == 0
        assert res.trace["final_passed"] is False

    def test_weights_adjusted_for_route(self):
        """Router 'specific' should shift weight towards BM25 before retrieval."""
        from agent import AgenticRAG, AnswerGrader, QueryRouter

        llm = ScriptedLLM(
            router_response="SPECIFIC",
            grader_responses=['{"score": 0.9, "reason": "ok"}'],
        )
        retrieve_fn, r_log = _make_retrieve_fn()
        generate_fn, _ = _make_generate_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
        )

        agent.run("q", chat_history=[])

        _, b_w, g_w = r_log[0]["weights"]
        assert b_w > 0.3
        assert g_w < 0.3


# =============================================================================
# Phase 3.5: QueryDecomposer
# =============================================================================

class TestQueryDecomposer:
    def test_simple_query_skips_llm(self):
        from agent.decomposer import QueryDecomposer

        llm = ScriptedLLM()
        d = QueryDecomposer(llm, max_subqueries=3, min_words=8)
        q = "Who is the CEO?"
        assert d.decompose(q) == [q]
        assert not any(c["kind"] == "decompose" for c in llm.calls)

    def test_complex_query_calls_llm(self):
        from agent.decomposer import QueryDecomposer

        llm = ScriptedLLM(decompose_response="1. First aspect\n2. Second aspect")
        d = QueryDecomposer(llm, max_subqueries=3, min_words=8)
        q = "one two three four five six seven and eight nine"
        subs = d.decompose(q)
        assert subs[0] == q
        assert len(subs) >= 2
        assert any(c["kind"] == "decompose" for c in llm.calls)

    def test_malformed_llm_falls_back(self):
        from agent.decomposer import QueryDecomposer

        llm = ScriptedLLM(decompose_response="")
        d = QueryDecomposer(llm, max_subqueries=3, min_words=8)
        q = "one two three four five six seven and eight nine"
        assert d.decompose(q) == [q]

    def test_llm_exception_falls_back(self):
        from agent.decomposer import QueryDecomposer

        class BoomLLM:
            def generate_response(self, **kw):
                raise RuntimeError("down")

        d = QueryDecomposer(BoomLLM(), max_subqueries=3, min_words=8)
        q = "one two three four five six seven and eight nine"
        assert d.decompose(q) == [q]

    def test_max_subqueries_cap(self):
        from agent.decomposer import QueryDecomposer

        llm = ScriptedLLM(
            decompose_response="1. a\n2. b\n3. c\n4. d\n5. e",
        )
        d = QueryDecomposer(llm, max_subqueries=2, min_words=8)
        q = "one two three four five six seven and eight nine"
        subs = d.decompose(q)
        assert subs[0] == q
        assert len(subs) <= 3


# =============================================================================
# Phase 3.5: RAGFusion
# =============================================================================

class TestRAGFusion:
    def test_single_subquery_preserves_one_chunk(self):
        from agent.rag_fusion import RAGFusion

        def retrieve_fn(query, weights):
            return f"ctx:{query}", [
                {"text": "chunk", "page": 1, "document": "D", "score": 0.5},
            ]

        fusion = RAGFusion(retrieve_fn, rrf_k=60, top_k=5)
        ctx1, s1 = fusion.retrieve(["only"], (0.4, 0.3, 0.3))
        assert len(s1) == 1
        assert s1[0]["text"] == "chunk"
        assert "chunk" in ctx1

    def test_overlap_dedupes(self):
        from agent.rag_fusion import RAGFusion

        def retrieve_fn(query, weights):
            if query == "a":
                return "", [
                    {"text": "overlap text", "page": 1, "document": "D", "score": 0.9},
                ]
            return "", [
                {"text": "overlap text", "page": 1, "document": "D", "score": 0.7},
            ]

        fusion = RAGFusion(retrieve_fn, rrf_k=60, top_k=5)
        _ctx, sources = fusion.retrieve(["a", "b"], (0.4, 0.3, 0.3))
        assert len(sources) == 1

    def test_disjoint_union(self):
        from agent.rag_fusion import RAGFusion

        def retrieve_fn(query, weights):
            if query == "a":
                return "", [{"text": "alpha", "page": 1, "document": "D", "score": 0.9}]
            return "", [{"text": "beta", "page": 2, "document": "D", "score": 0.8}]

        fusion = RAGFusion(retrieve_fn, rrf_k=60, top_k=5)
        _ctx, sources = fusion.retrieve(["a", "b"], (0.4, 0.3, 0.3))
        texts = {s["text"] for s in sources}
        assert texts == {"alpha", "beta"}


# =============================================================================
# Phase 3.5: AgenticRAG + decomposition
# =============================================================================

class TestAgenticRAG_Decomposition:
    def test_without_fusion_empty_sub_queries_trace(self):
        from agent import AgenticRAG, AnswerGrader, QueryRouter

        llm = ScriptedLLM(
            router_response="SPECIFIC",
            grader_responses=['{"score": 0.9, "reason": "good"}'],
        )
        retrieve_fn, _ = _make_retrieve_fn()
        generate_fn, _ = _make_generate_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
        )
        res = agent.run("short", chat_history=[])
        assert res.trace["attempts"][0]["sub_queries"] == []

    def test_fusion_populates_sub_queries_and_nested_retry(self):
        from agent import AgenticRAG, AnswerGrader, QueryRouter
        from agent.decomposer import QueryDecomposer
        from agent.rag_fusion import RAGFusion

        COMPLEX = "one two three four five six seven and eight nine"
        REWRITE_COMPLEX = "one two three four five six seven and eight rewrite"

        llm = ScriptedLLM(
            router_response="BROAD",
            grader_responses=[
                '{"score": 0.1, "reason": "bad"}',
                '{"score": 0.85, "reason": "ok"}',
            ],
            rewrite_response=REWRITE_COMPLEX,
            decompose_response="1. subq alpha\n2. subq beta",
        )
        retrieve_fn, r_log = _make_retrieve_fn()
        generate_fn, _ = _make_generate_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
            decomposer=QueryDecomposer(llm, max_subqueries=3, min_words=8),
            fusion=RAGFusion(retrieve_fn, rrf_k=60, top_k=5),
        )

        res = agent.run(COMPLEX, chat_history=[])

        assert res.trace["retries_used"] == 1
        assert len(res.trace["attempts"]) == 2
        assert len(res.trace["attempts"][0]["sub_queries"]) > 1
        assert len(res.trace["attempts"][1]["sub_queries"]) > 1
        # 3 sub-queries × 2 attempts
        assert len(r_log) == 6

    def test_rewrite_prompt_anchors_original_question(self):
        from agent import AgenticRAG, AnswerGrader, QueryRouter

        llm = ScriptedLLM(
            router_response="BROAD",
            grader_responses=[
                '{"score": 0.1, "reason": "bad"}',
                '{"score": 0.9, "reason": "ok"}',
            ],
            rewrite_response="rewritten",
        )
        retrieve_fn, _ = _make_retrieve_fn()
        generate_fn, _ = _make_generate_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
        )
        agent.run("original question", chat_history=[])

        rewrite_calls = [c for c in llm.calls if c["kind"] == "rewrite"]
        assert len(rewrite_calls) == 1
        assert "original question" in rewrite_calls[0]["prompt"]
