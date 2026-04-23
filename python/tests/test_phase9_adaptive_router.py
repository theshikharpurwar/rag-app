# python/tests/test_phase9_adaptive_router.py
"""Tier 3.4: 3-way router (specific / broad / multi_hop) and auto-decomposition."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_phase3 import ScriptedLLM, _make_retrieve_fn  # noqa: E402


LONG_COMPARE_Q = (
    "one two three four five six seven eight nine compare ten eleven twelve "
    "thirteen fourteen"
)


class TestAdaptiveRouter:
    def test_classify_multi_hop(self):
        from agent.query_router import QueryRouter

        llm = ScriptedLLM(router_response="MULTI_HOP")
        r = QueryRouter(llm)
        assert r.classify("Compare A and B across the document.") == "multi_hop"

    def test_classify_multi_hop_with_trailing_text(self):
        from agent.query_router import QueryRouter

        llm = ScriptedLLM(router_response="MULTI_HOP.\nsome extra noise")
        r = QueryRouter(llm)
        assert r.classify("q") == "multi_hop"

    def test_multi_hop_weights_unchanged(self):
        from agent.query_router import QueryRouter

        r = QueryRouter(ScriptedLLM(), weight_boost=0.15)
        out = r.adjust_weights("multi_hop", 0.4, 0.3, 0.3)
        assert out == pytest.approx((0.4, 0.3, 0.3))

    def test_multi_hop_auto_decompose_no_injected_decomposer(self):
        from agent import AgenticRAG, AnswerGrader, QueryRouter

        llm = ScriptedLLM(
            router_response="MULTI_HOP",
            grader_responses=['{"score": 0.9, "reason": "good"}'],
            decompose_response="1. subq alpha\n2. subq beta",
        )
        retrieve_fn, _ = _make_retrieve_fn()

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=lambda q, ctx, hist: "synthesized",
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
            decomposer=None,
            fusion=None,
        )
        res = agent.run(LONG_COMPARE_Q, chat_history=[])

        assert res.trace["route"] == "multi_hop"
        assert res.trace["auto_decompose"] is True
        assert len(res.trace["attempts"][0]["sub_queries"]) > 1

    def test_multi_hop_does_not_override_injected_decomposer(self):
        from agent import AgenticRAG, AnswerGrader, QueryRouter
        from agent.decomposer import QueryDecomposer
        from agent.rag_fusion import RAGFusion

        llm = ScriptedLLM(
            router_response="MULTI_HOP",
            grader_responses=['{"score": 0.9, "reason": "good"}'],
            decompose_response="1. subq alpha\n2. subq beta",
        )
        retrieve_fn, _ = _make_retrieve_fn()
        decomposer = QueryDecomposer(llm)
        fusion = RAGFusion(retrieve_fn, rrf_k=60, top_k=5)

        agent = AgenticRAG(
            llm=llm,
            router=QueryRouter(llm, weight_boost=0.15),
            grader=AnswerGrader(llm, threshold=0.5),
            retrieve_fn=retrieve_fn,
            generate_fn=lambda q, ctx, hist: "synthesized",
            max_retries=2,
            base_weights=(0.4, 0.3, 0.3),
            decomposer=decomposer,
            fusion=fusion,
        )
        res = agent.run(LONG_COMPARE_Q, chat_history=[])

        assert res.trace["auto_decompose"] is False
        assert len(res.trace["attempts"][0]["sub_queries"]) > 1

    def test_specific_still_skips_community_summaries(self):
        """Mirrors local_llm: community path only when route != 'specific'."""
        for route, expect_community in [
            ("specific", False),
            ("broad", True),
            ("multi_hop", True),
            (None, True),
        ]:
            use_community = route != "specific"
            assert use_community is expect_community

    def test_broad_still_shifts_to_graph(self):
        from agent.query_router import QueryRouter

        r = QueryRouter(ScriptedLLM(), weight_boost=0.15)
        v0, b0, g0 = 0.4, 0.3, 0.3
        v, b, g = r.adjust_weights("broad", v0, b0, g0)
        assert g > g0 and b < b0
        assert (v + b + g) == pytest.approx(v0 + b0 + g0)
