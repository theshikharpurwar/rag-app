# python/tests/test_phase8_graphrag.py
"""Tier 3.8: community summaries + global search + 4-way hybrid fusion."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class CommunitySummaryLLM:
    _n = 0

    def generate_response(self, prompt="", **kwargs):
        if "Summarize the following" in prompt:
            CommunitySummaryLLM._n += 1
            return f"Summary for community {CommunitySummaryLLM._n}: themes and links."
        return ""


class MockEmbedderSemantic:
    """Query aligned with first summary vector, orthogonal to second."""

    def encode_text(self, texts, task_type="search_document"):
        if task_type == "search_query":
            return [[1.0, 0.0, 0.0]]
        out = []
        for t in texts:
            if "first" in (t or "").lower():
                out.append([1.0, 0.0, 0.0])
            else:
                out.append([0.0, 1.0, 0.0])
        return out


def _sample_sources(n):
    return [
        {"page": 1, "source": "t.pdf", "chunk_index": i, "chunk_text": f"c{i}"}
        for i in range(n)
    ]


def test_community_summary_generation():
    from retrieval.knowledge_graph import KnowledgeGraph

    CommunitySummaryLLM._n = 0
    triples = [
        ("alpha", "rel", "beta"),
        ("gamma", "rel2", "delta"),
    ]
    src = _sample_sources(len(triples))
    kg = KnowledgeGraph()
    kg.build_from_triples(triples, src)
    kg._communities = {"alpha": 0, "beta": 0, "gamma": 1, "delta": 1}
    for n in kg.graph.nodes():
        kg.graph.nodes[n]["community_id"] = kg._communities[n]

    summaries = kg.generate_community_summaries(CommunitySummaryLLM(), embedder=None)
    assert len(summaries) >= 1
    for cid, entry in summaries.items():
        assert "summary" in entry
        assert entry["summary"]
        assert "entities" in entry


def test_community_summaries_persist_load(tmp_path):
    from retrieval.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.build_from_triples(
        [("a", "r", "b")],
        _sample_sources(1),
    )
    kg._communities = {"a": 0, "b": 0}
    kg._community_summaries = {
        0: {"summary": "S0", "entities": ["a", "b"], "embedding": [0.1, 0.2]},
    }
    p = tmp_path / "g.json"
    kg.save(str(p))

    kg2 = KnowledgeGraph()
    assert kg2.load(str(p))
    assert 0 in kg2.community_summaries
    assert kg2.community_summaries[0]["summary"] == "S0"
    assert kg2.community_summaries[0]["embedding"] == [0.1, 0.2]


def test_legacy_graph_no_summaries(tmp_path):
    from retrieval.knowledge_graph import KnowledgeGraph

    data = {
        "nodes": {"x": {"entity_type": "entity", "sources": []}},
        "edges": [],
        "communities": {},
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    kg = KnowledgeGraph()
    assert kg.load(str(p))
    assert kg.community_summaries == {}


def test_global_search_keyword_fallback():
    from retrieval.knowledge_graph import KnowledgeGraph
    from retrieval.graph_retrieval import GraphRetriever

    kg = KnowledgeGraph()
    kg._community_summaries = {
        0: {"summary": "machine learning neural networks", "entities": ["a", "b"]},
        1: {"summary": "cooking recipes pasta", "entities": ["c", "d"]},
    }
    gr = GraphRetriever(kg, llm=None, traversal_depth=1)
    out = gr.global_search("neural network learning", embedder=None, top_k=2)
    assert len(out) >= 1
    assert out[0]["retrieval_method"] == "community_summary"
    assert "neural" in out[0]["text"].lower() or "learning" in out[0]["text"].lower()


def test_global_search_semantic():
    from retrieval.knowledge_graph import KnowledgeGraph
    from retrieval.graph_retrieval import GraphRetriever

    kg = KnowledgeGraph()
    kg._community_summaries = {
        0: {"summary": "first community about topic A", "entities": ["a"], "embedding": [1.0, 0.0, 0.0]},
        1: {"summary": "second unrelated", "entities": ["b"], "embedding": [0.0, 1.0, 0.0]},
    }
    gr = GraphRetriever(kg, llm=None, traversal_depth=1)
    out = gr.global_search("topic A align", embedder=MockEmbedderSemantic(), top_k=2)
    assert len(out) >= 1
    assert "first" in out[0]["text"].lower()


def test_hybrid_retrieve_four_paths():
    from retrieval.rrf_fusion import hybrid_retrieve

    class MB25:
        def search(self, query, pdf_id=None, top_k=10):
            return [
                {
                    "text": "bm25 chunk text",
                    "page": 1,
                    "source": "doc",
                    "chunk_index": 0,
                    "score": 0.8,
                    "retrieval_method": "bm25",
                }
            ]

    hits = [
        SimpleNamespace(
            payload={"text": "vector chunk", "page": 1, "source": "doc", "chunk_index": 1},
            score=0.9,
        )
    ]

    kg = MagicMock()
    kg.num_nodes = 1

    gr = MagicMock()

    def _graph_score(q, top_k=10):
        return [
            {
                "text": "graph chunk",
                "page": 2,
                "source": "doc",
                "chunk_index": 2,
                "score": 0.7,
                "retrieval_method": "graph",
            }
        ]

    gr.score_chunks_by_graph.side_effect = _graph_score

    comm = [
        {
            "text": "community summary text",
            "page": "community",
            "source": "Community 0 (2 entities)",
            "chunk_index": 0,
            "score": 0.5,
            "retrieval_method": "community_summary",
        }
    ]

    fused = hybrid_retrieve(
        query="q",
        pdf_id="p1",
        vector_results=hits,
        bm25_index=MB25(),
        knowledge_graph=kg,
        graph_retriever=gr,
        vector_weight=0.25,
        bm25_weight=0.25,
        graph_weight=0.25,
        community_results=comm,
        community_weight=0.25,
        rrf_k=60,
        top_k=10,
    )
    assert len(fused) >= 1
    methods = set()
    for r in fused:
        methods.update(r.get("retrieval_methods", []))
    assert "community_summary" in methods


def test_community_weight_redistribution_math():
    v_w, b_w, g_w = 0.4, 0.3, 0.3
    community_w = 0.2
    scale = (1.0 - community_w) / max(v_w + b_w + g_w, 1e-9)
    v2, b2, g2 = v_w * scale, b_w * scale, g_w * scale
    assert abs(v2 + b2 + g2 + community_w - 1.0) < 1e-9
