"""
Phase 6: reranker default-on and fused-path reranking behavior.
"""
import importlib
import os
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import local_llm


class _DummyEmbedder:
    def encode_text(self, texts, task_type="search_query"):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _SpyReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, top_k=5):
        self.calls.append((query, len(documents), top_k))
        ranked = sorted(
            documents,
            key=lambda d: d.get("score", 0.0) if isinstance(d, dict) else getattr(d, "score", 0.0),
            reverse=True,
        )
        return ranked[:top_k]


def _install_fake_retrieval_modules(monkeypatch, *, bm25_load=True):
    bm25_mod = types.ModuleType("retrieval.bm25_search")

    class _FakeBM25:
        def load(self, _):
            return bm25_load

    bm25_mod.BM25Index = _FakeBM25

    rrf_mod = types.ModuleType("retrieval.rrf_fusion")

    def hybrid_retrieve(**kwargs):
        top_k = kwargs["top_k"]
        return [
            {
                "text": f"doc-{i}",
                "page": i,
                "source": "fixture.pdf",
                "score": float(top_k - i),
                "retrieval_methods": ["vector", "bm25"],
            }
            for i in range(top_k)
        ]

    rrf_mod.hybrid_retrieve = hybrid_retrieve

    monkeypatch.setitem(sys.modules, "retrieval.bm25_search", bm25_mod)
    monkeypatch.setitem(sys.modules, "retrieval.rrf_fusion", rrf_mod)


def test_fused_path_reranks_fused_topm(monkeypatch):
    _install_fake_retrieval_modules(monkeypatch, bm25_load=True)
    monkeypatch.setattr(local_llm, "ENABLE_KNOWLEDGE_GRAPH", False)
    monkeypatch.setattr(local_llm, "RERANK_TOP_M", 7)
    monkeypatch.setattr(local_llm, "CONTEXT_RETRIEVAL_LIMIT", 5)
    monkeypatch.setattr(local_llm, "embedder", _DummyEmbedder())

    reranker = _SpyReranker()
    monkeypatch.setattr(local_llm, "reranker", reranker)
    monkeypatch.setattr(local_llm, "get_qdrant_client", lambda: object())

    seen = {}

    def fake_retrieve_context(client, collection_name, query, pdf_id_filter, limit=5, rerank=True):
        seen["limit"] = limit
        seen["rerank"] = rerank
        return [{"text": f"vec-{i}", "score": i} for i in range(limit)]

    monkeypatch.setattr(local_llm, "retrieve_context", fake_retrieve_context)

    retrieve_fn, _ = local_llm.make_retrieve_pipeline("pdf-1")
    ctx, srcs = retrieve_fn("what is this", (0.4, 0.3, 0.3))

    assert seen["limit"] == 7
    assert seen["rerank"] is False
    assert reranker.calls and reranker.calls[0][1] == 7 and reranker.calls[0][2] == 5
    assert len(srcs) == 5
    assert "CONTENT FROM SOURCE 1" in ctx


def test_vector_only_path_still_reranks_in_retrieve_context(monkeypatch):
    monkeypatch.setattr(local_llm, "embedder", _DummyEmbedder())
    spy = _SpyReranker()
    monkeypatch.setattr(local_llm, "reranker", spy)

    class _Client:
        def query_points(self, **kwargs):
            points = [
                SimpleNamespace(payload={"text": f"t{i}"}, score=float(i)) for i in range(kwargs["limit"])
            ]
            return SimpleNamespace(points=points)

    out = local_llm.retrieve_context(
        _Client(),
        "documents",
        "query",
        "pdf-1",
        limit=3,
        rerank=True,
    )
    assert len(out) == 3
    assert spy.calls


def test_hybrid_path_does_not_double_rerank(monkeypatch):
    _install_fake_retrieval_modules(monkeypatch, bm25_load=True)
    monkeypatch.setattr(local_llm, "ENABLE_KNOWLEDGE_GRAPH", False)
    monkeypatch.setattr(local_llm, "RERANK_TOP_M", 6)
    monkeypatch.setattr(local_llm, "CONTEXT_RETRIEVAL_LIMIT", 5)
    monkeypatch.setattr(local_llm, "embedder", _DummyEmbedder())
    monkeypatch.setattr(local_llm, "reranker", _SpyReranker())
    monkeypatch.setattr(local_llm, "get_qdrant_client", lambda: object())

    calls = []

    def fake_retrieve_context(*args, **kwargs):
        calls.append(kwargs.get("rerank"))
        limit = kwargs.get("limit", 5)
        return [{"text": f"vec-{i}", "score": float(i)} for i in range(limit)]

    monkeypatch.setattr(local_llm, "retrieve_context", fake_retrieve_context)
    retrieve_fn, _ = local_llm.make_retrieve_pipeline("pdf-2")
    retrieve_fn("q", (0.4, 0.3, 0.3))

    assert calls == [False]


def test_reranker_model_env_override(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    fake_st = types.ModuleType("sentence_transformers")

    created = {}

    class _FakeCrossEncoder:
        def __init__(self, name, max_length=512):
            created["name"] = name
            created["max_length"] = max_length

    fake_st.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    import config.models as models_module
    import config as config_module
    import reranker.simple_reranker as rerank_module

    importlib.reload(models_module)
    importlib.reload(config_module)
    importlib.reload(rerank_module)
    rr = rerank_module.SimpleReranker()

    assert rr.model_name == "BAAI/bge-reranker-v2-m3"
    assert created["name"] == "BAAI/bge-reranker-v2-m3"


def test_rerank_top_m_capped_at_150(monkeypatch):
    monkeypatch.setenv("RERANK_TOP_M", "500")
    import config.models as models_module

    importlib.reload(models_module)
    assert models_module.RERANK_TOP_M == 150
