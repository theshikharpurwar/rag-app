# python/tests/test_phase11_colbert.py
"""Tier 3.6b: ColBERT-v2 reranker (mocked ragatouille)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "sentence_transformers" not in sys.modules:
    _st = MagicMock()
    _st.CrossEncoder = MagicMock()
    sys.modules["sentence_transformers"] = _st


class MockRAGModel:
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def rerank(self, query, documents, k):
        scored = sorted(enumerate(documents), key=lambda x: x[1])
        out = []
        for i, (_, doc) in enumerate(scored[:k]):
            out.append({"content": doc, "score": 1.0 - i * 0.1, "rank": i})
        return out


def _fake_ragatouille():
    return SimpleNamespace(RAGPretrainedModel=MockRAGModel)


def test_colbert_reranker_rerank_returns_top_k():
    import reranker.colbert_reranker as cr

    docs = [
        {"text": "zebra", "score": 0.1},
        {"text": "apple", "score": 0.9},
        {"text": "mango", "score": 0.5},
    ]
    with patch.dict(sys.modules, {"ragatouille": _fake_ragatouille()}):
        r = cr.ColBERTReranker(model_name="dummy")
    out = r.rerank("q", docs, top_k=2)
    assert len(out) == 2
    assert [d["text"] for d in out] == ["apple", "mango"]


def test_colbert_handles_dict_and_qdrant_formats():
    import reranker.colbert_reranker as cr

    class OrderModel:
        @classmethod
        def from_pretrained(cls, _):
            return cls()

        def rerank(self, query, documents, k):
            assert documents == ["from qdrant", "plain dict", "extra chunk"]
            return [
                {"content": "plain dict", "score": 1.0, "rank": 0},
                {"content": "from qdrant", "score": 0.5, "rank": 1},
            ]

    fake_rt = SimpleNamespace(RAGPretrainedModel=OrderModel)

    qdoc = SimpleNamespace(payload={"text": "from qdrant"})
    docs = [qdoc, {"text": "plain dict"}, {"text": "extra chunk"}]

    with patch.dict(sys.modules, {"ragatouille": fake_rt}):
        r = cr.ColBERTReranker(model_name="x")
    out = r.rerank("q", docs, top_k=2)
    assert out[0] is docs[1]
    assert out[1] is qdoc


def test_colbert_small_input_returns_all():
    import reranker.colbert_reranker as cr

    with patch.dict(sys.modules, {"ragatouille": _fake_ragatouille()}):
        r = cr.ColBERTReranker(model_name="dummy")
    docs = [{"text": "a"}, {"text": "b"}]
    assert r.rerank("q", docs, top_k=5) == docs


def test_get_reranker_factory_default(monkeypatch):
    monkeypatch.setattr("config.models.RERANKER_TYPE", "cross_encoder")
    from reranker import get_reranker

    with patch("reranker.simple_reranker.CrossEncoder", MagicMock()):
        r = get_reranker()
    from reranker.simple_reranker import SimpleReranker

    assert isinstance(r, SimpleReranker)


def test_get_reranker_factory_colbert_fallback(monkeypatch):
    monkeypatch.setattr("config.models.RERANKER_TYPE", "colbert")
    import reranker.colbert_reranker as cr

    class Bad:
        def __init__(self, *a, **k):
            raise ImportError("ragatouille unavailable")

    monkeypatch.setattr(cr, "ColBERTReranker", Bad)
    from reranker import get_reranker

    with patch("reranker.simple_reranker.CrossEncoder", MagicMock()):
        r = get_reranker()
    from reranker.simple_reranker import SimpleReranker

    assert isinstance(r, SimpleReranker)
