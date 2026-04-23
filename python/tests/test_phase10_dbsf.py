# python/tests/test_phase10_dbsf.py
"""Tier 3.5 DBSF: distribution-based score fusion."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.dbsf_fusion import _normalize_scores, dbsf_fuse  # noqa: E402
from retrieval.rrf_fusion import hybrid_retrieve, rrf_fuse  # noqa: E402


def _expected_norm(scores: list[float]) -> list[float]:
    n = len(scores)
    if n < 2:
        return [0.5] * n
    mu = sum(scores) / n
    variance = sum((s - mu) ** 2 for s in scores) / n
    sigma = variance ** 0.5
    if sigma < 1e-9:
        return [0.5] * n
    lower = mu - 3 * sigma
    span = 6 * sigma
    return [max(0.0, min(1.0, (s - lower) / span)) for s in scores]


def test_normalize_scores_basic():
    scores = [0.1, 0.5, 0.9]
    assert _normalize_scores(scores) == pytest.approx(_expected_norm(scores))


def test_normalize_scores_identical():
    assert _normalize_scores([0.7, 0.7, 0.7]) == [0.5, 0.5, 0.5]


def test_normalize_scores_single_item():
    assert _normalize_scores([3.14]) == [0.5]


def test_normalize_scores_clamps():
    scores = [10.0] * 20 + [1000.0]
    out = _normalize_scores(scores)
    assert all(0.0 <= x <= 1.0 for x in out)
    assert max(out) == pytest.approx(1.0)


def _chunk(text, score, method, page=1, src="d.pdf", idx=0):
    return {
        "text": text,
        "page": page,
        "source": src,
        "chunk_index": idx,
        "score": score,
        "retrieval_method": method,
    }


def test_dbsf_fuse_two_lists():
    a = _chunk("alpha text", 0.9, "vector", idx=0)
    b = _chunk("beta text", 0.2, "bm25", idx=1)
    fused = dbsf_fuse([[a], [b]], weights=[0.5, 0.5], top_k=5)
    assert len(fused) == 2
    for r in fused:
        assert "retrieval_methods" in r
        assert "method_scores" in r
        assert isinstance(r["score"], float)


def test_dbsf_preserves_score_gap():
    low_first = _chunk("docA", 0.01, "vector", idx=0)
    high_second = _chunk("docB", 0.99, "vector", idx=1)
    ranked = [low_first, high_second]

    rrf_out = rrf_fuse([ranked], weights=[1.0], k=60, top_k=5)
    dbsf_out = dbsf_fuse([ranked], weights=[1.0], top_k=5)

    assert rrf_out[0]["text"] == "docA"
    assert dbsf_out[0]["text"] == "docB"


def test_hybrid_retrieve_dispatches_dbsf(monkeypatch):
    monkeypatch.setattr("config.models.FUSION_METHOD", "dbsf")

    class MB25:
        def search(self, query, pdf_id=None, top_k=10):
            return [_chunk("bm25 only", 2.0, "bm25", idx=0)]

    hits = [
        SimpleNamespace(
            payload={"text": "vec", "page": 1, "source": "d", "chunk_index": 0},
            score=0.5,
        )
    ]

    with patch("retrieval.dbsf_fusion.dbsf_fuse") as mock_dbsf:
        mock_dbsf.return_value = []
        hybrid_retrieve(
            query="q",
            pdf_id="p1",
            vector_results=hits,
            bm25_index=MB25(),
            knowledge_graph=None,
            graph_retriever=None,
            vector_weight=0.5,
            bm25_weight=0.5,
            graph_weight=0.0,
            rrf_k=60,
            top_k=5,
        )
        mock_dbsf.assert_called_once()
