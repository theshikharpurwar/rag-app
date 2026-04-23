# python/tests/test_phase5_batch_embed.py
"""Phase 3.9: OllamaEmbedder batch /api/embed with legacy fallback."""
import os
import sys
from unittest.mock import patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings.ollama_embed import OllamaEmbedder


class MockResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def _tags_ok():
    return MockResponse(200, {"models": [{"name": "test-model:latest"}]})


@patch("embeddings.ollama_embed.requests.get", return_value=_tags_ok())
def test_batch_embed_single_request(mock_get):
    dim = 4
    texts = [f"chunk-{i}" for i in range(20)]
    embeddings_payload = [[float(i)] * dim for i in range(20)]

    def post_side_effect(url, **kwargs):
        assert url.endswith("/embed")
        assert kwargs["json"]["input"] == texts
        assert kwargs["json"]["options"]["task"] == "search_document"
        return MockResponse(200, {"embeddings": embeddings_payload})

    with patch("embeddings.ollama_embed.requests.post", side_effect=post_side_effect) as mpost:
        e = OllamaEmbedder(
            model_name="test-model",
            api_base="http://localhost:11434/api",
            batch_size=32,
            zero_vector_dim=dim,
        )
        out = e.encode_text(texts, task_type="search_document")

    assert mpost.call_count == 1
    assert len(out) == 20
    assert out[0] == [0.0] * dim
    assert out[19] == [19.0] * dim


@patch("embeddings.ollama_embed.requests.get", return_value=_tags_ok())
def test_batch_embed_chunks_large_input(mock_get):
    dim = 2
    texts = [f"t{i}" for i in range(100)]
    call_batches = []

    def post_side_effect(url, **kwargs):
        assert url.endswith("/embed")
        inp = kwargs["json"]["input"]
        call_batches.append(list(inp))
        vecs = [[float(len(call_batches) - 1), float(j)] for j in range(len(inp))]
        return MockResponse(200, {"embeddings": vecs})

    with patch("embeddings.ollama_embed.requests.post", side_effect=post_side_effect) as mpost:
        e = OllamaEmbedder(
            model_name="test-model",
            api_base="http://localhost:11434/api",
            batch_size=32,
            zero_vector_dim=dim,
        )
        out = e.encode_text(texts)

    assert mpost.call_count == 4
    assert [len(b) for b in call_batches] == [32, 32, 32, 4]
    assert len(out) == 100


@patch("embeddings.ollama_embed.requests.get", return_value=_tags_ok())
def test_batch_embed_fallback_on_404(mock_get):
    dim = 3
    texts = ["a", "b"]
    legacy_calls = []

    def post_side_effect(url, **kwargs):
        if url.endswith("/embed"):
            return MockResponse(404, text="not found")
        if url.endswith("/embeddings"):
            legacy_calls.append(kwargs["json"]["prompt"])
            return MockResponse(200, {"embedding": [1.0, 2.0, 3.0]})
        raise AssertionError(f"unexpected url {url}")

    with patch("embeddings.ollama_embed.requests.post", side_effect=post_side_effect) as mpost:
        e = OllamaEmbedder(
            model_name="test-model",
            api_base="http://localhost:11434/api",
            batch_size=32,
            zero_vector_dim=dim,
        )
        out = e.encode_text(texts)

    assert out == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    embed_calls = [c for c in mpost.call_args_list if c[0][0].endswith("/embed")]
    legacy_post_calls = [c for c in mpost.call_args_list if c[0][0].endswith("/embeddings")]
    assert len(embed_calls) == 1
    assert len(legacy_post_calls) == 2
    assert legacy_calls == ["a", "b"]


@patch("embeddings.ollama_embed.requests.get", return_value=_tags_ok())
def test_batch_embed_malformed_response_pads_zeros(mock_get):
    dim = 2
    texts = ["a", "b", "c", "d"]
    bad = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    with patch(
        "embeddings.ollama_embed.requests.post",
        return_value=MockResponse(200, {"embeddings": bad}),
    ):
        e = OllamaEmbedder(
            model_name="test-model",
            api_base="http://localhost:11434/api",
            batch_size=32,
            zero_vector_dim=dim,
        )
        out = e.encode_text(texts)

    assert len(out) == 4
    assert out[3] == [0.0, 0.0]


@patch("embeddings.ollama_embed.requests.get", return_value=_tags_ok())
def test_encode_single_text_still_works(mock_get):
    dim = 5
    captured = {}

    def post_side_effect(url, **kwargs):
        captured.update(kwargs)
        assert kwargs["json"]["input"] == ["hello"]
        return MockResponse(200, {"embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]]})

    with patch("embeddings.ollama_embed.requests.post", side_effect=post_side_effect):
        e = OllamaEmbedder(
            model_name="test-model",
            api_base="http://localhost:11434/api",
            batch_size=32,
            zero_vector_dim=dim,
        )
        vec = e.encode_single_text("hello", task_type="search_query")

    assert vec == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert captured["json"]["options"]["task"] == "search_query"