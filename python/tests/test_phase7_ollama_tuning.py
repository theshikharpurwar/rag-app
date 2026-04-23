"""
Phase 7: per-request Ollama tuning (num_batch, num_ctx, keep_alive) on chat; keep_alive on embed.
"""

import importlib
import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _reload_config_modules_after_test():
    yield
    import config.models as models_module
    import embeddings.ollama_embed as oe
    import llm.ollama_llm as ollama_llm_module

    importlib.reload(models_module)
    importlib.reload(ollama_llm_module)
    importlib.reload(oe)


def _tags_response(model: str = "gemma3:4b"):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"models": [{"name": model}]}
    return m


def _stream_chat_response(content: str = "ok"):
    class _Resp:
        status_code = 200

        def iter_lines(self, decode_unicode=False):
            yield json.dumps(
                {
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                    "eval_count": 8,
                    "eval_duration": 2_000_000_000,
                }
            ).encode()

        def close(self):
            pass

    return _Resp()


def test_chat_payload_carries_options_and_keep_alive(monkeypatch):
    import llm.ollama_llm as ollama_llm_module

    captured = {}

    def fake_post(url, json=None, timeout=120, stream=False, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return _stream_chat_response()

    monkeypatch.setattr(ollama_llm_module.requests, "post", fake_post)
    monkeypatch.setattr(ollama_llm_module.requests, "get", lambda *a, **k: _tags_response())

    llm = ollama_llm_module.OllamaLLM(
        model_name="gemma3:4b",
        api_base="http://localhost:11434/api",
        num_batch=256,
        num_ctx=4096,
        keep_alive="10m",
    )
    out = llm.generate_response("ping", max_tokens=100, temperature=0.4)
    assert out == "ok"
    p = captured["json"]
    assert p["keep_alive"] == "10m"
    assert p["options"]["num_batch"] == 256
    assert p["options"]["num_ctx"] == 4096
    assert p["options"]["num_predict"] == 100
    assert p["options"]["temperature"] == 0.4


def test_chat_num_ctx_omitted_when_not_set(monkeypatch):
    import llm.ollama_llm as ollama_llm_module

    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        return _stream_chat_response()

    monkeypatch.setattr(ollama_llm_module.requests, "post", fake_post)
    monkeypatch.setattr(ollama_llm_module.requests, "get", lambda *a, **k: _tags_response())

    llm = ollama_llm_module.OllamaLLM(
        model_name="gemma3:4b",
        api_base="http://localhost:11434/api",
        num_batch=128,
        num_ctx=0,
        keep_alive="5m",
    )
    llm.generate_response("x", max_tokens=10, temperature=0.0)
    assert "num_ctx" not in captured["json"]["options"]


def test_config_env_overrides_reload(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_BATCH", "1024")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")

    import config.models as models_module
    import llm.ollama_llm as ollama_llm_module

    importlib.reload(models_module)
    importlib.reload(ollama_llm_module)

    assert models_module.OLLAMA_NUM_BATCH == 1024
    assert models_module.OLLAMA_NUM_CTX == 8192
    assert models_module.OLLAMA_KEEP_ALIVE == "30m"

    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        return _stream_chat_response()

    monkeypatch.setattr(ollama_llm_module.requests, "post", fake_post)
    monkeypatch.setattr(ollama_llm_module.requests, "get", lambda *a, **k: _tags_response())

    llm = ollama_llm_module.OllamaLLM(model_name="gemma3:4b", api_base="http://localhost:11434/api")
    llm.generate_response("z", max_tokens=5, temperature=0.1)
    assert captured["json"]["options"]["num_batch"] == 1024
    assert captured["json"]["options"]["num_ctx"] == 8192
    assert captured["json"]["keep_alive"] == "30m"


def test_config_num_ctx_zero_means_none_after_reload(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "0")
    import config.models as models_module
    import llm.ollama_llm as ollama_llm_module

    importlib.reload(models_module)
    importlib.reload(ollama_llm_module)
    assert models_module.OLLAMA_NUM_CTX is None

    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        return _stream_chat_response()

    monkeypatch.setattr(ollama_llm_module.requests, "post", fake_post)
    monkeypatch.setattr(ollama_llm_module.requests, "get", lambda *a, **k: _tags_response())

    llm = ollama_llm_module.OllamaLLM(model_name="gemma3:4b", api_base="http://localhost:11434/api")
    llm.generate_response("z", max_tokens=5, temperature=0.1)
    assert "num_ctx" not in captured["json"]["options"]


def test_embed_api_and_legacy_carry_keep_alive(monkeypatch):
    import embeddings.ollama_embed as oe

    bodies = []

    def fake_post(url, json=None, **kwargs):
        bodies.append((url, json))
        r = MagicMock()
        if url.endswith("/embed"):
            r.status_code = 200
            r.json.return_value = {"embeddings": [[0.0, 0.0, 0.0]]}
            r.raise_for_status = lambda: None
        else:
            r.status_code = 200
            r.json.return_value = {"embedding": [0.1, 0.2]}
        return r

    monkeypatch.setattr(oe.requests, "get", lambda *a, **k: _tags_response("m"))
    monkeypatch.setattr(oe.requests, "post", fake_post)

    emb = oe.OllamaEmbedder(model_name="m", api_base="http://localhost:9/api", keep_alive="4m")
    emb.encode_text(["only"], task_type="search_document")
    embed_urls = [u for u, _ in bodies if u.endswith("/embed")]
    assert embed_urls
    assert bodies[0][1]["keep_alive"] == "4m"

    bodies.clear()
    emb._encode_legacy(["a"], "search_document")
    assert bodies[0][1]["keep_alive"] == "4m"
