# AGENTS.md

> Onboarding file for AI coding agents (and humans). Read this first before making changes.
> Full architecture, product backlog, and sprint artefacts live in [README_v2.md](README_v2.md).

---

## 1. What this project is

A **Neuro-Symbolic Agentic RAG** system for local document Q&A. Three layers:

1. **Agentic control loop** (query routing, self-correction, optional decomposition + RAG-Fusion) — *shipped, opt-in via env*.
2. **Hybrid knowledge store** — Qdrant vectors + NetworkX knowledge graph.
3. **Fusion retrieval** — Weighted RRF over vector + BM25 + graph traversal.

Runs entirely locally on 8GB RAM: Ollama, Qdrant, MongoDB, React frontend, Node/Express backend, and Python ML scripts. Current dev-compose defaults are `qwen3.5:0.8b` for generation and `nomic-embed-text:v1.5` for embeddings; `llm_triples` is still supported, but the practical local KG default is now regex-based `noun_phrase_cooccurrence`. No cloud APIs.

---

## 2. Implementation status — source of truth

Keep this table in sync with [README_v2.md](README_v2.md) when a phase changes.

| Phase | Scope | Status | Notes |
|---|---|---|---|
| **1. Foundation RAG** | PDF ingest, chunking, embedding, vector retrieval, LLM, chat history, Docker | **DONE** | Shipped and working |
| **2. Hybrid + Knowledge Graph** | BM25, entity extraction, NetworkX KG, Leiden communities, graph BFS, weighted RRF | **DONE** | Tests in [python/tests/test_phase2.py](python/tests/test_phase2.py). Uncommitted perf tweaks on disk (see §4). |
| **3. Agentic self-correction** | Query router, hallucination grader, retry + rewrite loop | **DONE** | Opt-in via `ENABLE_AGENT` (default `false`). Tests: [python/tests/test_phase3.py](python/tests/test_phase3.py). |
| **3.5. Multi-query retrieval** | Query decomposition + RAG-Fusion (RRF across sub-queries) | **DONE** | Opt-in via `ENABLE_DECOMPOSITION` (default `false`; requires `ENABLE_AGENT=true`). |
| **4. Evaluation** | LLM-as-Judge, Naive vs Neuro-Symbolic benchmark, report generation, KG visualisation | **BASELINE SHIPPED** | Harness: [python/evaluation/](python/evaluation/) — `run_benchmark` CLI, [python/tests/test_phase4.py](python/tests/test_phase4.py). First baseline report: [python/evaluation/reports/20260422T114922Z_report.md](python/evaluation/reports/20260422T114922Z_report.md) (n=10; neuro-symbolic 4 wins, naive 1, ties 5). **KG visualisation** still open. |
| **5. Optimization & depth** | Tier 2 (close Phase 4 scope) + Tier 3 (parallel eval, ablations, Adaptive-RAG, DBSF/LTR fusion, ColBERT, Ollama infra) | **PLANNED** | See [README_v2.md § Future Plans](README_v2.md#future-plans--tier-2--tier-3-post-phase-4-baseline) for the full table with research citations. |

---

## 3. Where to resume — next task

**Phase 4 baseline shipped.** LLM-as-Judge, naive-vs-neuro-symbolic runner, and Markdown + JSON reports are live in [python/evaluation/](python/evaluation/); first artifact is [python/evaluation/reports/20260422T114922Z_report.md](python/evaluation/reports/20260422T114922Z_report.md). Work queued for the next contributor is explicitly grouped as **Tier 2 / Tier 3 future plans** — see the full table in [README_v2.md § Future Plans](README_v2.md#future-plans--tier-2--tier-3-post-phase-4-baseline).

**Tier 2 — close Phase 4 scope (2.1–2.6 complete):**
- **2.1 KG visualisation** — ✅ completed via `python/evaluation/visualize_kg.py --pdf_id ...` (pyvis HTML + networkx/matplotlib PNG outputs).
- **2.2 Harder fixtures + distractor PDFs** — ✅ completed via `python/evaluation/datasets/harder.yaml` (15 curated questions) and benchmark ingest support for optional `distractor_pdfs` into the same collection.
- **2.3 Prompt-artifact cleanup** — ✅ completed (plain-prose system prompt + `_strip_leading_list_marker` in [python/local_llm.py](python/local_llm.py) `generate_rag_response`).
- **2.4 Ingest idempotency** — ✅ completed (`compute_embeddings.py --reset` deletes existing Qdrant points for the `pdf_id` before upsert; `evaluation/run_benchmark.py --reset` now forwards that flag during fixture ingest. Backend `/upload` still omits `--reset` — wire in a follow-up for UI re-upload idempotency).
- **2.5 Graph retrieval full chunk text** — ✅ completed (`chunk_text` now used in graph retrieval with `chunk_text_preview` fallback for legacy graph files).
- **2.6 Align eval neurosymbolic config with prod** — ✅ completed (evaluation `neurosymbolic` now respects `ENABLE_DECOMPOSITION`, and trace records the effective flag).
- **KG mode defaults + diagnostics** — ✅ completed: graph JSON now persists KG mode metadata plus compact chunk references; query-time graph loading skips incompatible mode files until re-ingest; ingest writes audit artifacts to `INDICES_DIR/diagnostics/{pdf_id}_kg_audit.json`.

**Tier 3 — scale & research depth (multi-day):** **3.1 parallel judge calls** — ✅ `EVAL_MAX_CONCURRENCY` wires up to three concurrent LLM judge calls per run in [python/evaluation/judge.py](python/evaluation/judge.py) (`ThreadPoolExecutor`); set `OLLAMA_NUM_PARALLEL` on the server for real speedup (see [DEPLOYMENT.md](DEPLOYMENT.md)). **3.4 Adaptive-RAG router** — ✅ 3-way `QueryRouter` (`specific` / `broad` / `multi_hop`): `multi_hop` keeps RRF weights balanced and **auto-enables** query decomposition + RAG-Fusion for that run even when `ENABLE_DECOMPOSITION=false`; trace includes `auto_decompose`; tests: [python/tests/test_phase9_adaptive_router.py](python/tests/test_phase9_adaptive_router.py). **3.3 statistical significance** — ✅ paired Wilcoxon + BCa bootstrap 95% CIs now render in [python/evaluation/report.py](python/evaluation/report.py), with `n < 20` automatically flagged as directional. **3.6a cross-encoder reranker default-on** — ✅ default `SKIP_RERANKING=false` in [docker-compose.yml](docker-compose.yml); reranking now applies on fused hybrid top-M in [python/local_llm.py](python/local_llm.py) (`RERANK_TOP_M`, default 50, capped at 150), with model selectable via `RERANKER_MODEL`; tests: [python/tests/test_phase6_rerank.py](python/tests/test_phase6_rerank.py). **3.6b ColBERT-v2 reranker** — ✅ optional `RERANKER_TYPE=colbert` via `get_reranker()` in [python/reranker/__init__.py](python/reranker/__init__.py) and [python/reranker/colbert_reranker.py](python/reranker/colbert_reranker.py) (ragatouille); `COLBERT_MODEL` env; falls back to cross-encoder if import/init fails; tests: [python/tests/test_phase11_colbert.py](python/tests/test_phase11_colbert.py). **3.7 Ollama infra tuning** — ✅ per-request `OLLAMA_NUM_BATCH`, optional `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE` in [python/llm/ollama_llm.py](python/llm/ollama_llm.py) and [python/embeddings/ollama_embed.py](python/embeddings/ollama_embed.py); bench CLI [python/evaluation/bench_ollama.py](python/evaluation/bench_ollama.py); host-side knobs + gemma3 KV-cache warning in [DEPLOYMENT.md](DEPLOYMENT.md); tests: [python/tests/test_phase7_ollama_tuning.py](python/tests/test_phase7_ollama_tuning.py). **3.9 batch ingest `/api/embed`** — ✅ [python/embeddings/ollama_embed.py](python/embeddings/ollama_embed.py) batches chunk texts via `POST /api/embed` (`EMBED_BATCH_SIZE`, default 32); legacy `/api/embeddings` per-item fallback on 404; tests: [python/tests/test_phase5_batch_embed.py](python/tests/test_phase5_batch_embed.py). **3.8 GraphRAG community summaries + global search** — ✅ opt-in `ENABLE_COMMUNITY_SUMMARIES` (default `false`): ingest-time LLM summary per Leiden community + summary embeddings in `_graph.json`; query-time `GraphRetriever.global_search` feeds a 4th RRF list when `route != "specific"` (agent specific-route skips); weight `COMMUNITY_SUMMARY_WEIGHT` (default `0.2`) scales vector/BM25/graph down proportionally; tests: [python/tests/test_phase8_graphrag.py](python/tests/test_phase8_graphrag.py). **3.5 DBSF fusion** — ✅ `FUSION_METHOD=dbsf` uses μ±3σ score normalization + weighted sum in [python/retrieval/dbsf_fusion.py](python/retrieval/dbsf_fusion.py); default `rrf` unchanged; [python/retrieval/rrf_fusion.py](python/retrieval/rrf_fusion.py) `hybrid_retrieve` dispatches; tests: [python/tests/test_phase10_dbsf.py](python/tests/test_phase10_dbsf.py). Remaining: ablation CLI (3.2). Rationale, research citations, and done-criteria live in [README_v2.md § Future Plans](README_v2.md#future-plans--tier-2--tier-3-post-phase-4-baseline).

Agent stack (see §2): [python/agent/](python/agent/) — `prompts.py`, `query_router.py` (3-way route + soft RRF nudge), `grader.py`, `decomposer.py`, `rag_fusion.py`, `control_loop.py`. Wired in [python/local_llm.py](python/local_llm.py) `main()`: when `ENABLE_AGENT=true`, `AgenticRAG.run(...)` replaces the single-shot generate path; when `ENABLE_DECOMPOSITION=true`, decomposition is injected for all eligible queries; when the router returns `multi_hop`, decomposition + fusion run for that query even if `ENABLE_DECOMPOSITION=false` (`trace.auto_decompose`). JSON payload shape (`{answer, sources}`) is unchanged.

Active env vars (all in [python/config/models.py](python/config/models.py)):
- `ENABLE_AGENT` — bool, default `false`. Master toggle for the agent loop.
- `AGENT_MAX_RETRIES` — int, default `2`.
- `AGENT_CONFIDENCE_THRESHOLD` — float, default `0.5`. Below this, grader triggers a retry.
- `AGENT_ROUTER_WEIGHT_BOOST` — float, default `0.15`. Soft RRF nudge from the router.
- `ENABLE_DECOMPOSITION` — bool, default `false`. Phase 3.5 multi-query retrieval (requires agent).
- `AGENT_DECOMP_MAX_SUBQUERIES` — int, default `3`. Cap on LLM-emitted sub-questions.
- `AGENT_DECOMP_MIN_WORDS` — int, default `8`. Skip decomposition when the query is shorter and has no multi-part triggers.
- `AGENT_FUSION_RRF_K` — int, default `60`. RRF `k` for merging ranked lists across sub-queries.
- `JUDGE_LLM_MODEL` — str, default same as `LLM_MODEL`. LLM-as-Judge for Phase 4 metrics.
- `EVAL_OUTPUT_DIR` — path, default `python/evaluation/reports` (from config file location if unset).
- `EVAL_JUDGE_TEMPERATURE` — float, default `0.0`. Judge calls use this temperature.
- `EVAL_MAX_CONCURRENCY` — int, default `1`. Caps concurrent Phase 4 judge LLM calls per run (`min(3, value)`); question/config loop stays serial. Raise only if Ollama can serve parallel generations (`OLLAMA_NUM_PARALLEL`); see [DEPLOYMENT.md](DEPLOYMENT.md).
- `OLLAMA_NUM_BATCH` — int, default `512`. Passed as Ollama `options.num_batch` on `/api/chat`.
- `OLLAMA_NUM_CTX` — int, default `0` (omit = model default). When set to a positive value, passed as `options.num_ctx` (smaller values truncate long prompts).
- `OLLAMA_KEEP_ALIVE` — str, default `5m`. Top-level `keep_alive` on `/api/chat` and embed requests.
- `EMBED_BATCH_SIZE` — int, default `32`. Max texts per Ollama `POST /api/embed` request in [python/embeddings/ollama_embed.py](python/embeddings/ollama_embed.py) (ingest + query embedding).
- `RERANKER_MODEL` — str, default `cross-encoder/ms-marco-MiniLM-L-6-v2`. Cross-encoder model used by [python/reranker/simple_reranker.py](python/reranker/simple_reranker.py); can be overridden (for example `BAAI/bge-reranker-v2-m3`).
- `RERANKER_TYPE` — str, default `cross_encoder`. Set `colbert` for late-interaction reranking (requires `ragatouille`; heavier model).
- `COLBERT_MODEL` — str, default `colbert-ir/colbertv2.0`. Pretrained id passed to ragatouille when `RERANKER_TYPE=colbert`.
- `RERANK_TOP_M` — int, default `50`, capped at `150`. Fused/vector candidate count before final top-`CONTEXT_RETRIEVAL_LIMIT` rerank in [python/local_llm.py](python/local_llm.py).
- `ENABLE_COMMUNITY_SUMMARIES` — bool, default `false`. Per-community LLM summaries during ingest (requires `ENABLE_KNOWLEDGE_GRAPH=true`).
- `COMMUNITY_SUMMARY_WEIGHT` — float, default `0.2`. RRF weight for the community-summary list when summaries exist and the query is not router-`specific`.
- `KG_EXTRACTION_MODE` — str, default `llm_triples` in config; dev compose overrides to `noun_phrase_cooccurrence`. Supported modes: `llm_triples`, `noun_phrase_cooccurrence`.
- `KG_NP_EXTRACTOR` — str, default `regex`. Fast-mode noun phrase extractor implementation.
- `KG_TRIPLE_BATCH_SIZE` — int, default `3`. Batch size for `llm_triples` extraction calls.
- `KG_EXTRACT_CONCURRENCY` — int, default `1`. Parallel extraction workers for `llm_triples`.
- `KG_EXTRACT_MAX_CHARS` — int, default `1200`. Max chunk characters passed to one `llm_triples` extraction prompt.
- `KG_STORAGE_PRETTY` — bool, default `false`. Pretty-print persisted `_graph.json` instead of compact storage.
- `KG_LLM_MODEL` — str, default same as `LLM_MODEL`. LLM used only for `llm_triples` extraction and community summaries.
- `FUSION_METHOD` — str, default `rrf`. Set `dbsf` for distribution-based score fusion in hybrid retrieve; `RRF_K` applies only to RRF.

---

## 4. Uncommitted work at handoff — DO NOT CLOBBER

Before editing retrieval or ingest code, run `git status` / `git diff`. If bytecode (`__pycache__`, `*.pyc`) appears tracked, prefer adding those patterns to `.gitignore` rather than committing churn.

---

## 5. Code map

Only the files that matter. Start here.

### Python (ML / retrieval)
- [python/compute_embeddings.py](python/compute_embeddings.py) — ingestion pipeline: PDF → chunks → embeddings → Qdrant + BM25 + KG (see lines ~496–534 for Phase 2 wiring).
- [python/local_llm.py](python/local_llm.py) — query pipeline: embed → retrieve → RRF fuse → rerank → LLM. Hybrid retrieval; Phase 3 / 3.5 agent wiring in `main()`; `init_runtime()` + `make_retrieve_pipeline()` for evaluation and subprocess use.
- [python/evaluation/](python/evaluation/) — Phase 4: `judge.py`, `runner.py`, `report.py`, `run_benchmark.py`, `datasets/`.
- [python/retrieval/](python/retrieval/) — `bm25_search.py`, `entity_extractor.py`, `knowledge_graph.py`, `graph_retrieval.py`, `rrf_fusion.py`, `dbsf_fusion.py`.
- [python/reranker/](python/reranker/) — `simple_reranker.py`, `colbert_reranker.py`, `get_reranker()` in `__init__.py`.
- [python/embeddings/ollama_embed.py](python/embeddings/ollama_embed.py) — `OllamaEmbedder` wrapper.
- [python/llm/ollama_llm.py](python/llm/ollama_llm.py) — `OllamaLLM` wrapper (uses `/api/chat`, structured messages).
- [python/config/models.py](python/config/models.py) — **all tunable knobs**. Read this before changing behaviour.
- [python/agent/](python/agent/) — Phase 3–3.5: `prompts.py`, `query_router.py`, `grader.py`, `decomposer.py`, `rag_fusion.py`, `control_loop.py`.
- [python/tests/test_phase2.py](python/tests/test_phase2.py), [python/tests/test_phase3.py](python/tests/test_phase3.py), [python/tests/test_phase4.py](python/tests/test_phase4.py), [python/tests/test_phase5_batch_embed.py](python/tests/test_phase5_batch_embed.py), [python/tests/test_phase6_rerank.py](python/tests/test_phase6_rerank.py), [python/tests/test_phase7_ollama_tuning.py](python/tests/test_phase7_ollama_tuning.py), [python/tests/test_phase8_graphrag.py](python/tests/test_phase8_graphrag.py), [python/tests/test_phase9_adaptive_router.py](python/tests/test_phase9_adaptive_router.py), [python/tests/test_phase10_dbsf.py](python/tests/test_phase10_dbsf.py), [python/tests/test_phase11_colbert.py](python/tests/test_phase11_colbert.py) — pytest suites.

### Backend (Node/Express)
- [backend/routes/api.js](backend/routes/api.js) — `/upload`, `/query`, `/reset`, `/pdfs`. Spawns Python as subprocess; reads JSON from stdout.
- [backend/server.js](backend/server.js) — Express bootstrap.
- [backend/models/pdf.js](backend/models/pdf.js) — MongoDB schema.

### Frontend (React + Nginx)
- [frontend/src/components/](frontend/src/components/) — `ChatInterface.js` is the primary UI.
- [frontend/src/api.js](frontend/src/api.js) — axios client (`queryRAG(question, history)`, `uploadPDF(file)`).

### Orchestration
- [docker-compose.yml](docker-compose.yml) — dev stack (MongoDB, Qdrant, backend, frontend).
- [docker-compose.prod.yml](docker-compose.prod.yml), [docker-compose.ghcr.yml](docker-compose.ghcr.yml) — production variants.

---

## 6. Environment knobs worth knowing

Defined in [python/config/models.py](python/config/models.py) and surfaced through [docker-compose.yml](docker-compose.yml). Most likely to be changed:

| Variable | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `qwen3.5:0.8b` in dev compose (`qwen3.5:0.8b` in config) | Ollama model name for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text:v1.5` | 768-dim embedder (current dev default; override if preferred) |
| `ENABLE_KNOWLEDGE_GRAPH` | `true` | Toggle KG extraction during ingest (expensive — many LLM calls) |
| `KG_EXTRACTION_MODE` | `noun_phrase_cooccurrence` in dev compose (`llm_triples` in config) | Select KG extraction path: fast regex co-occurrence vs LLM triples |
| `KG_NP_EXTRACTOR` | `regex` | Fast-mode noun phrase extractor implementation |
| `KG_TRIPLE_BATCH_SIZE` | `3` | Batch size for `llm_triples` extraction |
| `KG_EXTRACT_CONCURRENCY` | `1` | Parallelism for `llm_triples` extraction |
| `KG_EXTRACT_MAX_CHARS` | `1200` | Max chunk text passed to one `llm_triples` prompt |
| `KG_STORAGE_PRETTY` | `false` | Pretty-print persisted `_graph.json` instead of compact JSON |
| `KG_LLM_MODEL` | same as `LLM_MODEL` | LLM used for `llm_triples` extraction/community summaries |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` / `GRAPH_WEIGHT` | 0.4 / 0.3 / 0.3 | Hybrid fusion weights (α, β, γ) for RRF or DBSF |
| `RRF_K` | 60 | RRF smoothing constant (used only when `FUSION_METHOD=rrf`) |
| `FUSION_METHOD` | `dbsf` in dev compose (`rrf` in config) | `rrf` (rank fusion) or `dbsf` (μ±3σ score normalization + weighted sum) |
| `GRAPH_TRAVERSAL_DEPTH` | 2 | BFS depth in graph retrieval |
| `ENABLE_COMMUNITY_SUMMARIES` | `false` | Ingest-time GraphRAG-style summaries per Leiden community (extra LLM + embed calls) |
| `COMMUNITY_SUMMARY_WEIGHT` | `0.2` | RRF weight for community-summary retrieval when enabled and not agent-`specific` |
| `SKIP_RERANKING` | `false` | Disable cross-encoder reranking when memory-constrained; default keeps reranking on |
| `USE_DOCLING` | `false` | Use IBM Docling instead of PyMuPDF4LLM |
| `INDICES_DIR` | `/app/backend/uploads/indices` in dev compose (`/app/uploads/indices` in config) | Where BM25 `.pkl`, KG `.json`, and `diagnostics/{pdf_id}_kg_audit.json` are persisted |
| `OLLAMA_HOST_URL` | `http://host.containers.internal:11434` | Must be reachable from inside container |
| `OLLAMA_NUM_BATCH` | `512` | Ollama `options.num_batch` on `/api/chat` |
| `OLLAMA_NUM_CTX` | `0` (unset) | Positive value sets `options.num_ctx`; default leaves model context size |
| `OLLAMA_KEEP_ALIVE` | `5m` | `keep_alive` on `/api/chat` and `/api/embed` |
| `ENABLE_AGENT` | `true` in dev compose (`false` in config) | Phase 3 master toggle. When true, `local_llm.py` runs the agentic loop instead of a single generate call |
| `AGENT_MAX_RETRIES` | `2` | Max extra retrieve+generate+grade cycles after the first attempt |
| `AGENT_CONFIDENCE_THRESHOLD` | `0.5` | Grader score below which a retry is triggered |
| `AGENT_ROUTER_WEIGHT_BOOST` | `0.15` | How much the router shifts RRF weight between BM25 and graph |
| `ENABLE_DECOMPOSITION` | `true` in dev compose (`false` in config) | When `ENABLE_AGENT=true`, run query decomposition + RAG-Fusion for complex queries |
| `AGENT_DECOMP_MAX_SUBQUERIES` | `3` | Max sub-questions from the decomposer LLM (original query is always retrieved too) |
| `AGENT_DECOMP_MIN_WORDS` | `8` | Below this word count (and no multi-part triggers), skip decomposition |
| `AGENT_FUSION_RRF_K` | `60` | RRF smoothing for merging results across sub-queries |
| `JUDGE_LLM_MODEL` | same as `LLM_MODEL` | Ollama model for Phase 4 LLM-as-Judge |
| `EVAL_OUTPUT_DIR` | `python/evaluation/reports` (from package path) | Where benchmark JSON + markdown are written |
| `EVAL_JUDGE_TEMPERATURE` | `0.0` | Judge sampling temperature |
| `EVAL_MAX_CONCURRENCY` | `1` | Max concurrent judge LLM calls per run (1–3); benchmark loop over questions/configs stays serial |
| `EMBED_BATCH_SIZE` | `32` | Max chunk texts per Ollama `POST /api/embed` request ([python/embeddings/ollama_embed.py](python/embeddings/ollama_embed.py)) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking (`BAAI/bge-reranker-v2-m3` supported as an override) |
| `RERANKER_TYPE` | `cross_encoder` | `cross_encoder` (default) or `colbert` (ragatouille / ColBERT-v2) |
| `COLBERT_MODEL` | `colbert-ir/colbertv2.0` | HuggingFace pretrained id when `RERANKER_TYPE=colbert` |
| `RERANK_TOP_M` | `50` (cap `150`) | Candidates kept before final rerank to top-5 in [python/local_llm.py](python/local_llm.py) |

---

## 7. Run / test / debug

```bash
# Start everything (needs Ollama already running on host with models pulled)
docker compose up --build -d

# Watch backend + Python logs (most debugging happens here)
docker compose logs -f backend

# Run Phase 2 + 3 + 4 unit tests
cd python && python -m pytest tests/test_phase2.py tests/test_phase3.py tests/test_phase4.py tests/test_phase5_batch_embed.py tests/test_phase6_rerank.py tests/test_phase7_ollama_tuning.py tests/test_phase8_graphrag.py tests/test_phase9_adaptive_router.py tests/test_phase10_dbsf.py tests/test_phase11_colbert.py -v

# Phase 4 benchmark (needs Ollama + Qdrant; ingests dataset fixture when not skipped)
cd python && python -m evaluation.run_benchmark --dataset evaluation/datasets/sample.yaml

# Reset Qdrant + Mongo + uploads (for clean re-ingestion)
# Via API:
curl -X POST http://localhost:5000/api/reset
# Or Qdrant directly:
curl -X DELETE http://localhost:6333/collections/documents

# Required Ollama models (on host)
ollama pull qwen3.5:0.8b
ollama pull nomic-embed-text:v1.5
```

Endpoints: frontend `:3000`, backend `:5000`, Qdrant dashboard `:6333/dashboard`.

---

## 8. Gotchas / sharp edges

1. **Subprocess IPC uses stdout.** The Node backend parses Python stdout as JSON. **Never `print()` to stdout** from Python scripts invoked as subprocesses (compute_embeddings.py, local_llm.py, qdrant_utils.py). Always write diagnostics to stderr: `print(..., file=sys.stderr)`. Libraries that print (pymupdf4llm, docling) must be muted or redirected. This has broken in the past.
2. **Qdrant API.** Use `client.query_points(...)`, not the deprecated `client.search(...)`.
3. **Ollama `:latest` tag.** Model availability check in `check_dependencies()` must accept both `name` and `name:latest`.
4. **Hybrid graceful degradation.** At query time, `local_llm.py` tries to load `{pdf_id}_bm25.pkl` and `{pdf_id}_graph.json` from `INDICES_DIR`. If either/both are missing (e.g. legacy PDF ingested before Phase 2), retrieval falls back cleanly to vector-only. Do not remove these try/except paths.
5. **Ollama connectivity from containers.** Must use `host.containers.internal` (Podman/Docker Desktop). Plain `localhost` will not reach the host Ollama.
6. **Vector-size coupling.** Qdrant collection `documents` is created with `DEFAULT_VECTOR_SIZE` from config. Changing `EMBEDDING_MODEL` to a different-dimension model requires deleting and recreating the collection (old vectors are incompatible).
7. **KG mode matters more than batching now.** `llm_triples` is still supported, but on small local models it is expensive and can add tens of seconds per ingest. The current local default is `KG_EXTRACTION_MODE=noun_phrase_cooccurrence` with `KG_NP_EXTRACTOR=regex`, which is dramatically faster but noisier. Turn `ENABLE_COMMUNITY_SUMMARIES=true` only when you want the extra recall enough to justify the ingest-time LLM + embed cost.
8. **Graph mode metadata is persisted.** `_graph.json` now stores KG mode metadata plus compact chunk references. Query-time graph loading skips incompatible graph files until the PDF is re-ingested in the active mode.
9. **KG audits are always nearby.** Ingest writes `INDICES_DIR/diagnostics/{pdf_id}_kg_audit.json` with timings, counts, and mode-specific extraction details. Use that before guessing about KG quality or latency.
10. **Reset is recursive now.** `/api/reset` clears upload subdirectories recursively before resetting Qdrant/Mongo, so stale `EISDIR` errors from unlinking `images/` or `indices/` should be gone after a rebuild.
11. **Agent loop inflates LLM calls.** With defaults (`AGENT_MAX_RETRIES=2`), worst-case ≈ **1 router + 3×(decompose + generate + grade) + 2 rewrites** (up to **12** LLM calls when decomposition runs every attempt). Without `ENABLE_DECOMPOSITION`, decompose calls are skipped. Tail latency on small local models can still reach many minutes on the exhaust path. Tune the dev-compose defaults before assuming this is a bug.

---

## 9. Conventions

- **No stdout noise** in Python subprocess entry points (see gotcha #1).
- **No comments that narrate the code.** Only add comments that explain *why*, not *what*.
- **Config centralisation.** Every new knob goes in [python/config/models.py](python/config/models.py) with an `os.environ.get(...)` default. Do not scatter `os.environ` reads across the codebase.
- **When completing a phase:** update the status table in §2 of this file AND the "Current Implementation Status" / "Roadmap" sections of [README_v2.md](README_v2.md) together.
- **Tests:** each phase gets its own `test_phaseN.py` under [python/tests/](python/tests/) with a `MockLLM` for deterministic output.
- **Commits:** Conventional commits style (`feat:`, `fix:`, `refactor:`, `docs:`, `perf:`). Match what's already in `git log`.

---

## 10. Canonical references

- **Full architecture, product backlog, sprint activities, test cases:** [README_v2.md](README_v2.md)
- **Research framing / IP:** [PROVISIONAL_PATENT_SPECIFICATION_v2.md](PROVISIONAL_PATENT_SPECIFICATION_v2.md)
- **Deployment / prod stack:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **Patent filing workflow:** [patent-guides/](patent-guides/)

---

*Last reviewed: the checklist above reflects the repo state at the time of the last agent handoff. If §2 is older than a week and code has changed, re-verify before trusting it.*
