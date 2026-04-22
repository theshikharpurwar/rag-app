# AGENTS.md

> Onboarding file for AI coding agents (and humans). Read this first before making changes.
> Full architecture, product backlog, and sprint artefacts live in [README_v2.md](README_v2.md).

---

## 1. What this project is

A **Neuro-Symbolic Agentic RAG** system for local document Q&A. Three layers:

1. **Agentic control loop** (query routing, self-correction, optional decomposition + RAG-Fusion) — *shipped, opt-in via env*.
2. **Hybrid knowledge store** — Qdrant vectors + NetworkX knowledge graph.
3. **Fusion retrieval** — Weighted RRF over vector + BM25 + graph traversal.

Runs entirely locally on 8GB RAM: Ollama (`gemma3:4b` + `nomic-embed-text-v2-moe`), Qdrant, MongoDB, React frontend, Node/Express backend, Python ML scripts. No cloud APIs.

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

---

## 3. Where to resume — next task

**Phase 4 (remaining): knowledge graph visualisation** — LLM-as-Judge + naive vs neuro-symbolic benchmark + reports are in [python/evaluation/](python/evaluation/) (see [README_v2.md](README_v2.md) roadmap).

Agent stack (see §2): [python/agent/](python/agent/) — `prompts.py`, `query_router.py`, `grader.py`, `decomposer.py`, `rag_fusion.py`, `control_loop.py`. Wired in [python/local_llm.py](python/local_llm.py) `main()`: when `ENABLE_AGENT=true`, `AgenticRAG.run(...)` replaces the single-shot generate path; when `ENABLE_DECOMPOSITION=true` as well, complex queries are decomposed and retrieved per sub-query, then RRF-merged. JSON payload shape (`{answer, sources}`) is unchanged.

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
- `EVAL_MAX_CONCURRENCY` — int, default `1` (reserved; benchmark runs serially).

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
- [python/retrieval/](python/retrieval/) — `bm25_search.py`, `entity_extractor.py`, `knowledge_graph.py`, `graph_retrieval.py`, `rrf_fusion.py`.
- [python/embeddings/ollama_embed.py](python/embeddings/ollama_embed.py) — `OllamaEmbedder` wrapper.
- [python/llm/ollama_llm.py](python/llm/ollama_llm.py) — `OllamaLLM` wrapper (uses `/api/chat`, structured messages).
- [python/reranker/](python/reranker/) — optional cross-encoder reranker.
- [python/config/models.py](python/config/models.py) — **all tunable knobs**. Read this before changing behaviour.
- [python/agent/](python/agent/) — Phase 3–3.5: `prompts.py`, `query_router.py`, `grader.py`, `decomposer.py`, `rag_fusion.py`, `control_loop.py`.
- [python/tests/test_phase2.py](python/tests/test_phase2.py), [python/tests/test_phase3.py](python/tests/test_phase3.py), [python/tests/test_phase4.py](python/tests/test_phase4.py) — pytest suites.

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
| `LLM_MODEL` | `gemma3:4b` | Ollama model name for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text-v2-moe` | 768-dim embedder |
| `ENABLE_KNOWLEDGE_GRAPH` | `true` | Toggle KG extraction during ingest (expensive — many LLM calls) |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` / `GRAPH_WEIGHT` | 0.4 / 0.3 / 0.3 | RRF fusion weights (α, β, γ) |
| `RRF_K` | 60 | RRF smoothing constant |
| `GRAPH_TRAVERSAL_DEPTH` | 2 | BFS depth in graph retrieval |
| `SKIP_RERANKING` | `true` | Skip cross-encoder to save ~250MB RAM |
| `USE_DOCLING` | `false` | Use IBM Docling instead of PyMuPDF4LLM |
| `INDICES_DIR` | `/app/uploads/indices` | Where BM25 `.pkl` and KG `.json` are persisted (per `pdf_id`) |
| `OLLAMA_HOST_URL` | `http://host.containers.internal:11434` | Must be reachable from inside container |
| `ENABLE_AGENT` | `false` | Phase 3 master toggle. When true, `local_llm.py` runs the agentic loop instead of a single generate call |
| `AGENT_MAX_RETRIES` | `2` | Max extra retrieve+generate+grade cycles after the first attempt |
| `AGENT_CONFIDENCE_THRESHOLD` | `0.5` | Grader score below which a retry is triggered |
| `AGENT_ROUTER_WEIGHT_BOOST` | `0.15` | How much the router shifts RRF weight between BM25 and graph |
| `ENABLE_DECOMPOSITION` | `false` | When `ENABLE_AGENT=true`, run query decomposition + RAG-Fusion for complex queries |
| `AGENT_DECOMP_MAX_SUBQUERIES` | `3` | Max sub-questions from the decomposer LLM (original query is always retrieved too) |
| `AGENT_DECOMP_MIN_WORDS` | `8` | Below this word count (and no multi-part triggers), skip decomposition |
| `AGENT_FUSION_RRF_K` | `60` | RRF smoothing for merging results across sub-queries |
| `JUDGE_LLM_MODEL` | same as `LLM_MODEL` | Ollama model for Phase 4 LLM-as-Judge |
| `EVAL_OUTPUT_DIR` | `python/evaluation/reports` (from package path) | Where benchmark JSON + markdown are written |
| `EVAL_JUDGE_TEMPERATURE` | `0.0` | Judge sampling temperature |
| `EVAL_MAX_CONCURRENCY` | `1` | Placeholder; benchmarks run one question at a time |

---

## 7. Run / test / debug

```bash
# Start everything (needs Ollama already running on host with models pulled)
docker compose up --build -d

# Watch backend + Python logs (most debugging happens here)
docker compose logs -f backend

# Run Phase 2 + 3 + 4 unit tests
cd python && python -m pytest tests/test_phase2.py tests/test_phase3.py tests/test_phase4.py -v

# Phase 4 benchmark (needs Ollama + Qdrant; ingests dataset fixture when not skipped)
cd python && python -m evaluation.run_benchmark --dataset evaluation/datasets/sample.yaml

# Reset Qdrant + Mongo + uploads (for clean re-ingestion)
# Via API:
curl -X POST http://localhost:5000/api/reset
# Or Qdrant directly:
curl -X DELETE http://localhost:6333/collections/documents

# Required Ollama models (on host)
ollama pull gemma3:4b
ollama pull nomic-embed-text-v2-moe
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
7. **KG extraction is the slowest step** on ingest (many LLM calls per chunk). The in-progress `batch_size=3` change in `entity_extractor.py` is the fix — keep it.
8. **Agent loop inflates LLM calls.** With defaults (`AGENT_MAX_RETRIES=2`), worst-case ≈ **1 router + 3×(decompose + generate + grade) + 2 rewrites** (up to **12** LLM calls when decomposition runs every attempt). Without `ENABLE_DECOMPOSITION`, decompose calls are skipped. Tail latency on gemma3:4b can reach many minutes on the exhaust path. `ENABLE_AGENT=false` by default — flip it on once you're OK with the cost.

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
