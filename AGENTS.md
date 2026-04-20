# AGENTS.md

> Onboarding file for AI coding agents (and humans). Read this first before making changes.
> Full architecture, product backlog, and sprint artefacts live in [README_v2.md](README_v2.md).

---

## 1. What this project is

A **Neuro-Symbolic Agentic RAG** system for local document Q&A. Three layers:

1. **Agentic control loop** (query routing, self-correction) — *not built yet*.
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
| **3. Agentic self-correction** | Query router, hallucination grader, query decomposition, RAG-Fusion, retry loop | **IN PROGRESS** | Router + grader + retry loop shipped (opt-in via `ENABLE_AGENT=true`). Tests: [python/tests/test_phase3.py](python/tests/test_phase3.py). Decomposer + RAG-Fusion deferred to **Phase 3.5**. |
| **4. Evaluation** | LLM-as-Judge, Naive vs Neuro-Symbolic benchmark, KG visualisation | **NOT STARTED** | No harness yet |

---

## 3. Where to resume — next task

**Phase 3.5: add query decomposition and RAG-Fusion to the existing agent loop.**

Phase 3 medium scope already shipped (see §2):

| File | Status | Notes |
|---|---|---|
| [python/agent/prompts.py](python/agent/prompts.py) | ✅ | `ROUTER_PROMPT`, `GRADER_PROMPT`, `REWRITE_PROMPT` |
| [python/agent/query_router.py](python/agent/query_router.py) | ✅ | `QueryRouter.classify` + `adjust_weights` (soft RRF bias) |
| [python/agent/grader.py](python/agent/grader.py) | ✅ | `AnswerGrader.grade` returns `GradeResult(score, reason, passed)` |
| [python/agent/control_loop.py](python/agent/control_loop.py) | ✅ | `AgenticRAG.run` — route → retrieve → generate → grade → rewrite → retry |
| `python/agent/decomposer.py` | ❌ TODO | Break complex queries into sub-queries |
| `python/agent/rag_fusion.py` | ❌ TODO | Multi-query retrieval + dedup/merge (Phase 3.5) |

**What's already wired:** in [python/local_llm.py](python/local_llm.py) `main()`, when `ENABLE_AGENT=true` the final generation step is replaced by `AgenticRAG.run(...)`. The JSON payload shape (`{answer, sources}`) is unchanged so backend/frontend need no edits. Retrieval is encapsulated in an inner `_retrieve_and_format(query, weights)` helper so a multi-query RAG-Fusion version can drop in by swapping the `retrieve_fn` constructor arg.

**Phase 3.5 plan:** add `decomposer.py` (LLM-based sub-query generator) and `rag_fusion.py` (call `retrieve_fn` once per sub-query, RRF-merge across sub-queries, dedup). Wire via a new `ENABLE_DECOMPOSITION` flag so it can be turned on independently.

Active env vars (all in [python/config/models.py](python/config/models.py)):
- `ENABLE_AGENT` — bool, default `false`. Master toggle.
- `AGENT_MAX_RETRIES` — int, default `2`.
- `AGENT_CONFIDENCE_THRESHOLD` — float, default `0.5`. Below this, grader triggers a retry.
- `AGENT_ROUTER_WEIGHT_BOOST` — float, default `0.15`. Soft RRF nudge from the router.

---

## 4. Uncommitted work at handoff — DO NOT CLOBBER

`git status` currently shows modifications that are in-progress perf work. Before editing these files, run `git diff` and preserve the intent:

| File | In-progress change |
|---|---|
| [python/retrieval/entity_extractor.py](python/retrieval/entity_extractor.py) | Added `batch_size=3` param and `_extract_batch()`: batches multiple chunks into one LLM call during KG construction to cut Ollama round-trips. |
| [python/retrieval/graph_retrieval.py](python/retrieval/graph_retrieval.py) | In `extract_query_entities()`: added keyword-match fast path against existing graph nodes; only falls back to LLM when keyword match finds nothing. |
| [README_v2.md](README_v2.md) | Minor edit |
| `python/retrieval/__pycache__/*.pyc`, `python/tests/__pycache__/*.pyc` | Stale bytecode tracked in git. Should be added to `.gitignore` (`__pycache__/`, `*.pyc`). |

Decide with the user before committing: either (a) commit these as a "perf: batch KG extraction + keyword-first graph entity match" commit, or (b) keep them as WIP while Phase 3 is built on top.

---

## 5. Code map

Only the files that matter. Start here.

### Python (ML / retrieval)
- [python/compute_embeddings.py](python/compute_embeddings.py) — ingestion pipeline: PDF → chunks → embeddings → Qdrant + BM25 + KG (see lines ~496–534 for Phase 2 wiring).
- [python/local_llm.py](python/local_llm.py) — query pipeline: embed → retrieve → RRF fuse → rerank → LLM. Lines ~496–578 do hybrid retrieval; line ~585 is the Phase 3 wiring point.
- [python/retrieval/](python/retrieval/) — `bm25_search.py`, `entity_extractor.py`, `knowledge_graph.py`, `graph_retrieval.py`, `rrf_fusion.py`.
- [python/embeddings/ollama_embed.py](python/embeddings/ollama_embed.py) — `OllamaEmbedder` wrapper.
- [python/llm/ollama_llm.py](python/llm/ollama_llm.py) — `OllamaLLM` wrapper (uses `/api/chat`, structured messages).
- [python/reranker/](python/reranker/) — optional cross-encoder reranker.
- [python/config/models.py](python/config/models.py) — **all tunable knobs**. Read this before changing behaviour.
- [python/agent/](python/agent/) — Phase 3 loop: `prompts.py`, `query_router.py`, `grader.py`, `control_loop.py`. Phase 3.5 (`decomposer.py`, `rag_fusion.py`) lives here.
- [python/tests/test_phase2.py](python/tests/test_phase2.py), [python/tests/test_phase3.py](python/tests/test_phase3.py) — pytest suites.

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

---

## 7. Run / test / debug

```bash
# Start everything (needs Ollama already running on host with models pulled)
docker compose up --build -d

# Watch backend + Python logs (most debugging happens here)
docker compose logs -f backend

# Run Phase 2 + 3 unit tests
cd python && python -m pytest tests/test_phase2.py tests/test_phase3.py -v

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
8. **Agent loop inflates LLM calls.** With defaults (`AGENT_MAX_RETRIES=2`), worst-case query = 1 route + 3×(generate + grade) + 2 rewrites = up to 9 LLM calls. Tail latency on gemma3:4b can hit 20–40s. `ENABLE_AGENT=false` by default for this reason — flip it on once you're OK with the cost.

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
