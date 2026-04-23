
# Neuro-Symbolic Agentic RAG System

> **A Tri-Layer Cognitive Architecture for Intelligent Document Understanding**  
> Combining the structural precision of Knowledge Graphs (*Symbolic*), the semantic flexibility of Vector Embeddings (*Neural*), and the adaptive reasoning of Autonomous Agents — all running locally on consumer hardware (8GB RAM).

---

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Proposed Architecture](#proposed-architecture)
- [Current Implementation Status](#current-implementation-status)
- [Technical Pipeline](#technical-pipeline)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Configuration Reference](#configuration-reference)
- [Algorithmic Contributions](#algorithmic-contributions)
- [Evaluation Framework](#evaluation-framework)
- [Roadmap](#roadmap)
- [Product Backlog](#product-backlog)
- [Sprint Backlog](#sprint-backlog)
- [Sprint Activities](#sprint-activities)
  - [Daily Scrum](#daily-scrum)
  - [Functional Documents](#functional-documents)
  - [UI Design](#ui-design)
  - [Architecture Document](#architecture-document)
  - [Functional Test Case Document](#functional-test-case-document)
  - [Demo of Deliverables](#demo-of-deliverables)
  - [Sprint Retrospective](#sprint-retrospective)
- [Research References](#research-references)
- [License](#license)

---

## Abstract

Standard Retrieval-Augmented Generation (RAG) systems suffer from three fundamental weaknesses: they are **blind** (vector search misses exact keywords), **brittle** (they cannot connect facts across document sections), and **dumb** (when retrieval fails, they hallucinate instead of retrying). This project proposes and implements a **Neuro-Symbolic Agentic RAG** architecture that addresses all three limitations through a novel Tri-Layer Cognitive Architecture:

1. **Layer 1 — Agentic Control Loop ("The Brain")**: A self-correcting agent that routes queries to the optimal retrieval strategy, evaluates answer quality, and autonomously retries with rewritten queries when confidence is low.

2. **Layer 2 — Hybrid Knowledge Store ("The Memory")**: A dual-store combining Qdrant (vector similarity for unstructured semantics) with a Knowledge Graph (NetworkX for structured entity relationships, community detection, and multi-hop reasoning).

3. **Layer 3 — Graph-Augmented Fusion Retrieval ("The Algorithm")**: A custom Multi-Route Retrieval Algorithm that fuses vector cosine similarity (Path A: Fast) with graph traversal scoring (Path B: Deep) using Weighted Reciprocal Rank Fusion (RRF).

The system operates entirely locally on consumer hardware (8GB RAM minimum, no cloud APIs), making it suitable for privacy-sensitive document analysis in academic, legal, and enterprise contexts.

---

## Problem Statement

| Problem | Description | Our Solution |
|---|---|---|
| **Blind Retrieval** | Pure vector search misses exact keywords, dates, and numbers. Searching for "Section 3.2" returns semantically similar but wrong sections. | **Hybrid Search**: BM25 keyword matching fused with cosine vector similarity via RRF |
| **Brittle Context** | Standard RAG treats each chunk independently. It cannot "connect the dots" — e.g., linking a person mentioned on page 1 to their credentials on page 5. | **Knowledge Graphs**: Entity extraction builds a structured graph where multi-hop traversal discovers implicit relationships |
| **Dumb Failure** | When retrieval returns irrelevant context, standard RAG generates hallucinated answers with high confidence. No error detection or recovery. | **Agentic Self-Correction**: A Grader Agent evaluates answer faithfulness; if citations are insufficient, the query is decomposed or rewritten and retrieval retries |
| **Lossy Extraction** | Standard PDF text extraction destroys tables, headings, and document structure — critical for academic papers and reports. | **Structure-Preserving Extraction**: PyMuPDF4LLM renders PDFs as Markdown, preserving tables, headers, and lists for the LLM |

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: AGENTIC CONTROL LOOP                    │
│                                                                     │
│   ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐  │
│   │  Query    │───▶│  Retrieval │───▶│  Answer  │───▶│  Grader    │  │
│   │  Router   │    │  Strategy  │    │  Gen     │    │  Agent     │  │
│   └──────────┘    └───────────┘    └──────────┘    └─────┬──────┘  │
│        │                                                  │         │
│        │              ◀── RETRY with rewritten query ────┘         │
│        │                  (if confidence < threshold)               │
├────────┼────────────────────────────────────────────────────────────┤
│        ▼           LAYER 2: HYBRID KNOWLEDGE STORE                  │
│                                                                     │
│   ┌────────────────┐              ┌─────────────────────────┐      │
│   │  QDRANT        │              │  KNOWLEDGE GRAPH        │      │
│   │  Vector Store   │              │  (NetworkX/Neo4j)       │      │
│   │                │              │                         │      │
│   │  768-dim embeds │              │  Entity Nodes           │      │
│   │  Cosine search  │              │  Relation Edges         │      │
│   │  nomic-v2-moe  │              │  Leiden Communities     │      │
│   └───────┬────────┘              └───────────┬─────────────┘      │
│           │                                   │                     │
├───────────┼───────────────────────────────────┼─────────────────────┤
│           ▼      LAYER 3: FUSION RETRIEVAL    ▼                     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │            WEIGHTED RECIPROCAL RANK FUSION              │      │
│   │                                                         │      │
│   │  Score(d) = α · 1/(k + rank_vec) + (1-α) · 1/(k + rank_graph) │
│   │                                                         │      │
│   │  Path A (Fast): BM25 + Vector Cosine Similarity         │      │
│   │  Path B (Deep): Graph Traversal (BFS/Dijkstra)          │      │
│   └─────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Current Implementation Status

### ✅ Phase 1: Foundation RAG (Complete)

The core retrieval-augmented generation pipeline is fully implemented and operational:

| Component | Implementation | Details |
|---|---|---|
| **PDF Ingestion** | `pymupdf4llm` | Structure-preserving Markdown extraction (tables, headers, lists) |
| **Text Chunking** | Custom `chunk_text()` | Semantic chunking with paragraph/sentence boundary awareness and configurable overlap |
| **Embedding** | `nomic-embed-text-v2-moe` | 768-dimensional Mixture-of-Experts embeddings via Ollama (multilingual) |
| **Vector Store** | Qdrant | Cosine similarity search with pdf_id filtering and batch upsert |
| **LLM** | `gemma3:4b` | Google's 2025 multimodal model via Ollama `/api/chat` endpoint |
| **Chat History** | Structured messages | Real `user`/`assistant` role turns passed to the LLM (not raw text concatenation) |
| **Reranking** | Cross-encoder (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` for result refinement (toggle via `SKIP_RERANKING`) |
| **Frontend** | React SPA | PDF upload, real-time chat interface, conversation history management |
| **Backend** | Node.js/Express | REST API orchestrating Python ML scripts, MongoDB metadata, file management |
| **Containerization** | Docker Compose | Full orchestration: frontend (Nginx), backend (Node+Python), MongoDB, Qdrant |

### ✅ Phase 2: Hybrid Search & Knowledge Graphs (Complete)

| Component | Status | Description |
|---|---|---|
| BM25 Keyword Search | ✅ Implemented | Sparse retrieval via `rank_bm25` (BM25Okapi), per-document index |
| RRF Fusion | ✅ Implemented | Weighted Reciprocal Rank Fusion merging 3 retrieval paths (α=0.4, β=0.3, γ=0.3) |
| Entity Extraction | ✅ Implemented | LLM-powered `(Subject, Predicate, Object)` triple extraction with normalization |
| Knowledge Graph | ✅ Implemented | NetworkX directed graph with Leiden community detection (igraph fallback) |
| Graph Traversal | ✅ Implemented | BFS scoring from query entities, distance-weighted chunk ranking |
| Hybrid Pipeline | ✅ Implemented | Graceful degradation: falls back to vector-only for legacy PDFs |

### ✅ Phase 3: Agentic Self-Correction + Phase 3.5 Multi-Query Retrieval (Done)

Shipped behind `ENABLE_AGENT=true` (default `false`, so Phase 2 behaviour is preserved). Optional **Phase 3.5**: `ENABLE_DECOMPOSITION=true` adds LLM query decomposition and RAG-Fusion (RRF across sub-queries) inside each retry attempt.

| Component | Status | Description |
|---|---|---|
| Query Router | ✅ Implemented | `QueryRouter` 3-way: **Specific** (→ bias BM25), **Broad** (→ bias graph / community context), **Multi-hop** (balanced RRF weights; auto decomposition + RAG-Fusion for that query even when `ENABLE_DECOMPOSITION=false`, see `trace.auto_decompose`) |
| Hallucination Grader | ✅ Implemented | `AnswerGrader` returns 0.0–1.0 faithfulness+relevance score; parse failures safely default to pass |
| Self-Correction Loop | ✅ Implemented | `AgenticRAG.run()` orchestrates route → retrieve → generate → grade → rewrite → retry (capped by `AGENT_MAX_RETRIES`); rewrite prompt anchors the **original** user question |
| Query Decomposition | ✅ Implemented | `QueryDecomposer` breaks complex questions into sub-queries (heuristic skip for short/simple queries) |
| RAG-Fusion | ✅ Implemented | `RAGFusion` calls `retrieve_fn` per sub-query and RRF-merges chunk lists (Shi et al., 2024) |

Implementation: [python/agent/](python/agent/). Tests: [python/tests/test_phase3.py](python/tests/test_phase3.py), [python/tests/test_phase9_adaptive_router.py](python/tests/test_phase9_adaptive_router.py). Wired into [python/local_llm.py](python/local_llm.py) `main()` behind `ENABLE_AGENT` / `ENABLE_DECOMPOSITION`.

### Phase 4: Evaluation harness (baseline shipped — KG visualisation pending)

| Component | Status | Description |
|---|---|---|
| LLM-as-Judge | Shipped | Faithfulness, answer relevancy, context recall (plus deterministic conciseness) in [python/evaluation/judge.py](python/evaluation/judge.py) |
| Naive vs neuro-symbolic benchmark | Shipped | Vector-only vs hybrid + `AgenticRAG` + decomposition + RAG-Fusion in [python/evaluation/runner.py](python/evaluation/runner.py) (independent of `ENABLE_AGENT` env) |
| Report generation | Shipped | JSON + Markdown aggregates in [python/evaluation/report.py](python/evaluation/report.py) |
| CLI | Shipped | `python -m evaluation.run_benchmark` — see [python/evaluation/datasets/README.md](python/evaluation/datasets/README.md) |
| Knowledge graph visualisation | Not started | Follow-up |

Tests: [python/tests/test_phase4.py](python/tests/test_phase4.py).

**First baseline artifact** (`sample.yaml`, 10 questions × naive + neuro-symbolic on `gemma3:4b`):
[`python/evaluation/reports/20260422T114922Z_report.md`](python/evaluation/reports/20260422T114922Z_report.md)

| Metric | Naive | Neuro-Symbolic | Δ |
|---|---|---|---|
| faithfulness | 0.975 ± 0.042 | **0.990 ± 0.032** | **+0.015** |
| answer_relevancy | **0.970 ± 0.048** | 0.960 ± 0.052 | −0.010 |
| context_recall | 1.000 | 1.000 | tie |
| conciseness | 0.773 ± 0.272 | **0.800 ± 0.266** | **+0.027** |

**Win rate (higher mean of four metrics per question):** neuro-symbolic **4**, naive **1**, tie 5.
Neuro-symbolic dominates multi-hop (3/4 wins + 1 tie) and broad-summary questions — the advantage is most visible where naive's vector-only recall slipped to 2–3 sources while hybrid + fusion retrieved the top-5 limit. On a 5-page fixture `context_recall = 1.000` is structural rather than evidence of retrieval quality; larger/harder corpora are the next step. Results are directional (n=10, not statistically significant).

---

## Technical Pipeline

### Upload Pipeline (One-Time Per Document)

```
PDF File Upload
    │
    ▼
┌─────────────────────────┐
│  PyMuPDF4LLM             │  Structure-preserving extraction
│  to_markdown(page_chunks) │  Tables → Markdown tables
│                           │  Headers → # Markdown headers
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Semantic Chunker        │  Paragraph-aware splitting
│  chunk_text()            │  Sentence boundary respect
│  ~500 char chunks        │  Configurable overlap
│  with 100 char overlap   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  nomic-embed-text-v2-moe │  768-dimensional vectors
│  via Ollama API          │  MoE architecture
│  Batch encoding          │  task_type: search_document
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Qdrant Vector DB        │  Cosine similarity index
│  Batch upsert (50/batch) │  Metadata: page, source,
│  Collection: documents   │  chunk_index, extractor
└─────────────────────────┘
```

### Query Pipeline (Every User Question) — Phase 2 Hybrid

```
User Question + Chat History
    │
    ▼
┌─────────────────────────┐
│  nomic-embed-text-v2-moe │  Encode query
│  task_type: search_query │  768-dim vector
└───────────┬─────────────┘
            │
    ┌───────┴────────────────────────────┐
    │                                    │
    ▼ Path A (Neural)                   ▼ Path B (Keyword)
┌─────────────────────┐   ┌─────────────────────────┐
│  Qdrant Vector      │   │  BM25 Keyword Search     │
│  Cosine Similarity  │   │  rank_bm25 (BM25Okapi)   │
│  Top-20 candidates  │   │  pdf_id filtered          │
└────────┬────────────┘   └──────────┬──────────────┘
         │                           │
         │    ┌──────────────────┐    │
         │    │ Path C (Symbolic)│    │
         │    │ KG Graph Traversal│   │
         │    │ BFS from query   │    │
         │    │ entities         │    │
         │    └───────┬──────────┘    │
         │            │              │
         └────────────┼──────────────┘
                      ▼
       ┌──────────────────────────┐
       │  Weighted RRF Fusion     │
       │  α=0.4 (vector)          │
       │  β=0.3 (BM25)            │
       │  γ=0.3 (graph)           │
       │  k=60 smoothing          │
       └───────────┬──────────────┘
                   │
                   ▼  (optional, SKIP_RERANKING=false)
       ┌─────────────────────────┐
       │  Cross-Encoder Reranker  │
       │  Re-score by relevance   │
       └───────────┬─────────────┘
                   │
                   ▼
       ┌─────────────────────────────────────┐
       │  Structured Message Assembly         │
       │  [SYSTEM] RAG instructions + context │
       │  [USER/ASSISTANT] chat history       │
       │  [USER] current question             │
       └───────────┬─────────────────────────┘
                   │
                   ▼
       ┌─────────────────────────┐
       │  gemma3:4b via Ollama    │
       │  /api/chat endpoint      │
       │  Temperature: 0.7        │
       └───────────┬─────────────┘
                   │
                   ▼
               Frontend
           (React Chat UI)
```

---

## Technology Stack

### Core Infrastructure

| Layer | Technology | Purpose |
|---|---|---|
| **Containerization** | Docker Compose (Podman compatible) | Service orchestration |
| **Frontend** | React 18 + Nginx | Single-page application, API proxy |
| **Backend** | Node.js 18 + Express | REST API, process management |
| **ML Runtime** | Python 3.11 (venv) | Embedding, retrieval, generation scripts |

### AI/ML Models

| Model | Provider | Role | Size |
|---|---|---|---|
| `gemma3:4b` | Google (2025) | LLM for answer generation | ~3.3GB RAM |
| `nomic-embed-text-v2-moe` | Nomic AI | Text embedding (768-dim, multilingual) | ~1.2GB RAM |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Microsoft | Search result reranking | ~250MB RAM |

### Data Stores

| Store | Technology | Purpose |
|---|---|---|
| **Vector Database** | Qdrant | Embedding storage, cosine similarity search |
| **Document Database** | MongoDB | PDF metadata, file references |
| **Knowledge Graph** | NetworkX | Entity-relation graphs, community detection |

### Document Processing

| Library | Purpose |
|---|---|
| `pymupdf4llm` | Primary: Structure-preserving PDF → Markdown extraction |
| `PyMuPDF` (fitz) | Fallback: Image extraction from PDFs |
| `Docling` (IBM) | Optional: ML-powered layout analysis (toggle via `USE_DOCLING`) |

---

## Project Structure

```
rag-app/
├── docker-compose.yml          # Service orchestration (MongoDB, Qdrant, Backend, Frontend)
├── backend/
│   ├── Dockerfile              # Combined Node.js + Python environment
│   ├── server.js               # Express server, Python process spawner
│   ├── routes/
│   │   └── api.js              # REST endpoints: /upload, /query, /reset, /pdfs
│   └── models/
│       └── pdf.js              # MongoDB schema for PDF metadata
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatInterface.js    # Chat UI with history management
│   │   ├── api.js                  # Frontend API client (queryRAG with history)
│   │   └── App.js                  # Root component
│   └── Dockerfile              # React build + Nginx serve
├── python/
│   ├── config/
│   │   ├── models.py           # Centralized model configuration
│   │   └── __init__.py         # Config exports
│   ├── embeddings/
│   │   └── ollama_embed.py     # OllamaEmbedder: text embedding via Ollama API
│   ├── llm/
│   │   └── ollama_llm.py       # OllamaLLM: /api/chat structured message interface
│   ├── reranker/
│   │   └── simple_reranker.py  # Cross-encoder reranking (optional)
│   ├── agent/                  # Agentic control loop (query router, grader, retry)
│   ├── evaluation/             # Phase 4: LLM-as-Judge benchmark + reports
│   ├── retrieval/              # Hybrid retrieval algorithms (RRF, graph traversal)
│   ├── utils/
│   │   └── qdrant_utils.py     # Collection management utilities
│   ├── compute_embeddings.py   # PDF ingestion pipeline (extract → chunk → embed → store)
│   ├── local_llm.py            # Query pipeline (retrieve → rerank → generate)
│   └── requirements.txt        # Python dependencies
└── start.sh / start.bat        # Platform-specific launch scripts
```

---

## Installation & Setup

### Prerequisites

- **Docker** (or Podman) with Docker Compose
- **Ollama** installed and running on the host machine
- **8GB RAM minimum** (16GB recommended for full feature set)

### Step 1: Pull Required Models

```bash
# LLM model
ollama pull gemma3:4b

# Embedding model
ollama pull nomic-embed-text-v2-moe
```

### Step 2: Clone and Launch

```bash
git clone https://github.com/theshikharpurwar/rag-app.git
cd rag-app
docker compose up --build -d
```

### Step 3: Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### Step 4: Usage

1. Upload a PDF document through the web interface.
2. Wait for ingestion to complete (watch logs with `docker compose logs -f backend`).
3. Ask questions about the document in the chat interface.
4. The system maintains conversation history across turns.

---

## Configuration Reference

All configuration is centralized in `docker-compose.yml` and `python/config/models.py`:

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemma3:4b` | Ollama model for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text-v2-moe` | Ollama model for text embeddings |
| `OLLAMA_HOST_URL` | `http://host.containers.internal:11434` | Ollama API endpoint |
| `QDRANT_HOST` | `qdrant` | Qdrant service hostname |
| `QDRANT_PORT` | `6333` | Qdrant service port |
| `MONGODB_URI` | `mongodb://mongo:27017/rag_db` | MongoDB connection string |
| `USE_DOCLING` | `false` | Enable IBM Docling for ML-powered extraction |
| `SKIP_RERANKING` | `true` | Disable cross-encoder reranking (saves ~250MB RAM) |
| `SKIP_IMAGES` | `true` | Skip image embedding (text-only embedder) |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace model cache (Docker volume) |
| `ENABLE_KNOWLEDGE_GRAPH` | `true` | Toggle KG extraction during PDF ingestion (Phase 2) |
| `VECTOR_WEIGHT` | `0.4` | α — weight for vector similarity in RRF fusion |
| `BM25_WEIGHT` | `0.3` | β — weight for BM25 keyword search in RRF fusion |
| `GRAPH_WEIGHT` | `0.3` | γ — weight for graph traversal in RRF fusion |
| `RRF_K` | `60` | RRF smoothing constant |
| `GRAPH_TRAVERSAL_DEPTH` | `2` | BFS depth limit for knowledge graph traversal |
| `ENABLE_COMMUNITY_SUMMARIES` | `false` | Generate per-community LLM summaries (+ embeddings) during ingest |
| `COMMUNITY_SUMMARY_WEIGHT` | `0.2` | RRF weight for community-summary global search when summaries exist |
| `INDICES_DIR` | `/app/uploads/indices` | Directory for BM25/KG persistence files |
| `ENABLE_AGENT` | `false` | Phase 3 master toggle: run the agentic self-correction loop instead of a single generate call |
| `AGENT_MAX_RETRIES` | `2` | Max extra retrieve+generate+grade cycles after the first attempt |
| `AGENT_CONFIDENCE_THRESHOLD` | `0.5` | Grader score (0.0–1.0) below which a retry is triggered |
| `AGENT_ROUTER_WEIGHT_BOOST` | `0.15` | How much the router softly shifts RRF weight between BM25 (specific) and graph (broad) |
| `ENABLE_DECOMPOSITION` | `false` | Phase 3.5: decomposition + RAG-Fusion (requires `ENABLE_AGENT=true`) |
| `AGENT_DECOMP_MAX_SUBQUERIES` | `3` | Max sub-questions from the decomposer |
| `AGENT_DECOMP_MIN_WORDS` | `8` | Skip decomposition for shorter queries without multi-part triggers |
| `AGENT_FUSION_RRF_K` | `60` | RRF `k` when merging retrieval lists across sub-queries |
| `JUDGE_LLM_MODEL` | same as `LLM_MODEL` | Ollama model for Phase 4 LLM-as-Judge |
| `EVAL_OUTPUT_DIR` | (default under `python/evaluation/reports`) | Benchmark JSON + Markdown output directory |
| `EVAL_JUDGE_TEMPERATURE` | `0.0` | Temperature for judge LLM calls |
| `EVAL_MAX_CONCURRENCY` | `1` | Max concurrent judge LLM calls per question/config (1–3); full benchmark loop stays serial |

### Performance Profiles

| Profile | `SKIP_RERANKING` | `USE_DOCLING` | RAM Usage | Best For |
|---|---|---|---|---|
| **Lean (8GB)** | `true` | `false` | ~5.5GB | Development, demos |
| **Balanced (12GB)** | `false` | `false` | ~6.5GB | Production (text PDFs) |
| **Full (16GB+)** | `false` | `true` | ~7.5GB | Complex PDFs with tables |

---

## Algorithmic Contributions

This project implements or adapts the following algorithms, each with documented rationale:

### 1. Semantic-Aware Text Chunking

**File**: `python/compute_embeddings.py` — `chunk_text()`

Unlike naive fixed-length splitting, our chunker respects semantic boundaries:
- Splits at paragraph boundaries first (`\n\n`).
- Falls back to sentence boundaries (`(?<=[.!?])\s+`).
- Maintains configurable overlap (default 100 chars) for context continuity.
- Merges tiny chunks (<100 chars) to prevent embedding noise.

### 2. Structure-Preserving PDF Extraction

**File**: `python/compute_embeddings.py` — `pymupdf4llm.to_markdown()`

Standard text extraction (`fitz.get_text()`) outputs flat strings, destroying document structure. Our pipeline uses PyMuPDF4LLM to render PDFs as Markdown, preserving:
- Table structure (rows, columns, headers)
- Heading hierarchy (H1, H2, H3)
- List formatting (ordered and unordered)

This directly improves embedding quality because the LLM can reason about "row 3 of the table" rather than a garbled string.

### 3. Weighted Reciprocal Rank Fusion

**Formula**:

```
Score(d) = α · 1/(k + rank_vec) + (1-α) · 1/(k + rank_graph)
```

Where `α` controls the weight between vector similarity (neural) and graph traversal (symbolic), `k` is a smoothing constant (default 60), and `rank` is the position in each ranked list. This is adapted from Cormack et al. (2009) for dual-store retrieval.

### 4. Agentic Self-Correction Loop

A state machine that implements:
- **Query Classification**: 3-way routing — specific (BM25 bias), broad (graph bias), multi-hop (balanced fusion + optional auto decomposition).
- **Hallucination Detection**: Post-generation check for citation grounding.
- **Adaptive Retry**: Query rewriting with alternative strategies on failure.

---

## Evaluation Framework

### Evaluation Metrics

| Metric | What It Measures | How |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | LLM-as-Judge (Zheng et al., 2024) |
| **Answer Relevancy** | Does it address the question asked? | LLM-as-Judge scoring (0–1) |
| **Context Recall** | Did retrieval find all necessary info? | Comparison against gold-standard answers |
| **Conciseness** | Is the answer unnecessarily verbose? | Token-length ratio analysis |

### Comparison Methodology

A standardized evaluation comparing:
1. **Naive RAG** (vector-only, no history, no reranking)
2. **Neuro-Symbolic RAG** (hybrid retrieval + agent + graph)

Using 10+ multi-hop questions designed to expose single-strategy weaknesses (keyword-specific, cross-page connections, numerical queries).

---

## Roadmap

```
Phase 1: Foundation RAG ██████████████████████████████ 100%
├─ PDF extraction (PyMuPDF4LLM)              ✅
├─ Semantic chunking                          ✅
├─ Vector embedding (nomic-v2-moe)            ✅
├─ Qdrant storage + retrieval                 ✅
├─ LLM generation (gemma3:4b /api/chat)       ✅
├─ Chat history (structured messages)         ✅
├─ Cross-encoder reranking (optional)         ✅
├─ Docling integration (optional)             ✅
└─ Docker Compose orchestration               ✅

Phase 2: Hybrid Retrieval ██████████████████████████████ 100%
├─ BM25 keyword search (rank_bm25)           ✅
├─ Weighted RRF fusion (3-path + opt. community) ✅
├─ Entity extraction (LLM triples)            ✅
├─ NetworkX knowledge graph + JSON persist    ✅
├─ Leiden community detection                 ✅
├─ Graph traversal scoring (BFS)             ✅
└─ Hybrid pipeline with graceful degradation  ✅

Phase 3: Agentic Layer ████████████████████████████████ 100%
├─ Query router (specific / broad / multi_hop) ✅
├─ Hallucination grader                       ✅
├─ Self-correction loop (retry + rewrite)     ✅
├─ Query decomposition                        ✅
└─ RAG-Fusion (multi-query)                   ✅

Phase 4: Evaluation ██████████████████████████████  100%
├─ LLM-as-Judge framework                    ✅
├─ Naive vs Neuro-Symbolic comparison          ✅
├─ Benchmark report generation                ✅
├─ First baseline artifact (n=10, win 4-1-5)   ✅
└─ Knowledge graph visualization              ✅

Phase 5: Optimization & Depth (planned) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%
├─ Tier 2 — close Phase 4 scope & harden evaluation (≤ 1 day each)
│  ├─ 2.1 Knowledge graph visualization (pyvis/networkx)       ✅
│  ├─ 2.2 Larger/harder fixtures + distractor docs             ✅
│  ├─ 2.3 Prompt-artifact cleanup (markdown-list leakage)      ✅
│  ├─ 2.4 Ingest idempotency (--reset / pdf_id-scoped wipe)    ✅
│  ├─ 2.5 Graph retrieval returns full chunk text (not preview) ✅
│  └─ 2.6 Align eval neurosymbolic config with prod flags      ✅
└─ Tier 3 — scale, parallelism, research depth (multi-day)
   ├─ 3.1  Parallel judge calls (EVAL_MAX_CONCURRENCY wiring)  ✅
   ├─ 3.2  Ablation CLI (vector / +BM25 / +KG / +agent / +decomp) 🔲
   ├─ 3.3  Statistical significance (paired Wilcoxon, CIs)     ✅
   ├─ 3.4  Adaptive RAG router (complexity-aware path)         ✅
   ├─ 3.5  Learned fusion or DBSF alternative to RRF           🔲
   ├─ 3.6a Cross-encoder reranker default-on (bge-reranker-v2-m3) ✅
   ├─ 3.6b Late-interaction reranker for large topM (ColBERT-v2) 🔲
   ├─ 3.7  Ollama infra tuning (num_parallel, num_batch, FA)   ✅
   ├─ 3.8  GraphRAG community summaries + global-search path   ✅
   └─ 3.9  Batch ingest embedding (/api/embed array input)     ✅
```

*(Tier 1 = the Phase 4 baseline already shipped above; the numbering continues from there.)*

### Future Plans — Tier 2 & Tier 3 (post-Phase-4-baseline)

These are the concrete follow-ups identified while shipping the Phase 4 baseline. They are grouped by effort tier, roughly in the order they should be picked up.

#### Tier 2 — close Phase 4 scope & harden the evaluation (≤ 1 day each, except 2.2)

Ordered by expected impact-per-day. Top two are ½-day bug / correctness fixes and should land first.

| ID | Item | Why it matters | Done when |
|---|---|---|---|
| 2.5 | **Graph retrieval returns full chunk text** — completed by persisting `chunk_text` during entity extraction and reading it in graph retrieval with `chunk_text_preview` fallback for legacy graph JSON files | Restores graph-arm context quality in fused retrieval while preserving backwards compatibility for already-ingested PDFs. | ✅ Graph-backed sources now carry full chunk text on fresh ingest; legacy graph files continue to load via fallback. |
| 2.6 | **Align eval neurosymbolic config with prod flags** — completed by gating decomposition/RAG-Fusion in `evaluation/runner.py` behind `ENABLE_DECOMPOSITION` (matching `local_llm.py`) and recording the effective flag in trace metadata | Ensures benchmark behaviour matches deployed defaults and makes decomposition-on/off runs explicit in artifacts. | ✅ `neurosymbolic` is prod-faithful by default; reports include `trace.decomposition_enabled`. |
| 2.1 | **Knowledge graph visualization** — completed via `python -m evaluation.visualize_kg --pdf_id <pdf_id> --output-dir evaluation/reports/kg` to render `{pdf_id}_graph.json` as interactive HTML (`*_kg.html`) and static PNG (`*_kg.png`) | Closes the final deferred Phase 4 scope item and makes KG structure easy to inspect/share. | ✅ CLI and test coverage added; artifacts generated per `pdf_id` in `python/evaluation/reports/kg/`. |
| 2.3 | **Prompt-artifact cleanup** — completed via plain-prose instructions and `_strip_leading_list_marker` in `local_llm.py` `generate_rag_response` | Stops answers like `"1. 2 million dollars"` when the model emits a spurious list marker. | ✅ Regression tests on the post-processor; neurosymbolic answers avoid leading `1.` / bullet artifacts for single-fact replies. |
| 2.4 | **Ingest idempotency** — completed: `compute_embeddings.py --reset` runs a `pdf_id`-filtered delete in Qdrant before upsert (BM25/KG files were already overwritten per ingest); benchmark CLI now forwards `--reset` to fixture ingest | Re-ingest with new UUID point IDs no longer duplicates vectors for the same document when `--reset` is used. | ✅ CLI flag documented in script help; same PDF + `pdf_id` + `--reset` keeps point count stable across runs. |
| 2.2 | **Harder fixtures + distractor docs** — completed with `evaluation/datasets/harder.yaml` (15 curated questions) plus optional `meta.distractor_pdfs` support in dataset loading and benchmark ingest flow | Enables richer question coverage and same-collection distractor ingestion without changing retrieval algorithms. | ✅ New dataset + loader/ingest tests added; benchmark runner ingests fixture and distractors when configured. |

#### Tier 3 — scale, parallelism, research depth (multi-day each)

| ID | Item | Research anchor / rationale | Done when |
|---|---|---|---|
| 3.1 | **Parallel judge calls** — ✅ `EVAL_MAX_CONCURRENCY` runs the 3 LLM judge calls (faithfulness / relevancy / recall) concurrently via `ThreadPoolExecutor` in `evaluation/judge.py` (default `1` = serial). | Biggest wall-clock lever for the judge phase; real speedup needs Ollama `OLLAMA_NUM_PARALLEL` ≥ concurrency (see `DEPLOYMENT.md`). | ✅ Implemented with tests; measure end-to-end benchmark with server parallelism tuned. |
| 3.2 | **Ablation CLI** — `--ablation` flag that runs the same dataset across ≥ 4 configurations (vector-only / +BM25 / +BM25+KG / +agent / +agent+decomp) and emits a per-component contribution table | Single most defensible artifact for the provisional patent and for any academic write-up. | One `ablation.md` / `ablation.json` report with per-component deltas on a ≥ 20-question dataset. |
| 3.3 | **Statistical significance** — ✅ paired Wilcoxon signed-rank tests + bootstrap 95% CIs on per-question metric deltas now computed in `evaluation/report.py`; n < 20 is flagged as directional | Converts raw win counts into interpretable evidence and prevents over-claiming on small fixtures. | ✅ Report summary now includes per-metric Δ, CI, p-value, and `significant / directional / tie` verdicts. |
| 3.4 | **Adaptive RAG router** — ✅ 3-way `specific` / `broad` / `multi_hop` in [python/agent/query_router.py](python/agent/query_router.py); `multi_hop` does not shift RRF weights and **auto-enables** `QueryDecomposer` + `RAGFusion` in [python/agent/control_loop.py](python/agent/control_loop.py) when `ENABLE_DECOMPOSITION=false` (`trace.auto_decompose`). No parametric-only (no-retrieval) path yet; no web-search path. Tests: [python/tests/test_phase9_adaptive_router.py](python/tests/test_phase9_adaptive_router.py). | Routes comparative / multi-part questions into decomposition without turning decomposition on globally. | `MULTI_HOP` classification triggers sub-queries when the decomposer yields >1 line; injected decomposer is unchanged when `ENABLE_DECOMPOSITION=true`. |
| 3.5 | **Learned / distribution-based fusion** alternative to weighted RRF — start with Distribution-Based Score Fusion (DBSF, tuning-free) and, if labelled data becomes available, add a small LTR ensemble on top of the three retrieval lists | RRF is a robust default; LTR yields ~3–4% nDCG@10 on BEIR (46.5 → 48.2). DBSF is the safe midpoint. | DBSF selectable via `FUSION_METHOD` env; ablation row shows ≥ 1% nDCG@10 lift on ≥ 20-Q dataset before it replaces RRF. |
| 3.6a | **Cross-encoder reranker default-on** — ✅ default `SKIP_RERANKING=false`; reranker now runs on the fused hybrid top-M in `local_llm.py` (`RERANK_TOP_M`, default 50, capped at 150), with model selectable via `RERANKER_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`, optional `BAAI/bge-reranker-v2-m3`). | Public benchmarks (`recall@10` 72→94%, `precision@10` 65→91%) show cross-encoder reranking is the highest-yield single-stage addition we had not turned on yet. | ✅ Implemented with `test_phase6_rerank.py`; monitor benchmark quality + p95 latency when increasing `RERANK_TOP_M` or using heavier reranker models. |
| 3.6b | **Late-interaction reranker** for large topM — add ColBERT-v2 as an alternative reranker when topM ≥ 200 | Cross-encoders dominate precision at small M but scale poorly; ColBERT gives sub-10 ms rerank at M=200+. Only worth it after corpus grows. | ColBERT-v2 selectable via `RERANKER=colbert`; quality parity with cross-encoder at topM=50, wins at topM=200. |
| 3.7 | **Ollama infra tuning** — ✅ per-request `OLLAMA_NUM_BATCH`, optional `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE` wired through `OllamaLLM` + `OllamaEmbedder`; micro-bench `python -m evaluation.bench_ollama` (`gen` / `embed`). Host: `OLLAMA_NUM_PARALLEL`, `OLLAMA_FLASH_ATTENTION`. **Do NOT** set `OLLAMA_KV_CACHE_TYPE` ≠ `fp16` on gemma3 pre-0.12.5 (#9683 / #10945 / #11949). | Flash attention + `num_parallel` are the safe wins; KV-cache quantization is risky on gemma3 until Ollama 0.12.5. | ✅ Implemented + documented in `DEPLOYMENT.md` with bench CLI for before/after tok/s and wall-clock; tests: `test_phase7_ollama_tuning.py`. |
| 3.8 | **GraphRAG community summaries + global-search path** — ✅ opt-in `ENABLE_COMMUNITY_SUMMARIES`: ingest-time LLM + embed per Leiden community into `_graph.json`; `GraphRetriever.global_search` adds a 4th RRF list (weight `COMMUNITY_SUMMARY_WEIGHT`, vector/BM25/graph scaled down) when the agent route is not `specific` (non-agent queries still get the path). Tests: `test_phase8_graphrag.py`. | Closes the GraphRAG gap on global/summary-style retrieval without forcing extra ingest cost by default. | Summaries persist in graph JSON; hybrid fusion includes `community_summary` provenance when enabled; router-`specific` skips the path to protect factoid queries. |
| 3.9 | **Batch ingest embedding** — ✅ `OllamaEmbedder.encode_text` in `python/embeddings/ollama_embed.py` batches texts via `POST /api/embed` with array `input` (size capped by `EMBED_BATCH_SIZE`, default 32); per-item `POST /api/embeddings` fallback when `/api/embed` returns 404 (older Ollama). | 3–10× fewer HTTP round-trips on embedding-bound ingest; unblocks larger fixtures. | ✅ Implemented with `test_phase5_batch_embed.py`; measure ≥ 3× wall-clock on embedding-bound runs vs one-request-per-chunk baseline on the Tier 2.2 fixture. |

Pick by your current bottleneck: **benchmark speed →** 3.1 + 3.9; **patent / publication →** 3.2 + 3.3; **long-tail recall →** 3.4 + 3.5 + 3.6a; **broad-summary queries →** 3.8.

---

## Research References

This project draws from and adapts the following research:

1. **RAG-Fusion**: Shi et al. (2024) — Multi-query perspective generation for improved recall.
2. **HippoRAG**: Guo et al., Stanford (2024) — Memory-inspired knowledge graph retrieval using PageRank.
3. **Reciprocal Rank Fusion**: Cormack et al. (2009) — Score merging for heterogeneous retrieval systems.
4. **Leiden Algorithm**: Traag et al. (2019) — Community detection for knowledge graph clustering.
5. **LLM-as-Judge**: Zheng et al. (2024, MT-Bench) — Automated evaluation using LLM scoring.
6. **Docling**: IBM Research (2024) — ML-powered document layout analysis for RAG pipelines.
7. **GraphRAG**: Microsoft Research (2024) — Graph-based retrieval for global summarization queries.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Upload fails with "Parse Error" | stdout pollution from libraries | Redirect stdout to stderr (already fixed in codebase) |
| "Model not found" warning | Ollama stores `model:latest` | Run `ollama pull <model-name>` |
| Slow LLM responses | RAM pressure (>8GB total usage) | Set `SKIP_RERANKING=true` and `USE_DOCLING=false` |
| 0 results from Qdrant | Embedding model mismatch | Clear Qdrant collection and re-upload PDFs |
| Reset button fails | Config import error | Fixed: `print_current_config` now exported |

---

## Product Backlog

The Product Backlog is maintained using **MS Planner** and is refined iteratively. Below is the prioritised backlog with complete user stories and acceptance criteria.

### Epic 1: Document Ingestion & Processing

| ID | User Story | Priority | Story Points | Acceptance Criteria |
|---|---|---|---|---|
| US-101 | **As a** user, **I want to** upload PDF documents through the web interface **so that** I can ask questions about their content. | P1 — Must Have | 5 | ✅ AC1: Upload button accepts .pdf files up to 50MB. ✅ AC2: Upload progress indicator is displayed. ✅ AC3: Success/error notification appears after processing. ✅ AC4: PDF metadata (filename, page count, upload date) is stored in MongoDB. |
| US-102 | **As a** system, **I want to** extract text from PDFs as structured Markdown **so that** tables, headers, and lists are preserved for the LLM. | P1 — Must Have | 8 | ✅ AC1: PyMuPDF4LLM extracts text with `page_chunks=True`. ✅ AC2: Tables render as valid Markdown table syntax. ✅ AC3: Headings retain hierarchy (H1–H6). ✅ AC4: Extraction completes without stdout pollution (stderr redirect). |
| US-103 | **As a** system, **I want to** split extracted text into semantic chunks **so that** each chunk is a meaningful unit for embedding. | P1 — Must Have | 5 | ✅ AC1: Chunks respect paragraph boundaries first, then sentence boundaries. ✅ AC2: Overlap of 100 characters between consecutive chunks. ✅ AC3: No chunk is smaller than 100 characters (merged with predecessor). ✅ AC4: Chunk metadata includes page number, source filename, and chunk index. |
| US-104 | **As a** system, **I want to** embed text chunks using nomic-embed-text-v2-moe **so that** they are stored as 768-dimensional vectors for similarity search. | P1 — Must Have | 5 | ✅ AC1: Embeddings are generated via Ollama API in batches of 10. ✅ AC2: Task type is `search_document` for indexing. ✅ AC3: Batch upsert to Qdrant succeeds with 50 points per batch. ✅ AC4: Model availability is verified at startup (including `:latest` suffix matching). |
| US-105 | **As a** user, **I want** the option of ML-powered layout extraction via Docling **so that** complex PDFs with non-standard layouts are better understood. | P2 — Should Have | 8 | ✅ AC1: Docling is activated via `USE_DOCLING=true` environment variable. ✅ AC2: OCR is disabled by default (`do_ocr=False`) to avoid Tesseract dependency. ✅ AC3: Docling failure falls back to PyMuPDF4LLM gracefully. ✅ AC4: HybridChunker produces RAG-optimised chunks with page provenance. |

### Epic 2: Conversational Question-Answering

| ID | User Story | Priority | Story Points | Acceptance Criteria |
|---|---|---|---|---|
| US-201 | **As a** user, **I want to** ask questions about uploaded PDFs in a chat interface **so that** I get answers grounded in my documents. | P1 — Must Have | 8 | ✅ AC1: Chat input accepts text and submits on Enter. ✅ AC2: Response streams to the UI in real-time. ✅ AC3: Response includes source attribution (page numbers). ✅ AC4: Answer is generated within 10 seconds on 8GB RAM. |
| US-202 | **As a** user, **I want** the chat to remember previous turns **so that** I can ask follow-up questions without repeating context. | P1 — Must Have | 8 | ✅ AC1: Frontend stores conversation history as `user`/`assistant` role pairs. ✅ AC2: History is sent to the backend on each query. ✅ AC3: Backend constructs structured `/api/chat` messages with `system`, `user`, `assistant` roles. ✅ AC4: History is token-budgeted — oldest turns are dropped when exceeding `MAX_HISTORY_TOKENS`. |
| US-203 | **As a** user, **I want** the system to rerank search results for better accuracy **so that** the most relevant chunks are used for answering. | P2 — Should Have | 5 | ✅ AC1: Cross-encoder reranking is toggled via `SKIP_RERANKING` environment variable. ✅ AC2: When enabled, top-20 candidates are re-scored to top-5. ✅ AC3: Reranker model loads from HuggingFace cache volume (no re-download). ✅ AC4: When disabled, vector search directly returns top-5 results. |

### Epic 3: Knowledge Graph & Hybrid Retrieval

| ID | User Story | Priority | Story Points | Acceptance Criteria |
|---|---|---|---|---|
| US-301 | **As a** system, **I want to** extract entities and relationships from text chunks **so that** a knowledge graph can be constructed for multi-hop reasoning. | P1 — Must Have | 13 | ✅ AC1: LLM extracts `(Subject, Predicate, Object)` triples from each chunk. ✅ AC2: Entity resolution merges duplicate references (e.g., "Elon" and "Musk"). ✅ AC3: Triples are stored in NetworkX directed graph. ✅ AC4: Graph persists across queries for the same document. |
| US-302 | **As a** system, **I want to** fuse vector search results with graph traversal scores **so that** both semantic similarity and structural relationships inform retrieval. | P1 — Must Have | 8 | ✅ AC1: Weighted RRF formula implemented with configurable α parameter. ✅ AC2: Graph traversal uses BFS with configurable depth limit. ✅ AC3: Fused results outperform vector-only retrieval on multi-hop questions. ✅ AC4: Fusion latency adds <2 seconds to query time. |

### Epic 4: Agentic Self-Correction

| ID | User Story | Priority | Story Points | Acceptance Criteria |
|---|---|---|---|---|
| US-401 | **As a** system, **I want to** route queries to the optimal retrieval strategy **so that** keyword-specific and conceptual queries are handled differently. | P2 — Should Have | 5 | ✅ AC1: Router classifies query as "specific" or "broad" using LLM. ✅ AC2: Specific queries prioritise BM25/keyword search. ✅ AC3: Broad queries prioritise graph traversal + vector search. |
| US-402 | **As a** system, **I want to** evaluate generated answers for hallucination **so that** I can retry with a rewritten query when quality is low. | P1 — Must Have | 13 | ✅ AC1: Grader checks citation coverage in the answer. ✅ AC2: Confidence score computed on 0–1 scale. ✅ AC3: Retry triggered when score < 0.5. ✅ AC4: Maximum 3 retries before returning best attempt. ✅ AC5: Agent trace log shows reasoning at each step. |

---

## Sprint Backlog

User stories from the Product Backlog are committed to sprints and broken into implementable tasks.

### Committed User Stories & Task Breakdown

#### US-101: PDF Upload via Web Interface

| Task ID | Task | Assignee | Estimated Hours | Status |
|---|---|---|---|---|
| T-101.1 | Create file upload component in React (drag-and-drop + button) | Frontend | 4h | ✅ Done |
| T-101.2 | Implement `/api/upload` endpoint in Express (multer middleware) | Backend | 3h | ✅ Done |
| T-101.3 | Store PDF metadata in MongoDB (filename, page count, upload timestamp) | Backend | 2h | ✅ Done |
| T-101.4 | Spawn Python `compute_embeddings.py` as child process from Node.js | Backend | 3h | ✅ Done |
| T-101.5 | Parse JSON result from Python stdout; handle errors | Backend | 2h | ✅ Done |
| T-101.6 | Display upload success/error notification in UI | Frontend | 1h | ✅ Done |

#### US-102: Structure-Preserving PDF Extraction

| Task ID | Task | Assignee | Estimated Hours | Status |
|---|---|---|---|---|
| T-102.1 | Add `pymupdf4llm` to `requirements.txt` and verify Docker build | DevOps | 1h | ✅ Done |
| T-102.2 | Replace `page.get_text("text")` with `pymupdf4llm.to_markdown(page_chunks=True)` | Python | 3h | ✅ Done |
| T-102.3 | Redirect pymupdf4llm stdout → stderr to prevent JSON parse corruption | Python | 1h | ✅ Done |
| T-102.4 | Validate Markdown extraction: test with tables, headers, lists | QA | 2h | ✅ Done |
| T-102.5 | Implement Docling fallback path with `USE_DOCLING` env flag | Python | 4h | ✅ Done |
| T-102.6 | Configure Docling with `do_ocr=False` to avoid Tesseract dependency | Python | 1h | ✅ Done |

#### US-201: Conversational Q&A Interface

| Task ID | Task | Assignee | Estimated Hours | Status |
|---|---|---|---|---|
| T-201.1 | Build chat input component with send button and Enter key handler | Frontend | 3h | ✅ Done |
| T-201.2 | Implement `/api/query` endpoint calling `local_llm.py` | Backend | 3h | ✅ Done |
| T-201.3 | Embed query via `nomic-embed-text-v2-moe` with `search_query` task type | Python | 2h | ✅ Done |
| T-201.4 | Execute Qdrant `query_points()` with pdf_id filter | Python | 2h | ✅ Done |
| T-201.5 | Assemble structured `/api/chat` messages (system + context + query) | Python | 3h | ✅ Done |
| T-201.6 | Display LLM response with source page references in chat UI | Frontend | 2h | ✅ Done |

#### US-202: Chat History Preservation

| Task ID | Task | Assignee | Estimated Hours | Status |
|---|---|---|---|---|
| T-202.1 | Store conversation turns in React state as `user`/`assistant` pairs | Frontend | 2h | ✅ Done |
| T-202.2 | Update `api.js` `queryRAG()` to send history array to backend | Frontend | 1h | ✅ Done |
| T-202.3 | Modify `local_llm.py` to accept and format history as chat messages | Python | 3h | ✅ Done |
| T-202.4 | Switch from Ollama `/api/generate` to `/api/chat` endpoint | Python | 2h | ✅ Done |
| T-202.5 | Implement token budgeting: trim oldest turns to fit `MAX_HISTORY_TOKENS` | Python | 2h | ✅ Done |

---

## Sprint Activities

### Daily Scrum

Daily standup updates tracking progress, blockers, and planned work.

| Date | What Was Done | Blockers | Next Steps |
|---|---|---|---|
| Day 1 | Project setup: Docker Compose scaffolding, MongoDB/Qdrant services configured. Ollama installed and verified on host. | None | Implement PDF upload endpoint |
| Day 2 | PDF upload endpoint created. Multer middleware configured for file handling. MongoDB schema for PDF metadata defined. | None | Integrate Python embedding script |
| Day 3 | `compute_embeddings.py` integrated as child process. PyMuPDF text extraction working. Chunks embedded via `nomic-embed-text:v1.5`. | Qdrant `client.search()` deprecated — API incompatibility | Fix Qdrant API to use `query_points()` |
| Day 4 | Qdrant API fixed (`search()` → `query_points()`). Vector storage and retrieval verified. Basic chat interface created. | Ollama not reachable from Docker container | Fix Ollama network: bind to `0.0.0.0`, use `host.containers.internal` |
| Day 5 | Ollama connectivity resolved. LLM answering queries via `/api/generate`. Frontend chat component functional. | Chat treats each query independently — no context memory | Implement chat history |
| Day 6 | Chat history implemented: frontend stores turns, backend builds structured `/api/chat` messages. Switched from `/api/generate` to `/api/chat`. | HuggingFace models re-download on every container restart | Add `hf_cache` Docker volume |
| Day 7 | HF cache volume created. Upgraded embedding model to `nomic-embed-text-v2-moe` (MoE architecture, multilingual). Updated vector size config. | Old embeddings in Qdrant incompatible with new model | Clear Qdrant collection, re-upload PDFs |
| Day 8 | Integrated Docling (IBM) for ML-powered PDF extraction. Added `USE_DOCLING` flag with PyMuPDF fallback. | Docling error: "No OCR engine found" (missing Tesseract) | Configure `do_ocr=False` for Docling |
| Day 9 | Docling OCR disabled. Added `SKIP_RERANKING` and `SKIP_IMAGES` flags for 8GB RAM optimisation. Upgraded LLM to `gemma3:4b`. | System slow due to RAM pressure from multiple ML models | Disable non-essential components via flags |
| Day 10 | Replaced `fitz` extraction with `pymupdf4llm` (Markdown output). Fixed stdout pollution breaking JSON parsing. All core features verified working. | None — all blockers resolved | Begin knowledge graph and hybrid retrieval (Phase 2) |

---

### Functional Documents

#### Functional Requirement Specification

| Req ID | Requirement | Category | Priority | Implementation Status |
|---|---|---|---|---|
| FR-001 | System shall accept PDF uploads up to 50MB via REST API | Ingestion | P1 | ✅ Implemented |
| FR-002 | System shall extract text from PDFs preserving tables and headers as Markdown | Ingestion | P1 | ✅ Implemented |
| FR-003 | System shall split text into overlapping semantic chunks (500 chars, 100 overlap) | Ingestion | P1 | ✅ Implemented |
| FR-004 | System shall generate 768-dimensional embeddings using MoE model | Ingestion | P1 | ✅ Implemented |
| FR-005 | System shall store embeddings in Qdrant with page metadata | Ingestion | P1 | ✅ Implemented |
| FR-006 | System shall accept natural language queries via chat interface | Query | P1 | ✅ Implemented |
| FR-007 | System shall retrieve top-K relevant chunks via cosine similarity | Query | P1 | ✅ Implemented |
| FR-008 | System shall optionally rerank results using cross-encoder model | Query | P2 | ✅ Implemented |
| FR-009 | System shall generate answers using structured `/api/chat` messages | Query | P1 | ✅ Implemented |
| FR-010 | System shall maintain conversation history across multiple turns | Query | P1 | ✅ Implemented |
| FR-011 | System shall operate within 8GB RAM using configurable component flags | System | P1 | ✅ Implemented |
| FR-012 | System shall cache ML models across container restarts | System | P2 | ✅ Implemented |
| FR-013 | System shall provide alternative PDF extraction via Docling | Ingestion | P3 | ✅ Implemented |
| FR-014 | System shall construct a knowledge graph from extracted entities | Retrieval | P1 | ✅ Implemented |
| FR-015 | System shall fuse vector and graph retrieval via RRF | Retrieval | P1 | ✅ Implemented |
| FR-016 | System shall implement agentic self-correction with hallucination grading | Agent | P1 | ✅ Done (router + grader + retry; optional decomposition + RAG-Fusion via `ENABLE_DECOMPOSITION`) |

#### Non-Functional Requirements

| Req ID | Requirement | Category | Metric |
|---|---|---|---|
| NFR-001 | Query response time shall be under 10 seconds | Performance | Measured at LLM response start |
| NFR-002 | System shall run on 8GB RAM minimum | Resource | Docker stats monitoring |
| NFR-003 | All data shall remain local (no cloud API calls) | Privacy | No external network calls for ML |
| NFR-004 | System shall recover gracefully from component failures | Reliability | Fallback mechanisms for each ML component |
| NFR-005 | Docker Compose shall orchestrate all services with one command | Deployment | `docker compose up --build -d` |

---

### UI Design

#### Application Layout

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER: Neuro-Symbolic RAG System                    [⚙️]  │
├────────────────────┬─────────────────────────────────────────┤
│                    │                                         │
│   SIDEBAR          │   MAIN CHAT AREA                        │
│                    │                                         │
│   📄 Documents     │   ┌─────────────────────────────────┐   │
│   ├── Report.pdf   │   │ 🤖 Assistant                    │   │
│   ├── Paper.pdf    │   │ Based on Section 3.2 of your    │   │
│   └── Resume.pdf   │   │ document, the results show...   │   │
│                    │   │ [Source: Page 4]                │   │
│   ────────────     │   └─────────────────────────────────┘   │
│                    │                                         │
│   [📤 Upload PDF]  │   ┌─────────────────────────────────┐   │
│                    │   │ 👤 You                          │   │
│   ────────────     │   │ What were the key findings?     │   │
│                    │   └─────────────────────────────────┘   │
│   STATUS           │                                         │
│   Model: gemma3:4b │   ┌─────────────────────────────────┐   │
│   Embedder: nomic  │   │ 🤖 Assistant                    │   │
│   Chunks: 47       │   │ The key findings include...     │   │
│                    │   └─────────────────────────────────┘   │
│                    │                                         │
│                    │  ┌──────────────────────────────┬────┐  │
│                    │  │  Ask a question...           │ ➤  │  │
│                    │  └──────────────────────────────┴────┘  │
└────────────────────┴─────────────────────────────────────────┘
```

#### Key UI Components

| Component | Technology | Description |
|---|---|---|
| `ChatInterface.js` | React | Main chat area with message bubbles, source tags, and history state management |
| `FileUpload.js` | React | Drag-and-drop PDF upload with progress bar |
| `PdfList.js` | React | Sidebar listing uploaded documents with selection |
| `api.js` | Axios | API client with `queryRAG(question, history)` and `uploadPDF(file)` |
| Nginx Proxy | Config | `/api/*` requests proxied to backend on port 5000 |

---

### Architecture Document

#### High-Level System Architecture

```
                         ┌────────────────┐
                         │     USER       │
                         │   (Browser)    │
                         └───────┬────────┘
                                 │ HTTP :3000
                         ┌───────▼────────┐
                         │    NGINX       │
                         │  (Frontend)    │
                         │  React SPA     │
                         └───────┬────────┘
                                 │ Proxy /api/*
                         ┌───────▼────────┐
                         │   NODE.JS      │
                         │  (Express)     │
                         │  :5000         │
                         └──┬────┬────┬───┘
                            │    │    │
              ┌─────────────┘    │    └──────────────┐
              │                  │                   │
    ┌─────────▼──────┐  ┌───────▼────────┐  ┌───────▼────────┐
    │   PYTHON ML    │  │   MONGODB      │  │    QDRANT      │
    │   Scripts      │  │   :27017       │  │    :6333       │
    │                │  │   (Metadata)   │  │   (Vectors)    │
    │ compute_embed  │  └────────────────┘  └────────────────┘
    │ local_llm      │
    └────────┬───────┘
             │ HTTP :11434
    ┌────────▼───────┐
    │    OLLAMA      │
    │   (Host)       │
    │  gemma3:4b     │
    │  nomic-v2-moe  │
    └────────────────┘
```

#### Current Features Highlighted

| Feature | Component | Key File(s) |
|---|---|---|
| **Structure-preserving extraction** | Python ML | `compute_embeddings.py` — `pymupdf4llm.to_markdown()` |
| **MoE embedding** | Python ML → Ollama | `embeddings/ollama_embed.py` — `encode_text()` |
| **Chat history** | Frontend → Backend → Python | `ChatInterface.js` → `api.js` → `local_llm.py` → `ollama_llm.py` |
| **Structured LLM interaction** | Python ML → Ollama | `llm/ollama_llm.py` — `/api/chat` with role-based messages |
| **Resource-aware orchestration** | Docker Compose | `docker-compose.yml` — `SKIP_RERANKING`, `USE_DOCLING`, `SKIP_IMAGES` |
| **HF model caching** | Docker Compose | `docker-compose.yml` — `hf_cache` volume mounted at `HF_HOME` |

#### Data Flow

```
Upload: PDF → PyMuPDF4LLM → Chunks → nomic-v2-moe → Qdrant + BM25 Index + Knowledge Graph
Query:  Question → [Vector Search + BM25 + Graph Traversal] → RRF Fusion → [Rerank] → gemma3:4b → Answer
```

---

### Functional Test Case Document

#### Test Suite: Document Ingestion

| TC ID | Test Case | Preconditions | Steps | Expected Result | Status |
|---|---|---|---|---|---|
| TC-001 | Upload valid PDF | App running, no PDFs uploaded | 1. Click Upload 2. Select a 2-page PDF 3. Wait for processing | Success notification, PDF appears in sidebar, logs show `[PyMuPDF4LLM] Got 2 page(s) as Markdown` | ✅ Pass |
| TC-002 | Upload large PDF (>20 pages) | App running | 1. Upload 25-page academic paper 2. Monitor logs | All pages processed, chunks upserted in batches, no timeout | ✅ Pass |
| TC-003 | PDF with tables | App running | 1. Upload PDF containing data tables 2. Check extraction logs | Tables extracted as Markdown syntax, `extractor: "pymupdf4llm"` in metadata | ✅ Pass |
| TC-004 | Docling extraction toggle | `USE_DOCLING=true` in env | 1. Restart container 2. Upload PDF | Logs show `[Docling] Converting PDF (OCR disabled)...` | ✅ Pass |
| TC-005 | Docling fallback to PyMuPDF4LLM | Docling import fails | 1. Remove docling package 2. Upload PDF | Logs show `[Docling] Not installed. Falling back to PyMuPDF.` then `[PyMuPDF4LLM]` extraction | ✅ Pass |

#### Test Suite: Query & Retrieval

| TC ID | Test Case | Preconditions | Steps | Expected Result | Status |
|---|---|---|---|---|---|
| TC-101 | Basic question answering | PDF uploaded and indexed | 1. Type question about document content 2. Submit | Relevant answer displayed with source page references | ✅ Pass |
| TC-102 | Chat history context | Previous Q&A turn exists | 1. Ask follow-up question referencing previous answer | Answer correctly references prior context, logs show `N messages to LLM` where N > 3 | ✅ Pass |
| TC-103 | Cross-encoder reranking | `SKIP_RERANKING=false` | 1. Restart container 2. Ask question | Logs show reranker loaded, 20 candidates reduced to 5 | ✅ Pass |
| TC-104 | Vector search accuracy | PDF about specific topic uploaded | 1. Ask question using keywords from document 2. Check sources | Retrieved chunks contain keywords, page numbers match source document | ✅ Pass |
| TC-105 | Empty collection handling | No PDFs uploaded | 1. Ask a question | Graceful response: "No relevant documents found" or similar | ✅ Pass |

#### Test Suite: System & Resource Management

| TC ID | Test Case | Preconditions | Steps | Expected Result | Status |
|---|---|---|---|---|---|
| TC-201 | 8GB RAM operation | `SKIP_RERANKING=true`, `USE_DOCLING=false` | 1. Start app 2. Upload PDF 3. Ask 5 questions | All operations complete without OOM or disk swapping | ✅ Pass |
| TC-202 | Model cache persistence | HF models previously downloaded | 1. `docker compose down` 2. `docker compose up -d` | No model download at startup, logs show immediate model load | ✅ Pass |
| TC-203 | Ollama connectivity | Ollama running on host | 1. Start containers | Logs show `Model nomic-embed-text-v2-moe is available` (no false warning) | ✅ Pass |
| TC-204 | Qdrant collection reset | Documents exist in Qdrant | 1. Call reset endpoint or `curl -X DELETE .../collections/documents` | Collection deleted and recreated with correct vector size (768) | ✅ Pass |

---

### Demo of Deliverables

#### Implemented Deliverables

| # | Deliverable | Description | Demo Method |
|---|---|---|---|
| 1 | **PDF Upload & Processing** | Upload a PDF → automatic Markdown extraction → chunking → embedding → Qdrant storage | Live demo: Upload a research paper, show logs streaming extraction and embedding |
| 2 | **Conversational Q&A** | Ask questions → vector retrieval → LLM answer with source pages | Live demo: Ask 3 questions about the uploaded paper, showing accurate answers with page citations |
| 3 | **Chat History Awareness** | Follow-up questions reference prior context correctly | Live demo: Ask "Tell me about the methodology", then "How does that compare to the results?" |
| 4 | **Structure Preservation** | Tables and headers in PDFs are correctly understood by the LLM | Live demo: Upload a PDF with tables, ask about specific tabular data |
| 5 | **Resource-Aware Configuration** | Toggle components on/off for different RAM profiles | Live demo: Show `docker-compose.yml` flags, demonstrate lean vs full mode |
| 6 | **Docling Integration** | Alternative ML extraction with automatic fallback | Live demo: Enable `USE_DOCLING=true`, rebuild, show Docling logs |
| 7 | **Model Upgrades** | LLM (`gemma3:4b`), Embedder (`nomic-v2-moe`), Extractor (`pymupdf4llm`) | Show before/after comparison of answer quality |

#### Demo Script

1. **Start the application**: `docker compose up --build -d`
2. **Open browser**: Navigate to `http://localhost:3000`
3. **Upload a PDF**: Use a research paper with tables, abstract, and methodology sections
4. **Show logs**: `docker compose logs -f backend` — highlight `[PyMuPDF4LLM] Got N page(s) as Markdown`
5. **Ask questions**:
   - "Summarise the key findings of this paper"
   - "What does Table 1 show?" *(demonstrates table extraction)*
   - "How does that relate to the methodology?" *(demonstrates chat history)*
6. **Show Qdrant dashboard**: `http://localhost:6333/dashboard` — visualise stored vectors and metadata

---

### Sprint Retrospective

#### What Went Well ✅

| Area | Detail |
|---|---|
| **Modular Architecture** | The Python ML scripts are cleanly separated from the Node.js backend. Each can be modified independently. |
| **Fallback Design** | Every ML component (Docling, Reranker, Image Embedding) has a graceful fallback, preventing single points of failure. |
| **Incremental Model Upgrades** | Embedding model (v1.5 → v2-moe) and LLM (qwen → gemma) were swapped by changing 2 lines of config each. |
| **Docker Volume Caching** | Adding `hf_cache` volume eliminated 2-3 minute startup delays per container restart. |
| **Local-First Approach** | Entire system runs with zero external API calls, making it reproducible for any demo or evaluation. |

#### What Could Be Improved 🔧

| Area | Detail | Action Item |
|---|---|---|
| **Stdout Pollution** | pymupdf4llm prints to stdout, corrupting JSON IPC between Node and Python | Fixed via stderr redirect; future libraries need stdout audit |
| **API Breaking Changes** | Qdrant deprecated `client.search()` with no warning in our pinned version | Pin exact dependency versions in `requirements.txt` |
| **RAM Profiling** | No automated check for available RAM before enabling components | Add startup RAM check to auto-select performance profile |
| **Image Embedding** | `OllamaEmbedder` lacks a `get_embedding()` method for images | Need vision-capable embedding model or OCR pipeline |
| **Test Automation** | All testing is manual — no CI/CD or automated test suite | Add pytest suite for Python components, Jest for frontend |

#### Action Items for Next Iteration

| # | Action | Priority | Owner | Status |
|---|---|---|---|---|
| 1 | Implement BM25 keyword search + RRF fusion | P1 | Backend | ✅ Done (Phase 2) |
| 2 | Build knowledge graph extractor using LLM triples | P1 | Python ML | ✅ Done (Phase 2) |
| 3 | Create agentic query router and hallucination grader | P1 | Python ML | ✅ Done (Phase 3 medium scope: router + grader + retry loop behind `ENABLE_AGENT`) |
| 3b | Add query decomposition + RAG-Fusion multi-query | P2 | Python ML | ✅ Done (`ENABLE_DECOMPOSITION`) |
| 4 | Add automated test suite (pytest + Jest) | P2 | QA | 🔲 Planned |
| 5 | Implement LLM-as-Judge evaluation framework | P2 | Python ML | ✅ Done (Phase 4 — see [python/evaluation/](python/evaluation/)) |
| 6 | Add startup RAM profiling for auto-configuration | P3 | DevOps | 🔲 Planned |

---

## License

This project is developed as an academic major project.

---

*Built with ❤️ using Ollama, Qdrant, React, and Docker.*