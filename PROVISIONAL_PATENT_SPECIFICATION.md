# FORM 2
# THE PATENTS ACT, 1970
# (39 OF 1970)
# &
# THE PATENTS RULES, 2003

---

# PROVISIONAL SPECIFICATION

**(See Section 10; Rule 13)**

---

## TITLE OF THE INVENTION

**Neuro-Symbolic Agentic System and Method for Retrieval-Augmented Generation from Unstructured Documents Using Tri-Layer Cognitive Architecture**

---

## APPLICANT(S)

| Field | Details |
|---|---|
| **Name** | *[TO BE FILLED — Full legal name of applicant(s)]* |
| **Nationality** | Indian |
| **Address** | *[TO BE FILLED — Full postal address]* |

> **Note:** If the applicant is a student filing under an educational institution, the institution may be listed as co-applicant with appropriate assignment agreements.

---

## PREAMBLE

The following specification describes the invention.

---

## 1. FIELD OF THE INVENTION

The present invention relates generally to the field of **Artificial Intelligence (AI)** and **Natural Language Processing (NLP)**, and more particularly to a novel system and method for **Retrieval-Augmented Generation (RAG)** from unstructured document collections. The invention pertains to a **Tri-Layer Cognitive Architecture** that combines neural vector-based retrieval, symbolic knowledge graph reasoning, and autonomous agentic control to achieve accurate, context-aware, and self-correcting question-answering over PDF documents — operating entirely on local consumer hardware without dependency on cloud-based AI services.

---

## 2. BACKGROUND OF THE INVENTION

### 2.1 Prior Art and Existing Systems

Retrieval-Augmented Generation (RAG) is an established technique in Natural Language Processing wherein a Large Language Model (LLM) is provided with externally retrieved context documents to ground its responses in factual data, thereby reducing hallucination. The foundational RAG architecture, as described by Lewis et al. (2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS), follows a two-stage pipeline:

**Stage 1 — Retrieval:** A user query is converted into a dense vector embedding. This vector is compared against a pre-indexed corpus of document embeddings using cosine similarity. The top-K most similar document chunks are retrieved.

**Stage 2 — Generation:** The retrieved chunks are concatenated with the user query and passed to a generative LLM, which synthesizes an answer conditioned on the provided context.

Several commercial and open-source systems implement this architecture, including LangChain, LlamaIndex, and various cloud-hosted RAG services (e.g., Amazon Kendra, Azure AI Search). These systems typically rely on a single-modality retrieval mechanism (dense vector similarity) and a single-pass generation pipeline.

### 2.2 Limitations of Existing Systems

Despite their utility, existing RAG systems suffer from three critical limitations that degrade their reliability in production environments:

**Limitation 1 — Blind Retrieval (Keyword Insensitivity):**
Dense vector embeddings capture semantic meaning but frequently fail to retrieve documents containing specific keywords, numerical values, dates, or technical identifiers. For example, a query requesting "the result in Table 3.2" may retrieve semantically similar content about results and tables but miss the exact section. This is because cosine similarity operates on distributional semantics rather than lexical matching.

**Limitation 2 — Brittle Context (Inability to Connect Disparate Facts):**
Standard RAG treats each document chunk as an independent unit. It has no mechanism to recognise that Entity A mentioned on page 1 is the same entity referenced on page 7, or that Concept X is a prerequisite for understanding Concept Y. This inability to traverse relationships between facts leads to incomplete or superficial answers for questions requiring multi-hop reasoning.

**Limitation 3 — Dumb Failure (No Error Recovery):**
In existing systems, when the retrieval stage returns irrelevant or insufficient context, the generation stage proceeds regardless, producing hallucinated or fabricated answers with no indication of failure. There is no feedback mechanism to detect low-quality retrieval, no strategy to reformulate the query, and no capacity for self-correction.

**Limitation 4 — Lossy Document Extraction:**
Most existing systems use basic text extraction (e.g., PyMuPDF `get_text()`, pdfminer) that outputs flat character streams, destroying document structure. Tables become garbled strings, headings lose their hierarchy, and lists collapse into run-on text. This structural loss degrades the quality of both the stored embeddings and the LLM's ability to reason about the document.

### 2.3 Need for the Present Invention

There exists a need for a RAG system that:
- Combines **multiple retrieval modalities** (vector similarity and keyword matching) through a mathematically principled fusion mechanism;
- Maintains **structured knowledge representations** (entity-relation graphs) that enable multi-hop reasoning across document sections;
- Implements **autonomous error detection and recovery** through an agentic control loop;
- Preserves **document structure** (tables, headings, lists) during extraction; and
- Operates **entirely locally** on consumer hardware (8GB RAM) without cloud API dependencies, addressing data privacy and sovereignty concerns.

---

## 3. OBJECT OF THE INVENTION

The principal object of the present invention is to provide a **Neuro-Symbolic Agentic system and method** for Retrieval-Augmented Generation that overcomes the above-stated limitations of the prior art.

A further object is to provide a **Tri-Layer Cognitive Architecture** comprising:
- An **Agentic Control Loop** (Layer 1) that autonomously routes queries, evaluates answer quality, and triggers self-correcting retry mechanisms;
- A **Hybrid Knowledge Store** (Layer 2) that maintains both a vector embedding database and a symbolic knowledge graph, enabling both semantic similarity search and structured entity-relation traversal; and
- A **Graph-Augmented Fusion Retrieval Algorithm** (Layer 3) that combines vector cosine similarity scores with knowledge graph traversal scores using Weighted Reciprocal Rank Fusion.

A still further object is to provide a system that preserves document structure during PDF extraction by rendering documents as structured Markdown prior to embedding, thereby retaining tables, headings, and hierarchical formatting.

Yet another object is to achieve the above on **consumer-grade hardware** (8GB RAM minimum) by implementing a resource-aware model orchestration strategy that selectively activates or deactivates computational components (reranking, layout analysis) based on available system memory.

---

## 4. DETAILED DESCRIPTION OF THE INVENTION

### 4.1 System Overview

The invented system comprises a **containerised microservices architecture** orchestrated via Docker Compose, with the following interconnected services:

1. **Frontend Service** — A React-based single-page application served via Nginx, providing a web interface for document upload and conversational question-answering.
2. **Backend Service** — A Node.js/Express server that orchestrates the ML pipeline by spawning Python processes for embedding computation, retrieval, and generation.
3. **Document Database** — MongoDB, storing PDF metadata, file references, and processing state.
4. **Vector Database** — Qdrant, an in-memory vector similarity search engine storing 768-dimensional document embeddings.
5. **LLM Runtime** — Ollama, an inference server running quantised Large Language Models locally on the host machine's CPU/GPU.

### 4.2 Layer 3: Document Ingestion and Embedding Pipeline

The ingestion pipeline transforms raw PDF documents into searchable vector representations through the following novel process:

#### 4.2.1 Structure-Preserving Extraction

Unlike prior art systems that extract flat text, the present invention employs **PyMuPDF4LLM**, a purpose-built library that renders PDF content as structured Markdown. This process:
- Converts PDF tables into Markdown table syntax (rows, columns, header cells);
- Preserves heading hierarchy (H1 through H6);
- Maintains list formatting (ordered and unordered);
- Retains code block boundaries.

The output is a per-page Markdown string that preserves the document's logical structure, enabling downstream embedding models and LLMs to reason about structural relationships (e.g., "the value in row 3, column 2 of the table").

Additionally, the system provides an alternative extraction path via **IBM Docling**, which employs ONNX-based machine learning models for layout detection, table structure recognition, and reading order inference. This path is selectable via configuration (`USE_DOCLING=true`) and automatically falls back to PyMuPDF4LLM on failure.

#### 4.2.2 Semantic-Aware Text Chunking

The extracted Markdown text is split into overlapping chunks using a novel **three-tier boundary-aware algorithm**:

**Tier 1 — Paragraph Splitting:** The text is first split at paragraph boundaries (`\n\n`), respecting the document's natural semantic divisions.

**Tier 2 — Sentence Splitting:** If a paragraph exceeds the configured chunk size (default: 500 characters), it is further split at sentence boundaries using a lookbehind regex `(?<=[.!?])\s+`, ensuring no sentence is broken mid-thought.

**Tier 3 — Word Splitting:** If a single sentence exceeds the chunk size (e.g., very long technical descriptions), it is split at word boundaries as a last resort.

Each chunk maintains a configurable overlap window (default: 100 characters) with the preceding chunk. The overlap is adjusted to begin at the nearest sentence boundary to preserve contextual continuity. Chunks smaller than 100 characters are merged into the preceding chunk to eliminate embedding noise from very short text fragments.

#### 4.2.3 Vector Embedding

Each text chunk is embedded into a 768-dimensional vector space using the **nomic-embed-text-v2-moe** model, a Mixture-of-Experts (MoE) embedding model that routes each input through specialised expert subnetworks. The MoE architecture provides:
- Multilingual embedding capability;
- Higher quality representations than single-expert models at equivalent computational cost;
- Native support for task-type prefixing (`search_document` for indexing, `search_query` for retrieval), enabling asymmetric search.

Embeddings are generated via the Ollama API in batches (default batch size: 10) and upserted to Qdrant in batches of 50, with each point storing the embedding vector alongside metadata (page number, source filename, chunk index, extractor type).

### 4.3 Layer 3: Retrieval Pipeline

#### 4.3.1 Vector Similarity Search

Upon receiving a user query, the system:
1. Embeds the query using the same `nomic-embed-text-v2-moe` model with `task_type: search_query`;
2. Executes a filtered cosine similarity search against the Qdrant collection, constrained to the specific `pdf_id` of the active document;
3. Retrieves the top-K candidate chunks (default K=20 when reranking is enabled, K=5 otherwise).

#### 4.3.2 Cross-Encoder Reranking (Optional)

When activated (`SKIP_RERANKING=false`), the top-K candidates are re-scored using a **cross-encoder model** (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Unlike the bi-encoder used for initial retrieval (which independently encodes query and document), the cross-encoder processes the query-document pair jointly, producing a more accurate relevance score at the cost of higher latency. The top-5 results after reranking are passed to the generation stage.

#### 4.3.3 Reciprocal Rank Fusion (Planned Enhancement)

The system is designed to implement **Weighted Reciprocal Rank Fusion** to combine heterogeneous retrieval signals:

```
Score(d) = α · 1/(k + rank_vector) + (1 − α) · 1/(k + rank_graph)
```

Where:
- `α` ∈ [0, 1] is a tunable weight controlling the balance between vector (neural) and graph (symbolic) retrieval;
- `k` is a smoothing constant (default: 60) that prevents excessive influence from top-ranked results;
- `rank_vector` is the document's position in the vector similarity ranked list; and
- `rank_graph` is the document's position in the knowledge graph traversal ranked list.

This fusion formula, adapted from Cormack, Clarke, and Butt (2009), enables the system to benefit from both retrieval modalities without requiring score normalisation across heterogeneous ranking functions.

### 4.4 Layer 2: Hybrid Knowledge Store (Planned Enhancement)

The system architecture provides for a **dual-store knowledge representation**:

#### 4.4.1 Vector Store (Implemented — Qdrant)

Stores dense vector embeddings of document chunks, enabling semantic similarity search. Each point in the store contains:
- A 768-dimensional float vector;
- Payload metadata: `pdf_id`, `page`, `source`, `text`, `chunk_index`, `extractor`.

#### 4.4.2 Knowledge Graph (Planned — NetworkX)

A directed graph where:
- **Nodes** represent extracted entities (persons, organisations, concepts, methods);
- **Edges** represent relationships between entities, labelled with predicate types (e.g., "authored_by", "related_to", "prerequisite_of");
- **Communities** are detected using the **Leiden Algorithm** (Traag et al., 2019), enabling topic-level summarisation.

Entity extraction is performed by prompting the LLM with structured output instructions to produce `(Subject, Predicate, Object)` triples from each text chunk.

### 4.5 Layer 1: Agentic Control Loop (Planned Enhancement)

The topmost layer implements a **self-correcting agent state machine**:

#### 4.5.1 Query Router

A classification module that analyses the user query to determine the optimal retrieval strategy:
- **Specific queries** (containing identified names, dates, section references) → routed primarily to keyword/BM25 search;
- **Broad queries** (conceptual, comparative, multi-hop) → routed to graph traversal + vector search.

#### 4.5.2 Hallucination Grader

After the LLM generates a response, a **grading function** evaluates the answer's faithfulness to the retrieved context. The grading criteria include:
- Citation coverage: Does the answer reference information present in the retrieved chunks?
- Factual consistency: Are numerical values, names, and relationships accurately reproduced?
- Confidence threshold: If the grading score falls below a configurable threshold, the system triggers a retry.

#### 4.5.3 Adaptive Retry Mechanism

When a retry is triggered:
1. The original query is **decomposed** into sub-queries (e.g., "Compare A and B" → "Find A", "Find B");
2. The decomposed queries are executed through **alternative retrieval paths** (e.g., switching from vector-only to graph-only);
3. The results are merged and a new generation attempt is made;
4. The loop continues until the confidence threshold is met or a maximum retry count is reached.

### 4.6 Structured Conversation History

The system maintains conversational context across multiple turns by constructing structured message lists conforming to the OpenAI chat completion format:

```
[
  {"role": "system",    "content": "<RAG instructions + document extracts>"},
  {"role": "user",      "content": "<previous question 1>"},
  {"role": "assistant", "content": "<previous answer 1>"},
  {"role": "user",      "content": "<current question>"}
]
```

This structured format (as opposed to raw text concatenation used in prior art systems) ensures the LLM correctly interprets the conversational context, role boundaries, and instruction hierarchy. The history is token-budgeted: the system estimates token counts for each historical turn and includes only the most recent turns that fit within a configurable token limit (`MAX_HISTORY_TOKENS`), prioritising recency.

### 4.7 Resource-Aware Model Orchestration

A distinguishing feature of the present invention is its ability to operate on **consumer hardware with as little as 8GB of RAM**. This is achieved through a configurable component activation system:

| Component | RAM Impact | Configuration Flag | Purpose |
|---|---|---|---|
| Cross-encoder reranker | +250MB per query | `SKIP_RERANKING` | Improves retrieval precision |
| Docling layout analysis | +300MB at ingestion | `USE_DOCLING` | ML-powered table/heading detection |
| Image embedding | +0MB (disabled) | `SKIP_IMAGES` | Prevents broken image embedding attempts |
| HuggingFace model cache | Persistent volume | `HF_HOME` | Prevents repeated model downloads |

The system profiles are:
- **Lean (8GB):** Reranking disabled, Docling disabled — total ~5.5GB RAM;
- **Balanced (12GB):** Reranking enabled, Docling disabled — total ~6.5GB RAM;
- **Full (16GB+):** All components active — total ~7.5GB RAM.

---

## 5. NOVELTY AND INVENTIVE STEP

The present invention distinguishes itself from the prior art in the following aspects:

1. **Tri-Layer Cognitive Architecture:** No existing open-source RAG framework implements a unified architecture that combines agentic self-correction (Layer 1), hybrid neuro-symbolic knowledge stores (Layer 2), and graph-augmented fusion retrieval (Layer 3) in a single system designed for local execution.

2. **Structure-Preserving Pipeline:** The use of PyMuPDF4LLM for Markdown-native extraction with fallback to Docling for ML-powered layout analysis — with automatic failover between the two — is not found in existing RAG implementations, which typically use a single extraction method.

3. **Resource-Aware Orchestration:** The configurable component activation system that allows the same codebase to operate across 8GB to 16GB+ machines by selectively enabling/disabling ML components at runtime is a novel approach to democratising advanced RAG on consumer hardware.

4. **Three-Tier Boundary-Aware Chunking:** The semantic chunking algorithm that respects paragraph → sentence → word boundaries with sentence-aligned overlap is more sophisticated than the fixed-length or recursive character splitting used in LangChain and LlamaIndex.

---

## 6. DRAWINGS AND DIAGRAMS

### Figure 1: System Architecture (Tri-Layer Cognitive Architecture)

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
│   │  Vector Store   │              │  (NetworkX)             │      │
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
│   │  Score(d) = α·1/(k+rank_vec) + (1-α)·1/(k+rank_graph)  │      │
│   │                                                         │      │
│   │  Path A (Fast): BM25 + Vector Cosine Similarity         │      │
│   │  Path B (Deep): Graph Traversal (BFS/Dijkstra)          │      │
│   └─────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Figure 2: Document Ingestion Pipeline

```
PDF Upload → Structure-Preserving Extraction (PyMuPDF4LLM / Docling)
           → Markdown Output (tables, headers preserved)
           → Three-Tier Boundary-Aware Chunking
           → MoE Vector Embedding (nomic-embed-text-v2-moe, 768-dim)
           → Qdrant Vector Database (batch upsert with metadata)
```

### Figure 3: Query Processing Pipeline

```
User Query → Query Embedding (nomic-embed-text-v2-moe)
           → Qdrant Cosine Similarity Search (filtered by pdf_id)
           → [Optional] Cross-Encoder Reranking
           → [Planned] RRF Fusion with Knowledge Graph results
           → Structured Message Assembly (system + history + query)
           → LLM Generation (gemma3:4b via /api/chat)
           → [Planned] Hallucination Grading → Retry if low confidence
           → Response to User
```

---

## 7. TECHNOLOGY STACK

| Component | Technology | Version/Variant |
|---|---|---|
| LLM | Ollama + gemma3:4b | Google, March 2025 |
| Embedding Model | nomic-embed-text-v2-moe | Nomic AI, 768-dim MoE |
| Vector Database | Qdrant | Latest stable |
| Document Database | MongoDB | Latest stable |
| PDF Extraction (Primary) | PyMuPDF4LLM | Markdown output |
| PDF Extraction (ML) | Docling (IBM) | Optional, ONNX-based |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Optional |
| Knowledge Graph | NetworkX | Planned |
| Backend | Node.js 18 + Express | Python 3.11 ML bridge |
| Frontend | React 18 + Nginx | SPA |
| Orchestration | Docker Compose | Podman compatible |
| Operating System | Linux (Fedora) | Local deployment |

---

## 8. IMPLEMENTATION STATUS

| Component | Status | Description |
|---|---|---|
| PDF ingestion with structure preservation | **Implemented** | PyMuPDF4LLM Markdown extraction |
| Semantic chunking with boundary awareness | **Implemented** | Three-tier algorithm |
| MoE vector embedding via Ollama | **Implemented** | nomic-embed-text-v2-moe |
| Qdrant vector storage and retrieval | **Implemented** | Cosine search with pdf_id filtering |
| LLM generation via /api/chat | **Implemented** | Structured role-based messages |
| Conversational history management | **Implemented** | Token-budgeted message construction |
| Cross-encoder reranking | **Implemented** | Toggle via SKIP_RERANKING |
| Resource-aware component orchestration | **Implemented** | 3 performance profiles |
| Docling ML extraction integration | **Implemented** | Toggle via USE_DOCLING |
| Docker Compose orchestration | **Implemented** | Full microservices stack |
| BM25 keyword search | **Planned** | Sparse retrieval |
| Knowledge graph construction | **Planned** | NetworkX + Leiden communities |
| RRF fusion algorithm | **Planned** | Weighted rank fusion |
| Agentic query router | **Planned** | Strategy classification |
| Hallucination grader | **Planned** | Citation-checking agent |
| Adaptive retry mechanism | **Planned** | Query decomposition + rewrite |

---

*Date: ____________________*

*Signature of Applicant(s): ____________________*

*Name of Applicant(s): ____________________*
