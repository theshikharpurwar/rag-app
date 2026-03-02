# Prior Art Search Report
**Invention:** Neuro-Symbolic Agentic System for Retrieval-Augmented Generation (Tri-Layer Cognitive Architecture)
**Date:** 2 March 2026
**Searches Conducted:** Google Patents, USPTO, WIPO, Google Scholar, Indian Patent databases

---

## Overall Verdict

> [!TIP]
> **🟢 GREEN LIGHT — No exact match found.** No existing patent covers your specific combination of: Tri-Layer Architecture + Agentic Self-Correction + Neuro-Symbolic Knowledge Store + Fusion Retrieval + Resource-Aware Local Deployment. Individual components exist (as expected), but your **unified system design** is novel.

---

## Closest Matching Patents (Ranked by Relevance)

### 🔴 HIGH RELEVANCE — Must Differentiate

| # | Patent Number | Title | Assignee | Why It Matters | Your Differentiator |
|---|---|---|---|---|---|
| 1 | **US20250245255A1** | Neuro-symbolic retrieval augmented generation hybrid method | Unknown | **Closest match** — directly covers neuro-symbolic + RAG hybrid approach | Your system has a **three-layer** architecture (agentic control + hybrid store + fusion retrieval). This patent doesn't specify a tri-layer cognitive architecture, agentic self-correction loop, or resource-aware deployment |
| 2 | **US20250131289A1** | Providing meaningful information from datasets using knowledge graph + RAG | Unknown | Combines knowledge graphs with RAG for community-level and global retrieval | Your system adds an **agentic control loop** with hallucination grading and adaptive retry. Also operates on **consumer hardware** locally, not cloud |
| 3 | **US12135740B1** | Systems for generating unified metadata graph using RAG framework | Unknown | Uses domain-specific ontologies (similar to knowledge graphs) with RAG | Focuses on **metadata unification** across data sources, not document QA. No agentic self-correction or structure-preserving PDF pipeline |

### 🟡 MEDIUM RELEVANCE — Should Reference as Prior Art

| # | Patent Number | Title | Assignee | Overlap | Your Differentiator |
|---|---|---|---|---|---|
| 4 | **US12,254,272** | Context-aware semantic chunking for information retrieval in LLMs | **Citigroup** (Granted Mar 2025) | Semantic chunking + knowledge graph association for each chunk | Your chunking uses a **three-tier boundary-aware** algorithm (paragraph → sentence → word) with sentence-aligned overlap. Citigroup's is context-aware but different mechanism |
| 5 | **US20250098020A1** | Multi-layered caching strategy for LLM responses using hybrid retrieval + RRF | Unknown | Uses hybrid retrieval and Reciprocal Rank Fusion | Focuses on **caching** strategy, not document QA. Your RRF fuses **vector + knowledge graph** ranks, not cached responses |
| 6 | **WO/2025/184621** | RAG-based query reformulation pipeline for QA system | Unknown (WIPO) | Uses DAG + knowledge graph triplets + query reformulation | Your system uses **adaptive retry with query decomposition** (splitting complex queries into sub-queries), which is different from DAG-based reformulation |

### 🟢 LOW RELEVANCE — Good to Know

| # | Reference | Type | Overlap | Why Not a Threat |
|---|---|---|---|---|
| 7 | **SELF-RAG** (Asai et al., 2023) | Academic Paper | LLM self-evaluation using reflection tokens | Research paper, not a patent. Your approach uses a **separate grading agent**, not reflection tokens. Different mechanism |
| 8 | IBM Hallucination Reduction Patent (pending) | Patent Application | Multi-AI-model interaction for hallucination reduction | IBM's approach focuses on context integration across multiple AI models. Your system uses a **single LLM + grading function** within an agentic loop |
| 9 | **SymRAG** (Aug 2025) | Academic Paper | Adaptive query routing between symbolic/neural/hybrid pathways | Very similar concept to your query router, but it's a **research paper, not a patent**. Validates your approach |
| 10 | **NeuSym-RAG** (May 2025) | Academic Paper | Hybrid neural-symbolic retrieval for PDF QA with multi-view chunking | Academic work, closely related. Not patented. Actually **validates your invention's approach** |

---

## Component-by-Component Analysis

### 1. Tri-Layer Cognitive Architecture
| Finding | Details |
|---|---|
| **Exact match?** | ❌ **NO** — No patent found for a tri-layer (agentic + neuro-symbolic + fusion) architecture for RAG |
| **Similar?** | Some multi-layer architectures exist (e.g., WO2018132112A1 "Digital Twin Graph" with 3-layer DTG, Z2H2 framework), but **none apply to RAG** |
| **Novelty** | ✅ **Strong** |

### 2. Agentic Self-Correcting Control Loop
| Finding | Details |
|---|---|
| **Exact match?** | ❌ **NO** — No patent found for an agentic loop with hallucination grading + query decomposition + adaptive retry in a unified RAG system |
| **Similar?** | IBM patent (pending) addresses hallucination reduction but via multi-model interaction, not agentic loop. SELF-RAG uses reflection tokens (different mechanism) |
| **Novelty** | ✅ **Strong** |

### 3. Hybrid Knowledge Store (Vector DB + Knowledge Graph)
| Finding | Details |
|---|---|
| **Exact match?** | ⚠️ **PARTIAL** — US20250131289A1 and US20250245255A1 both combine knowledge graphs with RAG |
| **Similar?** | GraphRAG (Microsoft), US20250131289A1 |
| **Your edge** | Your **dual-store** approach (Qdrant vector + NetworkX graph with Leiden community detection) running **locally** is distinct from cloud-based approaches |
| **Novelty** | 🟡 **Moderate** — Must clearly differentiate in claims |

### 4. Weighted Reciprocal Rank Fusion (Vector + Graph)
| Finding | Details |
|---|---|
| **Exact match?** | ❌ **NO** — No patent specifically fuses **vector similarity ranks with knowledge graph traversal ranks** via weighted RRF |
| **Similar?** | US20250098020A1 uses RRF for caching. RRF itself is known (Cormack et al., 2009). General hybrid retrieval + RRF exists |
| **Your edge** | Your specific fusion formula: `Score(d) = α·1/(k+rank_vec) + (1-α)·1/(k+rank_graph)` with tunable α is a **novel application** of RRF |
| **Novelty** | ✅ **Strong** (application is novel, even though RRF algorithm is known) |

### 5. Structure-Preserving PDF → Markdown Extraction
| Finding | Details |
|---|---|
| **Exact match?** | ❌ **NO** — No patent covers PyMuPDF4LLM → Markdown extraction with Docling ML fallback in a RAG pipeline |
| **Similar?** | Adobe PDF-to-Markdown API, Marker, LlamaParse all do structure-preserving extraction. SynapsePatents processes tables/equations. But none are patented as part of a RAG system |
| **Your edge** | Your **dual-extractor with automatic failover** (PyMuPDF4LLM primary → Docling fallback) is unique |
| **Novelty** | 🟡 **Moderate** — The tools are open-source; your integration with failover is the novel part |

### 6. Three-Tier Boundary-Aware Chunking
| Finding | Details |
|---|---|
| **Exact match?** | ⚠️ **PARTIAL** — US12,254,272 (Citigroup, granted 2025) covers "context-aware semantic chunking" |
| **Similar?** | Recursive chunking (LangChain), similarity-based chunking, adaptive chunking |
| **Your edge** | Your specific **three-tier hierarchy** (paragraph → sentence → word) with **sentence-aligned overlap** and **minimum chunk merging** (100 char threshold) is a distinct algorithm |
| **Novelty** | 🟡 **Moderate** — Citigroup patent is close but uses different mechanism. Clearly describe your algorithm |

### 7. Resource-Aware Local Deployment
| Finding | Details |
|---|---|
| **Exact match?** | ❌ **NO** — No patent found for configurable component activation based on RAM in a RAG system |
| **Similar?** | Edge AI patents exist for resource-constrained devices, but focus on mobile/IoT, not RAG on desktop |
| **Your edge** | 3 performance profiles (Lean 8GB / Balanced 12GB / Full 16GB+) with toggle flags is **completely unique** |
| **Novelty** | ✅ **Strong** |

---

## Indian Patent Landscape

| Finding | Details |
|---|---|
| **RAG patents in India** | Very few. One known: Application **202341039746** (Computer Vision, not RAG) |
| **Neuro-symbolic RAG in India** | ❌ **None found** |
| **JURISMIND project** | Academic research on RAG for Indian patent rules — not patented |
| **Your advantage** | Filing in India gives you **first-mover advantage** in this space. Very little prior art domestically |

---

## Recommended Actions for Your Patent

> [!IMPORTANT]
> ### In Your Complete Specification, You Must:
>
> 1. **Cite these patents as prior art** in your Background section — examiners respect this
> 2. **Clearly differentiate** from US20250245255A1 (closest match) — emphasize: tri-layer design, agentic loop, resource-aware deployment
> 3. **Narrow claims** on the hybrid knowledge store (Component 3) — this has the most overlap
> 4. **Strengthen claims** on Components 1, 2, 4, and 7 — these are your strongest novelty points

### Your Strongest Novelty Claims (In Order)

1. 🥇 **Tri-Layer Cognitive Architecture** — unified system design, no match found
2. 🥈 **Resource-Aware Component Orchestration** — 3 profiles for consumer hardware, no match
3. 🥉 **Agentic Self-Correcting Loop** — hallucination grading + adaptive retry + query decomposition
4. 4️⃣ **Weighted RRF for Vector + Graph Fusion** — novel application of known algorithm
5. 5️⃣ **Dual-Extractor Pipeline with Failover** — PyMuPDF4LLM + Docling integration

---

## References

| # | Patent/Paper | Year | Relevant Component |
|---|---|---|---|
| 1 | US20250245255A1 — Neuro-symbolic RAG hybrid | 2025 | Overall system |
| 2 | US20250131289A1 — Knowledge graph + RAG | 2025 | Knowledge Store |
| 3 | US12135740B1 — Unified metadata graph via RAG | 2024 | Knowledge Store |
| 4 | US12,254,272 — Context-aware semantic chunking (Citigroup) | 2025 | Chunking |
| 5 | US20250098020A1 — Multi-layer caching + RRF | 2025 | RRF Fusion |
| 6 | WO/2025/184621 — RAG query reformulation | 2025 | Query routing |
| 7 | SELF-RAG (Asai et al.) | 2023 | Self-correction |
| 8 | SymRAG — Adaptive query routing | 2025 | Query routing |
| 9 | NeuSym-RAG — Hybrid neural-symbolic PDF QA | 2025 | Overall approach |
| 10 | Lewis et al. — Original RAG paper (NeurIPS) | 2020 | Foundation |
| 11 | Cormack, Clarke, Butt — RRF algorithm | 2009 | Fusion formula |
| 12 | Traag et al. — Leiden algorithm | 2019 | Community detection |
