# 🔍 Invention Disclosure Form — Review & Alignment Report

> [!IMPORTANT]
> This is your **college's internal Invention Disclosure Form** (SRM IST), **NOT** the Indian Patent Office Form 1. This form is for getting your college's IPR cell approval before you file with the government. Below are all misalignments with your current project and what to change.

---

## Summary: 9 Issues Found

| # | Field | Severity | Issue |
|---|---|---|---|
| 1 | Title | 🔴 Critical | Doesn't match your provisional specification |
| 2 | Description (Q2) | 🟡 Medium | Too vague — missing key technical details |
| 3 | Novelty (Q4) | 🟡 Medium | Missing several novel components |
| 4 | Advantages (Q5) | 🟡 Medium | Missing resource-aware deployment advantage |
| 5 | Drawbacks overcome (Q6) | 🟡 Medium | Missing structure-preserving extraction |
| 6 | Applications (Q7) | 🟢 Low | Fine but could mention current actual use case |
| 7 | Limitations (Q9) | 🟢 Low | Missing a key limitation |
| 8 | Development status (Q11) | 🔴 Critical | Outdated — all components are now fully implemented |
| 9 | Department field | 🟢 Low | Says "DSBS" — confirm this is correct |

---

## Detailed Changes Per Field

### 🔴 1. Title of Invention — MUST CHANGE

**Old Title:**
> A System and Method for Structured Multi-Hop Knowledge Retrieval Using Graph-Based Community Abstraction and Rank-Fused Semantic-Symbolic Search with Autonomous Validation

**New Title (must match your Provisional Specification):**
> Neuro-Symbolic Agentic System and Method for Retrieval-Augmented Generation from Unstructured Documents Using Tri-Layer Cognitive Architecture

```diff
- A System and Method for Structured Multi-Hop Knowledge Retrieval Using Graph-Based Community Abstraction and Rank-Fused Semantic-Symbolic Search with Autonomous Validation
+ Neuro-Symbolic Agentic System and Method for Retrieval-Augmented Generation from Unstructured Documents Using Tri-Layer Cognitive Architecture
```

> [!CAUTION]
> **The title MUST match exactly between this form, your Form 1, and your Form 2 (Provisional Specification).** Any mismatch can cause rejection or delays.

---

### 🟡 2. Description (Question 2) — UPDATE

**Old (too vague):**
> The present invention relates to a computer-implemented system and method for intelligent information retrieval and reasoning over large unstructured document repositories...

**Suggested replacement:**
> The present invention relates to a computer-implemented Neuro-Symbolic Agentic system and method for Retrieval-Augmented Generation (RAG) from unstructured PDF documents, employing a novel Tri-Layer Cognitive Architecture. **Layer 1** implements an agentic control loop with autonomous query routing, hallucination grading, and adaptive retry mechanisms. **Layer 2** maintains a hybrid knowledge store combining a Qdrant vector embedding database (768-dimensional, nomic-embed-text-v2-moe) with a NetworkX symbolic knowledge graph employing Leiden community detection. **Layer 3** implements a Graph-Augmented Fusion Retrieval algorithm using Weighted Reciprocal Rank Fusion to combine vector cosine similarity scores with knowledge graph traversal scores. The system features structure-preserving PDF extraction (PyMuPDF4LLM with Docling ML fallback), three-tier boundary-aware semantic chunking, cross-encoder reranking, and resource-aware model orchestration enabling deployment on consumer hardware with 8GB RAM. The invention comprises a system, method, and containerised microservices software architecture orchestrated via Docker Compose.

---

### 🟡 3. Novelty (Question 4) — ADD MISSING ITEMS

**Old claims (4 items):**
1. Hybrid integration of symbolic knowledge graph reasoning and semantic vector retrieval
2. Community-based abstraction of knowledge for hierarchical reasoning
3. Rank-fused combination of symbolic and semantic retrieval results
4. Autonomous grading and iterative self-correction of generated outputs

**Updated claims (7 items) — add these 3:**

```diff
  1. Hybrid integration of symbolic knowledge graph reasoning and semantic vector retrieval
  2. Community-based abstraction of knowledge for hierarchical reasoning
  3. Rank-fused combination of symbolic and semantic retrieval results
  4. Autonomous grading and iterative self-correction of generated outputs
+ 5. Tri-Layer Cognitive Architecture unifying agentic control, neuro-symbolic storage, and fusion retrieval
+ 6. Structure-preserving PDF extraction with dual-extractor failover (PyMuPDF4LLM → Docling)
+ 7. Resource-aware model orchestration with configurable performance profiles (8GB/12GB/16GB+)
```

---

### 🟡 4. Advantages (Question 5) — ADD

```diff
  • Enables multi-hop and cross-document reasoning
  • Reduces factual errors and hallucinations
  • Improves retrieval precision and contextual relevance
  • Supports large-scale document collections
  • Suitable for high-stakes decision-making environments
+ • Operates entirely on consumer hardware (8GB RAM minimum) without cloud API dependencies
+ • Preserves document structure (tables, headings, lists) during PDF extraction
+ • Three configurable performance profiles adapt to available hardware resources
```

---

### 🟡 5. Drawbacks Overcome (Question 6) — ADD

```diff
  Current technologies lack structured reasoning and self-validation mechanisms...
+ Additionally, existing RAG systems destroy document structure during PDF text extraction, 
+ causing tables, headings, and lists to become garbled text. The present invention overcomes
+ this through structure-preserving Markdown extraction. Furthermore, current systems require 
+ cloud-hosted LLMs with significant API costs and data privacy concerns. The present invention 
+ operates entirely locally on consumer hardware, addressing both cost and data sovereignty issues.
```

---

### 🟢 6. Applications (Question 7) — MINOR ADD

The existing list is fine. Optionally add:

```diff
  • Legal document analysis and case research
  • Medical guideline interpretation and clinical decision support
  • Regulatory and compliance auditing
  • Enterprise knowledge retrieval systems
  • Intelligent decision-support applications
+ • Academic and research document analysis
+ • Privacy-sensitive document processing (on-premises deployment)
```

---

### 🟢 7. Limitations (Question 9) — ADD

```diff
  • Initial computational overhead during knowledge graph construction
  • Dependence on the quality of input documents
  • Requires domain-specific configuration for optimal performance
+ • Performance varies with available system RAM (8GB minimum, 16GB+ recommended for full features)
+ • Currently limited to PDF document format
```

---

### 🔴 8. Development Status (Question 11) — MUST UPDATE

**Old:**
> Further development required: Yes. Further optimization, scalability evaluation, and domain-specific tuning are planned.

**Updated — ALL components are now implemented:**

> **Has the invention been tested experimentally?**
> Yes, the invention has been fully implemented and tested as a functional system.
>
> **Experimental approach:**
> The system was evaluated using controlled document datasets and query-based retrieval tasks. All 15 components listed in the provisional specification have been implemented and tested, including: PDF ingestion with structure preservation, semantic chunking, MoE vector embedding, Qdrant vector storage and retrieval, LLM generation, conversational history management, cross-encoder reranking, resource-aware component orchestration, Docling ML extraction, Docker Compose orchestration, BM25 keyword search, knowledge graph construction, RRF fusion algorithm, agentic query router, hallucination grader, and adaptive retry mechanism.
>
> **Documentation of experimental data:**
> System architecture, codebase, and experimental outputs are fully documented.
>
> **Further development required:**
> The core system is complete. Future work includes scalability benchmarking, domain-specific fine-tuning, and formal evaluation metrics.

---

### 🟢 9. Department — VERIFY

The form says **DSBS** (Department of Science & Behavioural Studies?). Confirm this is the correct department for your project. If it should be CSE or IT, update it.

---

## Inventor Details — VERIFY

| Field | Current | Action |
|---|---|---|
| **Faculty** | Dr. Varun P | ✅ Verify — is this still your project guide? |
| **Inventor 1** | Sarthak Kumar | ✅ Verify |
| **Inventor 2** | Shikhar Purwar (you) | ✅ Confirmed |
| **Inventor 3** | Akash Singh | ✅ Verify |
| **Department** | DSBS | ⚠️ Verify — should this be CSE/IT? |
| **Address fields** | "Department ___" (blank) | ❌ Fill in the department name |

---

## ⚠️ Important Note: This Form vs. Government Form 1

| Aspect | This Document | Government Form 1 |
|---|---|---|
| **What is it?** | **SRM IST internal** Invention Disclosure Form | **Indian Patent Office** Application for Grant of Patent |
| **Filed with** | Your college IPR cell | IP India (ipindiaonline.gov.in) |
| **Purpose** | Get college approval + check IP policy | Officially file for patent protection |
| **What's next?** | After college approves → file Form 1 with IP India | Your patent application begins |

> [!IMPORTANT]
> **Action sequence:**
> 1. ✅ Update this Invention Disclosure Form with the changes above
> 2. ✅ Submit to SRM IST IPR cell for approval
> 3. ✅ After approval → file Form 1 + Form 2 + Form 28 on ipindiaonline.gov.in
