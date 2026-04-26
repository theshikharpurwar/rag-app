# D:\rag-app\python\config\models.py
"""
Centralized model configuration for the RAG application.
Change model names here and they will be applied throughout the entire application.
"""

import os

# =============================================================================
# 🔧 MAIN CONFIGURATION - CHANGE MODELS HERE ONLY!
# =============================================================================

# LLM Model Configuration
LLM_MODEL_NAME = os.environ.get('LLM_MODEL', 'qwen3.5:0.8b')

# Embedding Model Configuration  
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text:v1.5')

# Max texts per Ollama POST /api/embed request (single round-trip per batch)
EMBED_BATCH_SIZE = int(os.environ.get('EMBED_BATCH_SIZE', '32'))
# Cross-encoder reranker model and oversampling size before rerank
RERANKER_MODEL = os.environ.get('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
RERANK_TOP_M = min(int(os.environ.get('RERANK_TOP_M', '50')), 150)
# Reranker backend: cross_encoder (default) or colbert (requires ragatouille)
RERANKER_TYPE = os.environ.get('RERANKER_TYPE', 'cross_encoder').lower().strip()
COLBERT_MODEL = os.environ.get('COLBERT_MODEL', 'colbert-ir/colbertv2.0')

# =============================================================================
# 📊 MODEL SPECIFICATIONS (Auto-derived from model names)
# =============================================================================

# Vector dimensions based on embedding model
EMBEDDING_VECTOR_SIZES = {
    'nomic-embed-text:v1.5': 768,
    'nomic-embed-text:latest': 768,
    'nomic-embed-text': 768,
    'nomic-embed-text-v2-moe': 768,   # MoE v2 — multilingual, same dim as v1.5
    'all-MiniLM-L6-v2': 384,
    'sentence-transformers/all-MiniLM-L6-v2': 384,
    'all-minilm': 384,
    'embeddinggemma': 768,
}

# Auto-detect vector size based on embedding model
DEFAULT_VECTOR_SIZE = EMBEDDING_VECTOR_SIZES.get(EMBEDDING_MODEL_NAME, 768)

# =============================================================================
# 🌐 SERVICE CONFIGURATION
# =============================================================================

# Qdrant Configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
DEFAULT_COLLECTION = 'documents'

# Ollama Configuration
OLLAMA_HOST_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
OLLAMA_API_BASE = f"{OLLAMA_HOST_URL}/api"

# Per-request /api/chat and /api/embed tuning (host-side: OLLAMA_NUM_PARALLEL, OLLAMA_FLASH_ATTENTION)
OLLAMA_NUM_BATCH = int(os.environ.get("OLLAMA_NUM_BATCH", "512"))
_OLLAMA_NUM_CTX_RAW = int(os.environ.get("OLLAMA_NUM_CTX", "0"))
OLLAMA_NUM_CTX = _OLLAMA_NUM_CTX_RAW if _OLLAMA_NUM_CTX_RAW > 0 else None
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "5m")

# =============================================================================
# 🎯 RAG PARAMETERS
# =============================================================================

# Context and retrieval settings
CONTEXT_RETRIEVAL_LIMIT = 5
MAX_CONTEXT_CHAR_LIMIT = 4096
MAX_HISTORY_TOKENS = 500

# PDF Processing settings
TEXT_CHUNK_SIZE = 800
TEXT_CHUNK_OVERLAP = 100
IMAGE_SAVE_DIR_RELATIVE = "images"
RENDERING_DPI = 150

# =============================================================================
# 🔗 PHASE 2: HYBRID RETRIEVAL & KNOWLEDGE GRAPHS
# =============================================================================

# Master toggle — set to 'false' to disable KG extraction during ingestion
ENABLE_KNOWLEDGE_GRAPH = os.environ.get('ENABLE_KNOWLEDGE_GRAPH', 'true').lower() == 'true'

# Reciprocal Rank Fusion parameters
RRF_K = int(os.environ.get('RRF_K', '60'))             # Smoothing constant
FUSION_METHOD = os.environ.get('FUSION_METHOD', 'rrf').lower().strip()  # 'rrf' or 'dbsf'
VECTOR_WEIGHT = float(os.environ.get('VECTOR_WEIGHT', '0.4'))   # α — vector similarity weight
BM25_WEIGHT = float(os.environ.get('BM25_WEIGHT', '0.3'))       # β — BM25 keyword weight
GRAPH_WEIGHT = float(os.environ.get('GRAPH_WEIGHT', '0.3'))     # γ — graph traversal weight

# Graph traversal
GRAPH_TRAVERSAL_DEPTH = int(os.environ.get('GRAPH_TRAVERSAL_DEPTH', '2'))

# Community summaries during ingest (requires ENABLE_KNOWLEDGE_GRAPH=true); extra LLM calls per Leiden community
ENABLE_COMMUNITY_SUMMARIES = (
    os.environ.get('ENABLE_COMMUNITY_SUMMARIES', 'false').lower() == 'true'
)
COMMUNITY_SUMMARY_WEIGHT = float(os.environ.get('COMMUNITY_SUMMARY_WEIGHT', '0.2'))

# KG extraction strategy and diagnostics
KG_EXTRACTION_MODE = os.environ.get('KG_EXTRACTION_MODE', 'llm_triples').lower().strip()
KG_NP_EXTRACTOR = os.environ.get('KG_NP_EXTRACTOR', 'regex').lower().strip()
KG_TRIPLE_BATCH_SIZE = int(os.environ.get('KG_TRIPLE_BATCH_SIZE', '3'))
KG_EXTRACT_CONCURRENCY = max(1, int(os.environ.get('KG_EXTRACT_CONCURRENCY', '1')))
KG_EXTRACT_MAX_CHARS = int(os.environ.get('KG_EXTRACT_MAX_CHARS', '1200'))
KG_STORAGE_PRETTY = os.environ.get('KG_STORAGE_PRETTY', 'false').lower() == 'true'
KG_LLM_MODEL = os.environ.get('KG_LLM_MODEL', LLM_MODEL_NAME)

# Directory for persisted BM25 indexes and knowledge graph files
INDICES_DIR = os.environ.get('INDICES_DIR', '/app/uploads/indices')

# =============================================================================
# 🤖 PHASE 3: AGENTIC SELF-CORRECTION
# =============================================================================

# Master toggle — when false, Phase 2 pipeline runs unchanged
ENABLE_AGENT = os.environ.get('ENABLE_AGENT', 'false').lower() == 'true'

# Retry loop: how many extra retrieve+generate+grade cycles after the first attempt
AGENT_MAX_RETRIES = int(os.environ.get('AGENT_MAX_RETRIES', '2'))

# Grader: answers with confidence below this threshold trigger a retry
AGENT_CONFIDENCE_THRESHOLD = float(os.environ.get('AGENT_CONFIDENCE_THRESHOLD', '0.5'))

# Router: soft nudge on RRF weights based on query classification.
# Kept small so a misclassification cannot collapse any single retrieval path.
AGENT_ROUTER_WEIGHT_BOOST = float(os.environ.get('AGENT_ROUTER_WEIGHT_BOOST', '0.15'))

# Phase 3.5: query decomposition + RAG-Fusion (only when ENABLE_AGENT=true)
ENABLE_DECOMPOSITION = os.environ.get('ENABLE_DECOMPOSITION', 'false').lower() == 'true'
AGENT_DECOMP_MAX_SUBQUERIES = int(os.environ.get('AGENT_DECOMP_MAX_SUBQUERIES', '3'))
AGENT_DECOMP_MIN_WORDS = int(os.environ.get('AGENT_DECOMP_MIN_WORDS', '8'))
AGENT_FUSION_RRF_K = int(os.environ.get('AGENT_FUSION_RRF_K', '60'))

# =============================================================================
# 📈 PHASE 4: EVALUATION HARNESS
# =============================================================================

# LLM-as-Judge model (defaults to main LLM)
JUDGE_LLM_MODEL = os.environ.get('JUDGE_LLM_MODEL', LLM_MODEL_NAME)

# Default directory for benchmark JSON + markdown reports (under python/ if relative)
_EVAL_OUT = os.environ.get('EVAL_OUTPUT_DIR', '')
EVAL_OUTPUT_DIR = _EVAL_OUT if _EVAL_OUT else os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'evaluation',
    'reports',
)
EVAL_OUTPUT_DIR = os.path.normpath(EVAL_OUTPUT_DIR)

EVAL_JUDGE_TEMPERATURE = float(os.environ.get('EVAL_JUDGE_TEMPERATURE', '0.0'))
# Max concurrent LLM judge calls per benchmark run (faithfulness / relevancy / recall); clamped to 1–3 in judge.
EVAL_MAX_CONCURRENCY = int(os.environ.get('EVAL_MAX_CONCURRENCY', '1'))

# =============================================================================
# 📝 LOGGING
# =============================================================================

def print_current_config():
    """Print the current model configuration for debugging."""
    print("=" * 60)
    print("🔧 CURRENT RAG APPLICATION CONFIGURATION")
    print("=" * 60)
    print(f"LLM Model:        {LLM_MODEL_NAME}")
    print(f"Embedding Model:  {EMBEDDING_MODEL_NAME}")
    print(f"Embed batch size: {EMBED_BATCH_SIZE}")
    print(f"Reranker model:   {RERANKER_MODEL}")
    print(f"Reranker type:    {RERANKER_TYPE}")
    print(f"ColBERT model:    {COLBERT_MODEL}")
    print(f"Rerank top-M:     {RERANK_TOP_M}")
    print(f"Vector Size:      {DEFAULT_VECTOR_SIZE}")
    print(f"Ollama Host:      {OLLAMA_HOST_URL}")
    print(f"Ollama num_batch: {OLLAMA_NUM_BATCH}")
    print(f"Ollama num_ctx:   {OLLAMA_NUM_CTX or '(model default)'}")
    print(f"Ollama keep_alive:{OLLAMA_KEEP_ALIVE}")
    print(f"Qdrant Host:      {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection:       {DEFAULT_COLLECTION}")
    print(f"Fusion method:    {FUSION_METHOD}")
    print(f"KG mode:          {KG_EXTRACTION_MODE}")
    print(f"KG NP extractor:  {KG_NP_EXTRACTOR}")
    print(f"KG batch size:    {KG_TRIPLE_BATCH_SIZE}")
    print(f"KG concurrency:   {KG_EXTRACT_CONCURRENCY}")
    print(f"KG max chars:     {KG_EXTRACT_MAX_CHARS}")
    print(f"KG pretty JSON:   {KG_STORAGE_PRETTY}")
    print(f"KG LLM model:     {KG_LLM_MODEL}")
    print(f"Agent Enabled:    {ENABLE_AGENT}")
    print(f"Decomposition:    {ENABLE_DECOMPOSITION}")
    print(f"Community sums:   {ENABLE_COMMUNITY_SUMMARIES}")
    print(f"Community RR wt: {COMMUNITY_SUMMARY_WEIGHT}")
    print(f"Judge LLM:        {JUDGE_LLM_MODEL}")
    print(f"Eval reports dir: {EVAL_OUTPUT_DIR}")
    print("=" * 60)

# =============================================================================
# 🚀 QUICK MODEL SWITCHING PRESETS (Uncomment to use)
# =============================================================================

# Preset 1: Gemma 3 + Nomic v1.5 (Current)
# LLM_MODEL_NAME = 'gemma3:1b'
# EMBEDDING_MODEL_NAME = 'nomic-embed-text:v1.5'

# Preset 2: Qwen + Nomic v1.5 (Lightweight)
# LLM_MODEL_NAME = 'qwen3:0.6b'  
# EMBEDDING_MODEL_NAME = 'nomic-embed-text:v1.5'

# Preset 3: Phi3 + Nomic Latest
# LLM_MODEL_NAME = 'phi3:mini'
# EMBEDDING_MODEL_NAME = 'nomic-embed-text:latest'

# Preset 4: Local Sentence Transformers (if using local models)
# LLM_MODEL_NAME = 'gemma3:1b'
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
