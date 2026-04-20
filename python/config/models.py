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
LLM_MODEL_NAME = os.environ.get('LLM_MODEL', 'gemma3:4b')

# Embedding Model Configuration  
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text-v2-moe')

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
VECTOR_WEIGHT = float(os.environ.get('VECTOR_WEIGHT', '0.4'))   # α — vector similarity weight
BM25_WEIGHT = float(os.environ.get('BM25_WEIGHT', '0.3'))       # β — BM25 keyword weight
GRAPH_WEIGHT = float(os.environ.get('GRAPH_WEIGHT', '0.3'))     # γ — graph traversal weight

# Graph traversal
GRAPH_TRAVERSAL_DEPTH = int(os.environ.get('GRAPH_TRAVERSAL_DEPTH', '2'))

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
    print(f"Vector Size:      {DEFAULT_VECTOR_SIZE}")
    print(f"Ollama Host:      {OLLAMA_HOST_URL}")
    print(f"Qdrant Host:      {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection:       {DEFAULT_COLLECTION}")
    print(f"Agent Enabled:    {ENABLE_AGENT}")
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
