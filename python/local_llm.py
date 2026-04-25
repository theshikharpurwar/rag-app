# FILE: python/fixed_local_llm.py
# Using centralized model configuration

import argparse
import json
import logging
import re
import sys
import requests
import os
from qdrant_client import QdrantClient, models
from embeddings.ollama_embed import OllamaEmbedder # Use Ollama embedder instead of heavy ML libraries
from llm.ollama_llm import OllamaLLM
import time

# Import centralized configuration
from config import (
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    EMBED_BATCH_SIZE,
    DEFAULT_VECTOR_SIZE,
    QDRANT_HOST,
    QDRANT_PORT,
    OLLAMA_HOST_URL,
    OLLAMA_API_BASE,
    DEFAULT_COLLECTION,
    CONTEXT_RETRIEVAL_LIMIT,
    MAX_CONTEXT_CHAR_LIMIT,
    MAX_HISTORY_TOKENS,
    # Phase 2: Hybrid Retrieval
    ENABLE_KNOWLEDGE_GRAPH,
    RRF_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    GRAPH_WEIGHT,
    GRAPH_TRAVERSAL_DEPTH,
    COMMUNITY_SUMMARY_WEIGHT,
    INDICES_DIR,
    # Phase 3: Agentic self-correction
    ENABLE_AGENT,
    AGENT_MAX_RETRIES,
    AGENT_CONFIDENCE_THRESHOLD,
    AGENT_ROUTER_WEIGHT_BOOST,
    ENABLE_DECOMPOSITION,
    AGENT_DECOMP_MAX_SUBQUERIES,
    AGENT_DECOMP_MIN_WORDS,
    AGENT_FUSION_RRF_K,
    RERANK_TOP_M,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_config_to_stderr():
    """Print the current model configuration for debugging to stderr."""
    print("=" * 60, file=sys.stderr)
    print("🔧 CURRENT RAG APPLICATION CONFIGURATION", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"LLM Model:        {LLM_MODEL_NAME}", file=sys.stderr)
    print(f"Embedding Model:  {EMBEDDING_MODEL_NAME}", file=sys.stderr)
    print(f"Vector Size:      {DEFAULT_VECTOR_SIZE}", file=sys.stderr)
    print(f"Ollama Host:      {OLLAMA_HOST_URL}", file=sys.stderr)
    print(f"Qdrant Host:      {QDRANT_HOST}:{QDRANT_PORT}", file=sys.stderr)
    print(f"Collection:       {DEFAULT_COLLECTION}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


_RUNTIME_INITIALIZED = False

# --- Client/Model references (populated by init_runtime) ---
embedder = None
llm = None
reranker = None
SKIP_RERANKING = os.environ.get("SKIP_RERANKING", "false").lower() == "true"


def init_runtime():
    """
    Load embedder, optional reranker, and LLM. Idempotent; safe to call from
    main() and from evaluation harness imports.
    """
    global embedder, llm, reranker, _RUNTIME_INITIALIZED, SKIP_RERANKING
    if _RUNTIME_INITIALIZED:
        return
    print_config_to_stderr()
    SKIP_RERANKING = os.environ.get("SKIP_RERANKING", "false").lower() == "true"

    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embedder = OllamaEmbedder(model_name=EMBEDDING_MODEL_NAME, batch_size=EMBED_BATCH_SIZE)
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to load embedding model: {e}", exc_info=True)
        sys.exit("Embedding model failed to load")

    try:
        if SKIP_RERANKING:
            logger.info("Reranker disabled via SKIP_RERANKING=true.")
            reranker = None
        else:
            from reranker import get_reranker

            logger.info("Initializing reranker model...")
            reranker = get_reranker()
            logger.info("Reranker model loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
        reranker = None

    try:
        logger.info(f"Initializing LLM: {LLM_MODEL_NAME} targeting {OLLAMA_API_BASE}")
        llm = OllamaLLM(model_name=LLM_MODEL_NAME, api_base=OLLAMA_API_BASE)
        logger.info(f"LLM instance for '{LLM_MODEL_NAME}' created.")
    except Exception as e:
        logger.critical(f"CRITICAL: LLM init/check failed: {e}", exc_info=True)
        sys.exit("LLM failed to initialize")

    _RUNTIME_INITIALIZED = True

def check_dependencies():
    """Check if Qdrant and Ollama are running and provide guidance if not."""
    missing_services = []
    guidance = []

    # Check Qdrant
    try:
        logger.info(f"Checking connection to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        client.get_collections()
        logger.info("✓ Qdrant is running")
    except Exception as e:
        logger.error(f"✗ Qdrant connection failed: {e}")
        missing_services.append("Qdrant")
        guidance.append("""
To run Qdrant locally:
1. Install Docker
2. Run: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
Or install Qdrant directly from https://qdrant.tech/documentation/install/
        """)

    # Check Ollama
    try:
        logger.info(f"Checking connection to Ollama at {OLLAMA_HOST_URL}...")
        response = requests.get(f"{OLLAMA_HOST_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama is running")
            models = response.json().get("models", [])
            if not any(m.get("name", "").startswith(LLM_MODEL_NAME) for m in models):
                logger.warning(f"Model '{LLM_MODEL_NAME}' not found in Ollama")
                guidance.append(f"""
Ollama is running but the model '{LLM_MODEL_NAME}' is not available.
To pull the model, run: ollama pull {LLM_MODEL_NAME}
                """)
        else:
            logger.error(f"✗ Ollama returned unexpected status: {response.status_code}")
            missing_services.append("Ollama")
    except Exception as e:
        logger.error(f"✗ Ollama connection failed: {e}")
        missing_services.append("Ollama")
        guidance.append(f"""
To run Ollama locally:
1. Download from https://ollama.com/download
2. Install and start the Ollama application
3. Pull models: ollama pull {LLM_MODEL_NAME} && ollama pull {EMBEDDING_MODEL_NAME}
   (Model names are defined in config/models.py)
        """)

    if missing_services:
        logger.error(f"Required services not available: {', '.join(missing_services)}")
        for guide in guidance:
            logger.info(guide)
        return False
    
    return True

def get_qdrant_client():
    """Initializes and returns a Qdrant client."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=20)
        logger.info(f"Qdrant client created for '{QDRANT_HOST}'")
        return client
    except Exception as e:
        logger.error(f"Failed to create Qdrant client for '{QDRANT_HOST}': {str(e)}")
        raise ConnectionError(f"Could not connect to Qdrant service '{QDRANT_HOST}'") from e

# --- Core RAG Functions ---
def retrieve_context(
    client,
    collection_name,
    query,
    pdf_id_filter,
    limit=CONTEXT_RETRIEVAL_LIMIT,
    rerank=True,
):
    """
    Retrieve context from Qdrant for a specific PDF ID based on query.
    Fetches more results if reranker is enabled for this retrieval call.
    """
    if not embedder:
        raise RuntimeError("Embedding model is not loaded.")
    if not pdf_id_filter:
        logger.error("pdf_id_filter required")
        return []
    logger.info(f"retrieve_context called with pdf_id_filter: '{pdf_id_filter}'")
    
    try:
        # Use the query directly without manipulation
        query = query.strip()
        logger.info(f"Using direct query for retrieval: '{query}'")

        # 4. Generate embedding with the correct task_type
        try:
            query_embedding = embedder.encode_text([query], task_type="search_query")[0]
        except Exception as emb_error:
            logger.error(f"Embedding generation failed: {emb_error}", exc_info=True)
            return []  # Return empty results if embedding fails
        
        # Create filter
        qdrant_filter = models.Filter(must=[models.FieldCondition(key="pdf_id", match=models.MatchValue(value=pdf_id_filter))])
        
        # Fetch more results when reranking this path, then reduce to `limit`.
        retrieval_limit = limit * 4 if (reranker and rerank) else limit
        logger.info(f"Searching collection '{collection_name}' (limit={retrieval_limit}) with filter...")
        
        # Search with the original query
        try:
            search_response = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=qdrant_filter,
                limit=retrieval_limit,
                with_payload=True,
                score_threshold=0.1  # Lower threshold for reranking
            )
            search_results = search_response.points
            
            logger.info(f"Retrieved {len(search_results)} results from Qdrant for pdf_id '{pdf_id_filter}'.")
            
            # Apply reranking if enabled for this retrieval call.
            if reranker and rerank and len(search_results) > limit:
                logger.info(f"Reranking {len(search_results)} results to top {limit}...")
                search_results = reranker.rerank(query, search_results, top_k=limit)
                logger.info(f"After reranking: {len(search_results)} results")
            
        except Exception as search_error:
            logger.error(f"Qdrant search failed: {search_error}", exc_info=True)
            return []  # Return empty if search fails
        
        return search_results[:limit]
        
    except Exception as e: 
        logger.error(f"Qdrant retrieval error: {e}", exc_info=True)
        return []  # Return empty list

def format_context_for_llm(results):
    """Formats retrieved context for the LLM prompt and extracts sources."""
    context_str = ""
    sources = []
    if not results:
        return context_str, sources
    logger.info("Formatting context for LLM...")

    for i, hit in enumerate(results):
        try:
            payload = hit.payload if isinstance(hit.payload, dict) else {}
            text = payload.get("text", "")
            page = payload.get("page", "N/A")
            doc_name = payload.get("source", "Unknown Document")
            score = hit.score if hasattr(hit, 'score') else 0.0
            
            # Clean and format the text
            text = text.strip()
            if text:
                # Simpler source formatting for LLM consumption
                context_str += f"CONTENT FROM SOURCE {i+1} (Document: {doc_name}, Page: {page}, Score: {score:.3f}):\n{text}\n\n"
                
                sources.append({
                    "id": i + 1,
                    "page": page,
                    "document": doc_name,
                    "score": score,
                    "text": text,
                })
        except Exception as e: 
            logger.warning(f"Failed to format hit {i}: {e}")
    
    return context_str.strip(), sources

def estimate_tokens(text):
    return len(text.split())  # Basic token estimate


_LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\.|[-*\u2022])\s+")


def _strip_leading_list_marker(text: str) -> str:
    stripped = text.lstrip()
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) <= 1 or not any(_LEADING_LIST_MARKER_RE.match(ln) for ln in lines[1:]):
        return _LEADING_LIST_MARKER_RE.sub("", stripped, count=1)
    return text


def _postprocess_rag_answer(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^(ANSWER:?|Answer:?)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"User:.*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"Human:.*$", "", text, flags=re.DOTALL).strip()
    text = _strip_leading_list_marker(text)
    return text.strip()


def _build_rag_messages(query, context_str, chat_history):
    if len(context_str) > MAX_CONTEXT_CHAR_LIMIT:
        context_str = context_str[:MAX_CONTEXT_CHAR_LIMIT]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions about documents. "
                "Answer only from the provided document extracts. Be concise and accurate. "
                "Reply in plain prose — do not begin the answer with a numbered list marker "
                "(for example '1.') or a bullet ('-', '*'), and do not wrap a single fact in a list.\n\n"
                f"Document extracts:\n{context_str}"
            ),
        }
    ]
    if chat_history:
        token_count = 0
        valid_turns = []
        for turn in reversed(chat_history):
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            turn_tokens = estimate_tokens(user_text + assistant_text)
            if token_count + turn_tokens > MAX_HISTORY_TOKENS:
                break
            valid_turns.insert(0, turn)
            token_count += turn_tokens
        for turn in valid_turns:
            messages.append({"role": "user", "content": turn.get("user", "")})
            messages.append({"role": "assistant", "content": turn.get("assistant", "")})
    messages.append({"role": "user", "content": query})
    return messages


def generate_rag_response(query, context_str, chat_history=None, system_instruction=None):
    """Generates a response using the LLM /api/chat with structured message roles."""
    if not llm:
        raise RuntimeError("LLM is not initialized.")

    # Handle missing context
    if not context_str:
        logger.warning("No context provided to LLM.")
        return "I couldn't find relevant information in the document to answer this question. Could you try rephrasing or asking about something else from the document?"

    if len(context_str) > MAX_CONTEXT_CHAR_LIMIT:
        logger.warning(f"Context length ({len(context_str)}) exceeds limit, truncating.")
        context_str = context_str[:MAX_CONTEXT_CHAR_LIMIT]

    messages = _build_rag_messages(query, context_str, chat_history)

    logger.info(f"Sending {len(messages)} messages to LLM (system + {len(chat_history or [])} history turns + query)")

    try:
        response = llm.generate_response(
            prompt=query,           # fallback only
            messages=messages,      # structured chat
            max_tokens=2000,
            temperature=0.7
        )

        if not response:
            logger.error("LLM returned empty response")
            return "I wasn't able to generate a proper response. Please try again with a different question."

        logger.info("Received response from LLM.")
        return _postprocess_rag_answer(response)

    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        return "I encountered an error while processing your question. Please try again."


def generate_rag_response_stream(query, context_str, chat_history=None, system_instruction=None):
    """
    Yields raw text deltas from the LLM. Caller applies _postprocess_rag_answer to the
    joined string for the final answer (e.g. SSE done event).
    """
    if not llm:
        raise RuntimeError("LLM is not initialized.")
    if not context_str:
        logger.warning("No context provided to LLM (stream).")
        yield (
            "I couldn't find relevant information in the document to answer this question. "
            "Could you try rephrasing or asking about something else from the document?"
        )
        return
    ctx = context_str
    if len(ctx) > MAX_CONTEXT_CHAR_LIMIT:
        logger.warning("Context length (%s) exceeds limit, truncating.", len(ctx))
        ctx = ctx[:MAX_CONTEXT_CHAR_LIMIT]
    messages = _build_rag_messages(query, ctx, chat_history)
    try:
        for chunk in llm.generate_response_stream(
            prompt=query, messages=messages, max_tokens=2000, temperature=0.7
        ):
            if chunk:
                yield chunk
    except Exception as e:
        logger.error("LLM streaming failed: %s", e, exc_info=True)
        yield "I encountered an error while processing your question. Please try again."


# --- End Core RAG Functions ---


def make_retrieve_pipeline(pdf_id: str, collection_name: str = DEFAULT_COLLECTION):
    """
    Build hybrid (RRF) or vector-only retrieve_fn for one document.
    Used by main() and by the Phase 4 evaluation runner.

    Returns:
        (retrieve_fn, qdrant_client) where retrieve_fn(query_text, weights, route=None) -> (context_str, sources).
    """
    qdrant_client = get_qdrant_client()

    bm25_index = None
    knowledge_graph = None
    graph_retriever = None

    bm25_path = os.path.join(INDICES_DIR, f"{pdf_id}_bm25.pkl")
    kg_path = os.path.join(INDICES_DIR, f"{pdf_id}_graph.json")

    try:
        from retrieval.bm25_search import BM25Index

        bm25_index = BM25Index()
        if not bm25_index.load(bm25_path):
            bm25_index = None
            logger.info(f"[Hybrid] No BM25 index for PDF {pdf_id}, skipping BM25 path")
    except Exception as e:
        logger.warning(f"[Hybrid] BM25 load failed: {e}")
        bm25_index = None

    if ENABLE_KNOWLEDGE_GRAPH:
        try:
            from retrieval.knowledge_graph import KnowledgeGraph
            from retrieval.graph_retrieval import GraphRetriever

            knowledge_graph = KnowledgeGraph()
            if knowledge_graph.load(kg_path):
                graph_retriever = GraphRetriever(
                    knowledge_graph,
                    llm=llm,
                    traversal_depth=GRAPH_TRAVERSAL_DEPTH,
                )
                logger.info(
                    f"[Hybrid] KG loaded: {knowledge_graph.num_nodes} nodes, "
                    f"{knowledge_graph.num_edges} edges"
                )
            else:
                knowledge_graph = None
                logger.info(f"[Hybrid] No KG for PDF {pdf_id}, skipping graph path")
        except Exception as e:
            logger.warning(f"[Hybrid] KG load failed: {e}")
            knowledge_graph = None
            graph_retriever = None

    def _retrieve_and_format(query_text, weights, route=None):
        v_w, b_w, g_w = weights
        has_hybrid_local = bm25_index is not None or graph_retriever is not None
        vector_limit = RERANK_TOP_M if (has_hybrid_local and reranker) else CONTEXT_RETRIEVAL_LIMIT
        retrieved = retrieve_context(
            qdrant_client,
            collection_name,
            query_text,
            pdf_id,
            limit=vector_limit,
            rerank=not has_hybrid_local,
        )

        if has_hybrid_local:
            try:
                from retrieval.rrf_fusion import hybrid_retrieve

                community_results = None
                community_w = 0.0
                v_adj, b_adj, g_adj = v_w, b_w, g_w
                use_community = route != "specific"
                if (
                    use_community
                    and graph_retriever is not None
                    and knowledge_graph is not None
                    and knowledge_graph.community_summaries
                ):
                    community_results = graph_retriever.global_search(
                        query_text, embedder=embedder, top_k=3
                    )
                    if community_results:
                        community_w = float(COMMUNITY_SUMMARY_WEIGHT)
                        denom = max(v_adj + b_adj + g_adj, 1e-9)
                        scale = (1.0 - community_w) / denom
                        v_adj *= scale
                        b_adj *= scale
                        g_adj *= scale

                fused = hybrid_retrieve(
                    query=query_text,
                    pdf_id=pdf_id,
                    vector_results=retrieved,
                    bm25_index=bm25_index,
                    knowledge_graph=knowledge_graph,
                    graph_retriever=graph_retriever,
                    vector_weight=v_adj,
                    bm25_weight=b_adj,
                    graph_weight=g_adj,
                    community_results=community_results,
                    community_weight=community_w,
                    rrf_k=RRF_K,
                    top_k=vector_limit,
                )
                if reranker and len(fused) > CONTEXT_RETRIEVAL_LIMIT:
                    logger.info(
                        f"[Hybrid] Reranking fused results {len(fused)} -> {CONTEXT_RETRIEVAL_LIMIT}"
                    )
                    fused = reranker.rerank(query_text, fused, top_k=CONTEXT_RETRIEVAL_LIMIT)

                ctx = ""
                srcs = []
                for i, r in enumerate(fused):
                    text = r.get("text", "")
                    page = r.get("page", "N/A")
                    source = r.get("source", "Unknown")
                    score = r.get("score", 0.0)
                    methods = r.get("retrieval_methods", ["unknown"])
                    if text.strip():
                        ctx += (
                            f"CONTENT FROM SOURCE {i+1} "
                            f"(Document: {source}, Page: {page}, "
                            f"Score: {score:.4f}, Methods: {', '.join(methods)}):\n"
                            f"{text}\n\n"
                        )
                        srcs.append(
                            {
                                "id": i + 1,
                                "page": page,
                                "document": source,
                                "score": score,
                                "methods": methods,
                                "text": text,
                            }
                        )

                logger.info(
                    f"[Hybrid] Fused context: {len(srcs)} chunks from {len(fused)} fused results"
                )
                return ctx.strip(), srcs
            except Exception as e:
                logger.error(
                    f"[Hybrid] Fusion failed, falling back to vector-only: {e}",
                    exc_info=True,
                )

        return format_context_for_llm(retrieved)

    return _retrieve_and_format, qdrant_client


# --- Main Execution ---
def main():
    # *** NOTE: Removed all Flask app related code ***
    parser = argparse.ArgumentParser(description='Process a query for RAG using local LLM')
    parser.add_argument('query', type=str, help='The query to process')
    parser.add_argument('--collection_name', type=str, default=DEFAULT_COLLECTION, help='Qdrant collection name')
    parser.add_argument('--pdf_id', required=True, help='MongoDB ID of the PDF to filter by')
    parser.add_argument('--history', type=str, default='[]', help='Chat history as a JSON string')
    args = parser.parse_args()

    logger.info("Checking required services...")
    if not check_dependencies():
        result = {
            "answer": "Error: Required services (Qdrant and/or Ollama) are not available. "
                     "Please check the console output for instructions on how to install and run them.",
            "sources": []
        }
        print(json.dumps(result))
        sys.exit(1)

    init_runtime()

    if not embedder or not llm:
        logger.critical("Models did not load.")
        result = {"answer": "Error: AI models failed.", "sources": []}
        print(json.dumps(result))
        sys.exit(1)

    try: # Parse history
        chat_history = json.loads(args.history) #... validate ...
    except Exception as e:
        logger.error(f"Invalid history: {e}")
        chat_history = []

    result = {}
    try:
        _retrieve_and_format, _qclient = make_retrieve_pipeline(
            args.pdf_id, args.collection_name
        )
        logger.info(f"Processing query for PDF ID '{args.pdf_id}': {args.query}")

        base_weights = (VECTOR_WEIGHT, BM25_WEIGHT, GRAPH_WEIGHT)

        if ENABLE_AGENT:
            from agent import (
                QueryRouter,
                AnswerGrader,
                AgenticRAG,
                QueryDecomposer,
                RAGFusion,
            )
            logger.info("[Agent] ENABLE_AGENT=true — running agentic control loop")
            decomposer = None
            fusion = None
            if ENABLE_DECOMPOSITION:
                decomposer = QueryDecomposer(
                    llm,
                    max_subqueries=AGENT_DECOMP_MAX_SUBQUERIES,
                    min_words=AGENT_DECOMP_MIN_WORDS,
                )
                fusion = RAGFusion(
                    _retrieve_and_format,
                    rrf_k=AGENT_FUSION_RRF_K,
                    top_k=CONTEXT_RETRIEVAL_LIMIT,
                )
                logger.info("[Agent] ENABLE_DECOMPOSITION=true — RAG-Fusion path enabled")
            agent = AgenticRAG(
                llm=llm,
                router=QueryRouter(llm, weight_boost=AGENT_ROUTER_WEIGHT_BOOST),
                grader=AnswerGrader(llm, threshold=AGENT_CONFIDENCE_THRESHOLD),
                retrieve_fn=_retrieve_and_format,
                generate_fn=lambda q, ctx, hist: generate_rag_response(q, ctx, hist),
                max_retries=AGENT_MAX_RETRIES,
                base_weights=base_weights,
                decomposer=decomposer,
                fusion=fusion,
            )
            ar = agent.run(args.query, chat_history)
            logger.info(f"[Agent] trace: {ar.trace}")
            result = {"answer": ar.answer, "sources": ar.sources}
        else:
            context_str, sources = _retrieve_and_format(args.query, base_weights)
            answer = generate_rag_response(args.query, context_str, chat_history)
            result = {"answer": answer, "sources": sources}

    except (ConnectionError, RuntimeError) as e: 
        logger.error(f"Error: {e}", exc_info=True)
        result = {"answer": f"Error: Unable to connect to necessary services. Please ensure Qdrant and Ollama are running properly.", "sources": []}
    except Exception as e: 
        logger.error(f"Unexpected error: {e}", exc_info=True)
        result = {
            "answer": "I encountered an unexpected error while processing your question. Please try again or rephrase your question.", 
            "sources": []
        }

    # Ensure result is valid JSON before printing
    if not result or not isinstance(result, dict) or "answer" not in result:
        result = {"answer": "The system produced an invalid response. Please try again with a different question.", "sources": []}
    
    # Make sure we always have sources field, even if empty
    if "sources" not in result:
        result["sources"] = []
        
    print(json.dumps(result)) # Output JSON result
    sys.exit(0)

if __name__ == "__main__":
    main() 