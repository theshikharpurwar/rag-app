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
    INDICES_DIR,
)

# Import reranker
from reranker import SimpleReranker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print current configuration for debugging to stderr instead of stdout
import sys
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

print_config_to_stderr()

# --- Configuration (now imported from central config) ---
# --- End Configuration ---

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

# --- Client/Model Initialization ---
embedder = None # Changed variable name for clarity
llm = None
reranker = None  # Add reranker

try:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    # Use Ollama embedder (no heavy ML dependencies)
    embedder = OllamaEmbedder(model_name=EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded.")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to load embedding model: {e}", exc_info=True)
    sys.exit("Embedding model failed to load")

SKIP_RERANKING = os.environ.get("SKIP_RERANKING", "false").lower() == "true"

try:
    if SKIP_RERANKING:
        logger.info("Reranker disabled via SKIP_RERANKING=true.")
        reranker = None
    else:
        logger.info("Initializing reranker model...")
        reranker = SimpleReranker()
        logger.info("Reranker model loaded successfully.")
except Exception as e:
    logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
    reranker = None  # Graceful degradation

try:
    logger.info(f"Initializing LLM: {LLM_MODEL_NAME} targeting {OLLAMA_API_BASE}")
    llm = OllamaLLM(model_name=LLM_MODEL_NAME, api_base=OLLAMA_API_BASE)
    logger.info(f"LLM instance for '{LLM_MODEL_NAME}' created.")
    
    # Check connection to host Ollama
    try:
        test_response = requests.get(f"{OLLAMA_HOST_URL}/api/tags", timeout=5)
        if test_response.status_code == 200:
            logger.info(f"Successfully connected to Ollama at {OLLAMA_HOST_URL}")
            # Check if target model exists
            available_models = [m.get('name') for m in test_response.json().get('models', [])]
            if LLM_MODEL_NAME not in available_models and f"{LLM_MODEL_NAME}:latest" not in available_models:
                logger.warning(f"Model '{LLM_MODEL_NAME}' not found in host Ollama models: {available_models}. Please pull it.")
        else:
            logger.warning(f"Connected to Ollama host {OLLAMA_HOST_URL} but got status {test_response.status_code}.")
    except requests.exceptions.RequestException as conn_err:
         logger.error(f"Could not connect to Ollama at {OLLAMA_HOST_URL}. Is Ollama running on the host? Error: {conn_err}")
         # Consider exiting if LLM is mandatory: sys.exit("Ollama connection failed")
except Exception as e:
     logger.critical(f"CRITICAL: LLM init/check failed: {e}", exc_info=True)
     sys.exit("LLM failed to initialize")

def get_qdrant_client():
    """Initializes and returns a Qdrant client."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=20)
        client.get_collections()
        logger.info(f"Connected to Qdrant service '{QDRANT_HOST}'")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant service '{QDRANT_HOST}': {str(e)}")
        raise ConnectionError(f"Could not connect to Qdrant service '{QDRANT_HOST}'") from e
# --- End Client/Model Initialization ---

# --- Core RAG Functions ---
def retrieve_context(client, collection_name, query, pdf_id_filter, limit=CONTEXT_RETRIEVAL_LIMIT):
    """
    Retrieve context from Qdrant for a specific PDF ID based on query.
    Fetches more results if reranker is available for better filtering.
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
        
        # Fetch more results if reranker is available (multiply by 4 for better reranking)
        retrieval_limit = limit * 4 if reranker else limit
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
            
            # Apply reranking if available
            if reranker and len(search_results) > limit:
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
    
    # Sort results by score to prioritize most relevant contexts
    sorted_results = sorted(results, key=lambda x: x.score if hasattr(x, 'score') else 0.0, reverse=True)
    
    for i, hit in enumerate(sorted_results):
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
                    "score": score
                })
        except Exception as e: 
            logger.warning(f"Failed to format hit {i}: {e}")
    
    return context_str.strip(), sources

def estimate_tokens(text):
    return len(text.split())  # Basic token estimate

def generate_rag_response(query, context_str, chat_history=None, system_instruction=None):
    """Generates a response using the LLM /api/chat with structured message roles."""
    if not llm:
        raise RuntimeError("LLM is not initialized.")

    # Handle missing context
    if not context_str:
        logger.warning("No context provided to LLM.")
        return "I couldn't find relevant information in the document to answer this question. Could you try rephrasing or asking about something else from the document?"

    # Truncate if too long
    if len(context_str) > MAX_CONTEXT_CHAR_LIMIT:
        logger.warning(f"Context length ({len(context_str)}) exceeds limit, truncating.")
        context_str = context_str[:MAX_CONTEXT_CHAR_LIMIT]

    # Build structured messages list for /api/chat
    messages = []

    # System message: role + document context
    messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions about documents. "
            "Answer only from the provided document extracts. Be concise and accurate.\n\n"
            f"Document extracts:\n{context_str}"
        )
    })

    # Inject chat history as real user/assistant turns (most recent MAX_HISTORY_TOKENS worth)
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
            messages.append({"role": "user",      "content": turn.get("user", "")})
            messages.append({"role": "assistant",  "content": turn.get("assistant", "")})

    # Current user question
    messages.append({"role": "user", "content": query})

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
        # Clean up artefacts
        response = re.sub(r'^(ANSWER:?|Answer:?)\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'User:.*$', '', response, flags=re.DOTALL).strip()
        response = re.sub(r'Human:.*$', '', response, flags=re.DOTALL).strip()
        return response.strip()

    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        return "I encountered an error while processing your question. Please try again."


def improve_response_structure(text, query):
    """Improve the structure of the response without changing the content."""
    try:
        logger.info("Improving response structure...")
        
        # If response is already well-structured, don't modify it
        if re.search(r'#+\s+\w+', text) and re.search(r'\n\n', text) and len(text.split('\n\n')) > 2:
            logger.info("Response already has good structure, keeping as is")
            return text
            
        # Split into paragraphs
        paragraphs = re.split(r'\n{2,}', text)
        
        # Extract potential title from the query
        query_words = query.lower().split()
        key_question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'describe', 'list']
        
        title = None
        if any(word in query_words for word in key_question_words):
            # Format query as title if it's a question
            title = query.strip()
            if not title.endswith('?'):
                title = title + '?'
            title = title[0].upper() + title[1:]
        else:
            # Otherwise, use the first few words of the response
            first_para = paragraphs[0] if paragraphs else text
            title_words = first_para.split()[:6]
            title = ' '.join(title_words) + '...'
        
        # Build the structured response
        structured_text = [f"# {title}\n"]
        
        # Add introduction
        if paragraphs and len(paragraphs) >= 1:
            intro = paragraphs[0]
            structured_text.append(intro + "\n")
        
        # Process the rest of the paragraphs
        remaining_paragraphs = paragraphs[1:] if len(paragraphs) > 1 else []
        
        # Add sections for longer responses
        if len(remaining_paragraphs) >= 2:
            # Look for patterns that might indicate list items
            list_pattern = r'^(\d+\.|\-|\*)\s+'
            
            current_section = []
            for i, para in enumerate(remaining_paragraphs):
                # Check if this paragraph could be a section heading
                is_short = len(para.split()) <= 8
                ends_with_colon = para.strip().endswith(':')
                has_list_marker = bool(re.match(list_pattern, para.strip()))
                
                if (is_short and ends_with_colon) and not has_list_marker and i < len(remaining_paragraphs) - 1:
                    # This looks like a section heading
                    if current_section:
                        structured_text.append('\n'.join(current_section) + "\n")
                        current_section = []
                    # Format as a heading
                    section_title = para.strip().rstrip(':')
                    structured_text.append(f"\n## {section_title}\n")
                elif has_list_marker:
                    # Preserve list formatting
                    if current_section:
                        structured_text.append('\n'.join(current_section) + "\n")
                        current_section = []
                    structured_text.append(para + "\n")
                else:
                    current_section.append(para)
            
            # Add any remaining section content
            if current_section:
                structured_text.append('\n'.join(current_section))
        else:
            # For shorter responses, just add the remaining paragraphs
            structured_text.extend(remaining_paragraphs)
        
        # Join everything with appropriate spacing
        result = '\n\n'.join([p for p in structured_text if p])
        
        # Ensure consistent spacing for list items
        result = re.sub(r'(\n\s*\n)(\d+\.|\-|\*)\s+', r'\n\n\2 ', result)
        
        # Clean up any excessive newlines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        logger.info("Response structure improved")
        return result
        
    except Exception as e:
        logger.error(f"Error improving response structure: {e}")
        # If anything goes wrong, return the original text
        return text
# --- End Core RAG Functions ---

# --- Main Execution ---
def main():
    # *** NOTE: Removed all Flask app related code ***
    parser = argparse.ArgumentParser(description='Process a query for RAG using local LLM')
    parser.add_argument('query', type=str, help='The query to process')
    parser.add_argument('--collection_name', type=str, default=DEFAULT_COLLECTION, help='Qdrant collection name')
    parser.add_argument('--pdf_id', required=True, help='MongoDB ID of the PDF to filter by')
    parser.add_argument('--history', type=str, default='[]', help='Chat history as a JSON string')
    args = parser.parse_args()

    # Check if dependencies are available
    logger.info("Checking required services...")
    if not check_dependencies():
        result = {
            "answer": "Error: Required services (Qdrant and/or Ollama) are not available. "
                     "Please check the console output for instructions on how to install and run them.",
            "sources": []
        }
        print(json.dumps(result))
        sys.exit(1)

    if not embedder or not llm: # Check models loaded
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
        qdrant_client = get_qdrant_client() # Connect to Qdrant
        logger.info(f"Processing query for PDF ID '{args.pdf_id}': {args.query}")

        # 1. Vector retrieval (Path A — always runs)
        retrieved_context = retrieve_context(qdrant_client, args.collection_name, args.query, args.pdf_id, limit=CONTEXT_RETRIEVAL_LIMIT)

        # 2. Load BM25 index and Knowledge Graph for hybrid retrieval (Phase 2)
        bm25_index = None
        knowledge_graph = None
        graph_retriever = None

        bm25_path = os.path.join(INDICES_DIR, f"{args.pdf_id}_bm25.pkl")
        kg_path = os.path.join(INDICES_DIR, f"{args.pdf_id}_graph.json")

        # Try to load BM25 index
        try:
            from retrieval.bm25_search import BM25Index
            bm25_index = BM25Index()
            if not bm25_index.load(bm25_path):
                bm25_index = None
                logger.info(f"[Hybrid] No BM25 index for PDF {args.pdf_id}, skipping BM25 path")
        except Exception as e:
            logger.warning(f"[Hybrid] BM25 load failed: {e}")
            bm25_index = None

        # Try to load Knowledge Graph and set up graph retriever
        if ENABLE_KNOWLEDGE_GRAPH:
            try:
                from retrieval.knowledge_graph import KnowledgeGraph
                from retrieval.graph_retrieval import GraphRetriever
                knowledge_graph = KnowledgeGraph()
                if knowledge_graph.load(kg_path):
                    graph_retriever = GraphRetriever(
                        knowledge_graph, llm=llm,
                        traversal_depth=GRAPH_TRAVERSAL_DEPTH
                    )
                    logger.info(f"[Hybrid] KG loaded: {knowledge_graph.num_nodes} nodes, "
                                 f"{knowledge_graph.num_edges} edges")
                else:
                    knowledge_graph = None
                    logger.info(f"[Hybrid] No KG for PDF {args.pdf_id}, skipping graph path")
            except Exception as e:
                logger.warning(f"[Hybrid] KG load failed: {e}")
                knowledge_graph = None
                graph_retriever = None

        # 3. Hybrid retrieval: fuse vector + BM25 + graph via RRF
        has_hybrid = bm25_index is not None or graph_retriever is not None
        if has_hybrid:
            try:
                from retrieval.rrf_fusion import hybrid_retrieve
                fused_results = hybrid_retrieve(
                    query=args.query,
                    pdf_id=args.pdf_id,
                    vector_results=retrieved_context,
                    bm25_index=bm25_index,
                    knowledge_graph=knowledge_graph,
                    graph_retriever=graph_retriever,
                    vector_weight=VECTOR_WEIGHT,
                    bm25_weight=BM25_WEIGHT,
                    graph_weight=GRAPH_WEIGHT,
                    rrf_k=RRF_K,
                    top_k=CONTEXT_RETRIEVAL_LIMIT,
                )

                # Format fused results for LLM
                context_str = ""
                sources = []
                for i, r in enumerate(fused_results):
                    text = r.get("text", "")
                    page = r.get("page", "N/A")
                    source = r.get("source", "Unknown")
                    score = r.get("score", 0.0)
                    methods = r.get("retrieval_methods", ["unknown"])
                    if text.strip():
                        context_str += (f"CONTENT FROM SOURCE {i+1} "
                                       f"(Document: {source}, Page: {page}, "
                                       f"Score: {score:.4f}, Methods: {', '.join(methods)}):\n"
                                       f"{text}\n\n")
                        sources.append({"id": i + 1, "page": page, "document": source,
                                        "score": score, "methods": methods})

                context_str = context_str.strip()
                logger.info(f"[Hybrid] Fused context: {len(sources)} chunks from "
                            f"{len(fused_results)} fused results")
            except Exception as e:
                logger.error(f"[Hybrid] Fusion failed, falling back to vector-only: {e}",
                             exc_info=True)
                has_hybrid = False

        # Fallback: vector-only context (original Phase 1 path)
        if not has_hybrid:
            context_str, sources = format_context_for_llm(retrieved_context)

        # 4. Generate response
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