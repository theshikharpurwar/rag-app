# FILE: python/local_llm.py
# (Corrected version: Removed Flask code, added imports, targets host Ollama)

import argparse
import json
import logging
import re
import sys
import requests
import os  # <--- FIX: Added import os
from qdrant_client import QdrantClient, models # Import models for Filter
from sentence_transformers import SentenceTransformer
# Assuming OllamaLLM class is correctly defined in llm/ollama_llm.py
from llm.ollama_llm import OllamaLLM

# Configure logging (ensure level is appropriate for debugging if needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Using localhost instead of Docker service names for local development
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost") # Changed from "qdrant" to "localhost"
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
# Ollama runs on the HOST machine
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434") # Changed from "http://host.docker.internal:11434"
OLLAMA_API_BASE = f"{OLLAMA_HOST_URL}/api"

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'tinyllama'
DEFAULT_COLLECTION = 'documents'
CONTEXT_RETRIEVAL_LIMIT = 5
MAX_CONTEXT_CHAR_LIMIT = 4096 # Keep updated limit
MAX_HISTORY_TOKENS = 500
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
        guidance.append("""
To run Ollama locally:
1. Download from https://ollama.com/download
2. Install and start the Ollama application
3. Pull a model: ollama pull tinyllama
        """)

    if missing_services:
        logger.error(f"Required services not available: {', '.join(missing_services)}")
        for guide in guidance:
            logger.info(guide)
        return False
    
    return True

# --- Client/Model Initialization ---
embedding_model = None
llm = None
try:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded.")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to load embedding model: {e}", exc_info=True)
    sys.exit("Embedding model failed to load")

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
    """Retrieve context from Qdrant for a specific PDF ID based on query."""
    if not embedding_model: raise RuntimeError("Embedding model is not loaded.")
    if not pdf_id_filter: logger.error("pdf_id_filter required"); return []
    logger.info(f"retrieve_context called with pdf_id_filter: '{pdf_id_filter}'")
    try:
        query_embedding = embedding_model.encode(query).tolist()
        qdrant_filter = models.Filter(must=[models.FieldCondition(key="pdf_id", match=models.MatchValue(value=pdf_id_filter))])
        logger.info(f"Constructed Qdrant Filter: {qdrant_filter.model_dump_json(indent=2)}")
        logger.info(f"Searching collection '{collection_name}' (limit={limit}) with filter...")
        # Note: client.search might be deprecated, use client.query_points if client version requires
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        logger.info(f"Retrieved {len(search_results)} results from Qdrant for pdf_id '{pdf_id_filter}'.")
        valid_results = [ hit for hit in search_results if hit.payload and isinstance(hit.payload.get("text"), str) and hit.payload.get("text").strip() ]
        if len(valid_results) < len(search_results): logger.warning(f"Filtered out {len(search_results) - len(valid_results)} results lacking text payload.")
        return valid_results
    except Exception as e: logger.error(f"Qdrant retrieval error: {e}", exc_info=True); return []

def format_context_for_llm(results):
    """Formats retrieved context for the LLM prompt and extracts sources."""
    context_str = ""; sources = []
    if not results: return context_str, sources
    logger.info("Formatting context for LLM...")
    for i, hit in enumerate(results):
        try:
            payload = hit.payload if isinstance(hit.payload, dict) else {}
            text = payload.get("text", "")
            page = payload.get("page", "N/A")
            doc_name = payload.get("source", "Unknown Document")
            score = hit.score if hasattr(hit, 'score') else 0.0
            if text and text.strip():
                context_str += f"Source [{i+1}] (Page: {page}, Document: {doc_name}, Score: {score:.3f}):\n{text}\n\n"
                sources.append({"id": i + 1, "page": page, "document": doc_name, "score": score})
        except Exception as e: logger.warning(f"Failed format hit {i}: {e}")
    return context_str.strip(), sources

def estimate_tokens(text): return len(text.split()) # Basic token estimate

def generate_rag_response(query, context_str, chat_history=None, system_instruction=None):
    """Generates a response using the LLM with context, history, and citation attempts."""
    if not llm: raise RuntimeError("LLM is not initialized.")
    if not system_instruction:
        system_instruction = ( # Default prompt with citation instruction
            "You are an AI assistant answering questions based ONLY on the provided document context. "
            "Use ONLY the information presented in the 'Source [n]' sections. "
            "Do not use any external knowledge or make assumptions. "
            "If the answer cannot be found in the context, state clearly: 'The provided context does not contain the answer to this question.' "
            "When using information from a source, **you MUST cite the source number** (e.g., [1], [2]) at the end of the sentence(s) referencing that source. "
            "Be concise."
        )
    history_str = ""; # Format history
    if chat_history:
        token_count = 0
        for turn in reversed(chat_history):
            turn_text = f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
            turn_tokens = estimate_tokens(turn_text)
            if token_count + turn_tokens > MAX_HISTORY_TOKENS: break
            history_str = turn_text + history_str; token_count += turn_tokens
        if history_str: history_str = f"Previous Conversation History:\n---\n{history_str.strip()}\n---\n\n"
    if context_str and len(context_str) > MAX_CONTEXT_CHAR_LIMIT: # Truncate
        logger.warning(f"Context length ({len(context_str)}) exceeds limit ({MAX_CONTEXT_CHAR_LIMIT}), truncating."); context_str = context_str[:MAX_CONTEXT_CHAR_LIMIT] + "..."
    if not context_str:
         prompt_for_llm = f"{history_str}Instruction: {system_instruction}\n\nUser Question: {query}\n\nContext: [No relevant context provided]\n\nAssistant Answer:"
         logger.warning("No context provided to LLM.")
    else:
        prompt_for_llm = ( f"{history_str}Instruction: {system_instruction}\n\nContext:\n---\n{context_str}\n---\n\nUser Question: {query}\n\nAssistant Answer (Cite sources like [1], [2]):" )
    logger.info(f"Sending request to LLM '{LLM_MODEL_NAME}' at {OLLAMA_API_BASE}...")
    try:
        response = llm.generate_response(prompt_for_llm) # Assumes OllamaLLM uses its internal api_base
        logger.info("Received response from LLM.")
        response = response.split("Assistant Answer")[-1].strip(':').strip()
        return response
    except Exception as e: logger.error(f"LLM generation failed: {e}", exc_info=True); return "LLM generation error."
# --- End Core RAG Functions ---


# --- Command Processing ---
def detect_command_type(query): # ... (same as previous) ...
    query_lower = query.lower().strip()
    if query_lower.startswith("extract keywords"): return "keywords"
    if any(term in query_lower for term in ["explain", "topics"]): return "explain_topics"
    if "list topics" in query_lower: return "topics"
    if query_lower.startswith("summarize"): return "summary"
    if query_lower.startswith("define "): match = re.match(r"define\s+(.+)", query, re.IGNORECASE); term = match.group(1).strip() if match else None; 
    if term: return "definition", term
    if "generate questions" in query_lower: return "questions"
    return "regular_query"

def process_command(client, collection_name, query, chat_history, pdf_id_filter, command_type, command_details=None):
    # ... (same logic as previous - defines prompts/limits, calls retrieve_context, calls generate_rag_response) ...
    logger.info(f"Processing command: {command_type} for PDF ID: {pdf_id_filter}")
    retrieval_query = query; system_instruction = None; limit = 5; query_for_llm = query;
    # Define specifics based on command_type...
    if command_type == "summary": retrieval_query = "Overall summary"; limit = 15; system_instruction = "Summarize..."; query_for_llm = "Summarize."
    elif command_type == "definition": term=command_details; limit = 7; retrieval_query = f"Define '{term}'"; system_instruction = f"Define '{term}'..."; query_for_llm = f"Define '{term}'."
    elif command_type == "questions": limit = 10; system_instruction = "Generate questions..."; query_for_llm = "Generate questions."
    elif command_type == "topics": limit = 15; system_instruction = "List topics..."; query_for_llm = "List topics."
    elif command_type == "explain_topics": limit = 15; system_instruction = "Explain topics..."; query_for_llm = "Explain topics."
    elif command_type == "keywords": limit = 10; system_instruction = "Extract keywords..."; query_for_llm = "Extract keywords."
    else: return process_regular_query_command(client, collection_name, query, chat_history, pdf_id_filter)

    retrieved_context = retrieve_context(client, collection_name, retrieval_query, pdf_id_filter, limit=limit)
    context_str, sources = format_context_for_llm(retrieved_context)
    if not context_str: answer = f"Could not retrieve context for command '{command_type}' from PDF '{pdf_id_filter}'."; return {"answer": answer, "sources": []}
    answer = generate_rag_response(query_for_llm, context_str, chat_history, system_instruction)
    return {"answer": answer, "sources": sources}

def process_regular_query_command(client, collection_name, query, chat_history, pdf_id_filter):
    # ... (same logic as previous - calls retrieve_context, calls generate_rag_response with default prompt) ...
    logger.info(f"Processing regular query for PDF ID {pdf_id_filter}: {query[:50]}...")
    retrieved_context = retrieve_context(client, collection_name, query, pdf_id_filter, limit=CONTEXT_RETRIEVAL_LIMIT)
    context_str, sources = format_context_for_llm(retrieved_context)
    answer = generate_rag_response(query, context_str, chat_history, system_prompt=None)
    return {"answer": answer, "sources": sources}
# --- End Command Processing ---


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

    if not embedding_model or not llm: # Check models loaded
         logger.critical("Models did not load."); result = {"answer": "Error: AI models failed.", "sources": []}
         print(json.dumps(result)); sys.exit(1)

    try: # Parse history
        chat_history = json.loads(args.history); #... validate ...
    except Exception as e: logger.error(f"Invalid history: {e}"); chat_history = []

    result = {}
    try:
        qdrant_client = get_qdrant_client() # Connect to Qdrant
        command_info = detect_command_type(args.query)
        command_name = command_info[0] if isinstance(command_info, tuple) else command_info
        command_details = command_info[1] if isinstance(command_info, tuple) else None
        logger.info(f"Processing query for PDF ID '{args.pdf_id}' as command: {command_name}")

        # Pass pdf_id to handlers
        if command_name == "regular_query":
             result = process_regular_query_command(qdrant_client, args.collection_name, args.query, chat_history, args.pdf_id)
        else:
             result = process_command(qdrant_client, args.collection_name, args.query, chat_history, args.pdf_id, command_name, command_details)

    except (ConnectionError, RuntimeError) as e: logger.error(f"Error: {e}", exc_info=True); result = {"answer": f"Error: {e}", "sources": []}; print(json.dumps(result)); sys.exit(1)
    except Exception as e: logger.error(f"Unexpected error: {e}", exc_info=True); result = {"answer": f"Unexpected error: {e}", "sources": []}; print(json.dumps(result)); sys.exit(1)

    print(json.dumps(result)) # Output JSON result
    sys.exit(0)

if __name__ == "__main__":
    main()