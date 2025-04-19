# FILE: python/local_llm.py
# (Corrected: No semicolons, cleaned whitespace, Setup A logic)

import argparse
import json
import logging
import re
import sys
import requests
import os
from qdrant_client import QdrantClient, models # Import models for Filter
from sentence_transformers import SentenceTransformer
# Assumes OllamaLLM class is correctly defined in llm/ollama_llm.py
# If not, you might need to implement basic request logic here or in the class
from llm.ollama_llm import OllamaLLM
import time

# Configure logging
# Increased level to DEBUG temporarily if needed for deep tracing
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST_URL", "http://host.docker.internal:11434")
OLLAMA_API_BASE = f"{OLLAMA_HOST_URL}/api"

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'tinyllama'
DEFAULT_COLLECTION = 'documents'
CONTEXT_RETRIEVAL_LIMIT = 5
MAX_CONTEXT_CHAR_LIMIT = 4096
MAX_HISTORY_TOKENS = 500
# --- End Configuration ---

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
    llm = OllamaLLM(model_name=LLM_MODEL_NAME) # Assumes class exists
    if hasattr(llm, 'api_base'):
         llm.api_base = OLLAMA_API_BASE
         logger.info(f"Set OllamaLLM api_base to: {OLLAMA_API_BASE}")
    else:
         logger.warning("Cannot dynamically set api_base on OllamaLLM instance. Ensure class implementation uses OLLAMA_API_BASE.")
    logger.info(f"LLM instance for '{LLM_MODEL_NAME}' created.")
    # Check connection to host Ollama
    try:
        logger.info(f"Checking connection to Ollama host at {OLLAMA_HOST_URL}...")
        test_response = requests.get(f"{OLLAMA_HOST_URL}/api/tags", timeout=10) # Increased timeout slightly
        test_response.raise_for_status()
        logger.info(f"Successfully connected to Ollama at {OLLAMA_HOST_URL}")
        available_models = [m.get('name') for m in test_response.json().get('models', [])]
        if LLM_MODEL_NAME not in available_models and f"{LLM_MODEL_NAME}:latest" not in available_models:
            logger.warning(f"Model '{LLM_MODEL_NAME}' not found in host Ollama models: {available_models}. Please run 'ollama pull {LLM_MODEL_NAME}' on the host.")
    except requests.exceptions.ConnectionError as conn_err:
         logger.error(f"Could not connect to Ollama at {OLLAMA_HOST_URL}. Is Ollama application running on the host? Error: {conn_err}")
    except requests.exceptions.RequestException as req_err:
         logger.error(f"Error communicating with Ollama at {OLLAMA_HOST_URL}: {req_err}")
except Exception as e:
     logger.critical(f"CRITICAL: LLM init/check failed: {e}", exc_info=True)
     sys.exit("LLM failed to initialize")

def connect_qdrant(host, port, retries=5, delay=3):
    """Connects to Qdrant with retries."""
    for attempt in range(retries):
        try:
            logger.info(f"Attempting to connect to Qdrant at {host}:{port} (Attempt {attempt + 1}/{retries})...")
            client = QdrantClient(host=host, port=port, timeout=20)
            client.get_collections() # Test connection
            logger.info("Qdrant client connected successfully.")
            return client
        except Exception as e:
            logger.warning(f"Qdrant connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max Qdrant connection retries reached.")
                raise ConnectionError(f"Could not connect to Qdrant at {host}:{port} after {retries} attempts") from e
# --- End Init ---


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
    context_str = ""
    sources = []
    if not results: return context_str, sources
    logger.info("Formatting context for LLM...")
    for i, hit in enumerate(results):
        try:
            payload = hit.payload or {}
            text = payload.get("text", "")
            page = payload.get("page", "N/A")
            doc_name = payload.get("source", "Unknown")
            score = hit.score or 0.0
            if text:
                 context_str += f"Source [{i+1}] (Page: {page}, Document: {doc_name}, Score: {score:.3f}):\n{text}\n\n"
                 sources.append({"id": i + 1, "page": page, "document": doc_name, "score": score})
        except Exception as e: logger.warning(f"Failed format hit {i}: {e}")
    return context_str.strip(), sources


def estimate_tokens(text):
    """Basic token estimation."""
    return len(text.split())


def generate_rag_response(query, context_str, chat_history=None, system_instruction=None):
    """Generates a response using the LLM with context, history, and citation attempts."""
    if not llm: raise RuntimeError("LLM is not initialized.")
    if not system_instruction:
        system_instruction = (
            "You are an AI assistant answering questions based ONLY on the provided document context. "
            "Use ONLY the information presented in the 'Source [n]' sections. "
            "Do not use any external knowledge or make assumptions. "
            "If the answer cannot be found in the context, state clearly: 'The provided context does not contain the answer to this question.' "
            "When using information from a source, **you MUST cite the source number** (e.g., [1], [2]) at the end of the sentence(s) referencing that source. "
            "Be concise."
        )
    history_str = ""
    if chat_history:
        token_count = 0
        for turn in reversed(chat_history):
            turn_text = f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
            turn_tokens = estimate_tokens(turn_text)
            if token_count + turn_tokens > MAX_HISTORY_TOKENS: break
            history_str = turn_text + history_str
            token_count += turn_tokens
        if history_str: history_str = f"Previous Conversation History:\n---\n{history_str.strip()}\n---\n\n"; logger.info(f"Including history ({token_count} tokens).")

    # Handle context truncation
    context_len = len(context_str)
    if context_str and context_len > MAX_CONTEXT_CHAR_LIMIT:
        logger.warning(f"Context length ({context_len}) exceeds limit ({MAX_CONTEXT_CHAR_LIMIT}), truncating.")
        context_str = context_str[:MAX_CONTEXT_CHAR_LIMIT] + "..."

    # Construct prompt
    if not context_str:
         prompt_for_llm = f"{history_str}Instruction: {system_instruction}\n\nUser Question: {query}\n\nContext: [No relevant context provided]\n\nAssistant Answer:"
         logger.warning("No context provided.")
    else:
        prompt_for_llm = ( f"{history_str}Instruction: {system_instruction}\n\nContext:\n---\n{context_str}\n---\n\nUser Question: {query}\n\nAssistant Answer (Cite sources like [1]):" )

    # Use the configured OLLAMA_API_BASE when generating response
    # Assuming llm object or its generate_response method uses self.api_base internally
    current_llm_target = getattr(llm, 'api_base', OLLAMA_API_BASE) # Get the actual target
    logger.info(f"Sending request to LLM '{LLM_MODEL_NAME}' at {current_llm_target}...")
    try:
        response = llm.generate_response(prompt_for_llm)
        logger.info("Received response from LLM.")
        response = response.split("Assistant Answer")[-1].strip(':').strip()
        return response
    except Exception as e: logger.error(f"LLM generation failed: {e}", exc_info=True); return "LLM generation error."
# --- End Core RAG Functions ---


# --- Command Processing ---
def detect_command_type(query):
    """Detect command type from query."""
    query_lower = query.lower().strip()
    if query_lower.startswith("extract keywords"): return "keywords"
    if any(term in query_lower for term in ["explain each topic", "explain all topics"]): return "explain_topics"
    if "list topics" in query_lower: return "topics"
    if query_lower.startswith("summarize"): return "summary"
    if query_lower.startswith("define "):
        match = re.match(r"define\s+(.+)", query, re.IGNORECASE)
        term = match.group(1).strip().rstrip('?.!') if match else None
        if term: term = re.sub(r'[^\w\s-]', '', term); return "definition", term
    if "generate questions" in query_lower: return "questions"
    return "regular_query"

def process_command(client, collection_name, query, chat_history, pdf_id_filter, command_type, command_details=None):
    """Handle specific commands."""
    logger.info(f"Processing command: {command_type} for PDF ID: {pdf_id_filter}")
    retrieval_query = query; system_instruction = None; limit = 5; query_for_llm = query
    # Define command specifics...
    if command_type == "summary": retrieval_query = "Overall summary"; limit = 15; system_instruction = "Summarize comprehensively..."; query_for_llm = "Summarize."
    elif command_type == "definition": term=command_details; limit = 7; 
    elif not term: return {"answer": "Specify term.", "sources":[]}; retrieval_query = f"Define '{term}'"; system_instruction = f"Define '{term}' based ONLY on context..."; query_for_llm = f"Define '{term}'."
    elif command_type == "questions": limit = 10; system_instruction = "Generate 3-5 questions based ONLY on context..."; query_for_llm = "Generate questions."
    elif command_type == "topics": limit = 15; system_instruction = "List main topics ONLY from context..."; query_for_llm = "List topics."
    elif command_type == "explain_topics": limit = 15; system_instruction = "Identify and explain main topics ONLY from context..."; query_for_llm = "Explain topics."
    elif command_type == "keywords": limit = 10; system_instruction = "Extract keywords and named entities ONLY from context..."; query_for_llm = "Extract keywords."
    else: return process_regular_query_command(client, collection_name, query, chat_history, pdf_id_filter)

    retrieved_context = retrieve_context(client, collection_name, retrieval_query, pdf_id_filter, limit=limit)
    context_str, sources = format_context_for_llm(retrieved_context)
    if not context_str: answer = f"Could not retrieve context for command '{command_type}'."; return {"answer": answer, "sources": []}
    answer = generate_rag_response(query_for_llm, context_str, chat_history, system_instruction)
    return {"answer": answer, "sources": sources}

def process_regular_query_command(client, collection_name, query, chat_history, pdf_id_filter):
    """Handle regular queries."""
    logger.info(f"Processing regular query for PDF ID {pdf_id_filter}: {query[:50]}...")
    retrieved_context = retrieve_context(client, collection_name, query, pdf_id_filter, limit=CONTEXT_RETRIEVAL_LIMIT)
    context_str, sources = format_context_for_llm(retrieved_context)
    answer = generate_rag_response(query, context_str, chat_history, system_prompt=None) # Use default prompt
    return {"answer": answer, "sources": sources}
# --- End Command Processing ---


# --- Main Execution ---
def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process query for RAG (Local Setup)')
    parser.add_argument('query', type=str, help='Query')
    parser.add_argument('--collection_name', type=str, default=DEFAULT_COLLECTION, help='Qdrant collection')
    parser.add_argument('--pdf_id', required=True, help='PDF ID to filter by')
    parser.add_argument('--history', type=str, default='[]', help='Chat history JSON')
    args = parser.parse_args()

    if not embedding_model or not llm:
         logger.critical("Models not loaded.")
         print(json.dumps({"answer": "Error: AI models failed.", "sources": []}))
         sys.exit(1)

    try:
        chat_history = json.loads(args.history)
        if not isinstance(chat_history, list): raise ValueError("History must be a list.")
        # Further validation could be added here if needed
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid chat history format: {e}")
        chat_history = [] # Default to empty on error

    result = {}
    try:
        qdrant_client = connect_qdrant(QDRANT_HOST, QDRANT_PORT)
        command_info = detect_command_type(args.query)
        command_name = command_info[0] if isinstance(command_info, tuple) else command_info
        command_details = command_info[1] if isinstance(command_info, tuple) else None
        logger.info(f"Processing PDF '{args.pdf_id}' command: {command_name}")

        # Pass pdf_id to handlers
        if command_name == "regular_query":
             result = process_regular_query_command(qdrant_client, args.collection_name, args.query, chat_history, args.pdf_id)
        else:
             result = process_command(qdrant_client, args.collection_name, args.query, chat_history, args.pdf_id, command_name, command_details)

    except (ConnectionError, RuntimeError) as e:
         logger.error(f"Execution Error: {e}", exc_info=True)
         result = {"answer": f"Error: {e}", "sources": []}
         print(json.dumps(result))
         sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        result = {"answer": f"Unexpected error: {e}", "sources": []}
        print(json.dumps(result))
        sys.exit(1)

    # Output result for backend
    print(json.dumps(result))
    sys.exit(0)

if __name__ == "__main__":
    main()