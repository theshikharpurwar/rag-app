# FILE: python/local_llm.py
# (Adds pdf_id argument and filtering to Qdrant search)

import argparse
import json
import logging
import re
import sys
import requests
from qdrant_client import QdrantClient, models # Import models for Filter
from sentence_transformers import SentenceTransformer
from llm.ollama_llm import OllamaLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'tinyllama'
DEFAULT_COLLECTION = 'documents'
CONTEXT_RETRIEVAL_LIMIT = 5
MAX_CONTEXT_CHAR_LIMIT = 4096 # Keep updated limit
MAX_HISTORY_TOKENS = 500
# --- End Configuration ---

# --- Client/Model Initialization (Keep as before) ---
embedding_model = None
llm = None
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
except Exception as e: logger.critical(f"CRITICAL: Embedding model load failed: {e}", exc_info=True); sys.exit(1)
try:
    llm = OllamaLLM(model_name=LLM_MODEL_NAME)
    logger.info(f"LLM instance for '{LLM_MODEL_NAME}' created.")
except Exception as e: logger.critical(f"CRITICAL: LLM init failed: {e}", exc_info=True); sys.exit(1)

def get_qdrant_client():
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=20)
        client.get_collections(); logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return client
    except Exception as e: logger.error(f"Failed to connect to Qdrant: {str(e)}"); raise ConnectionError(f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}") from e
# --- End Client/Model Initialization ---

# --- Core RAG Functions ---

# *** MODIFIED retrieve_context to accept and use pdf_id for filtering ***
def retrieve_context(client, collection_name, query, pdf_id_filter, limit=CONTEXT_RETRIEVAL_LIMIT):
    """Retrieve context from Qdrant for a specific PDF ID based on query."""
    if not embedding_model: raise RuntimeError("Embedding model is not loaded.")
    if not pdf_id_filter:
         logger.error("pdf_id_filter is required for retrieving context.")
         return [] # Cannot retrieve without knowing which document

    try:
        query_embedding = embedding_model.encode(query).tolist()

        # *** CREATE QDRANT FILTER ***
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="pdf_id", # The field stored in the payload
                    match=models.MatchValue(value=pdf_id_filter)
                )
            ]
        )
        logger.info(f"Searching collection '{collection_name}' (limit={limit}) with filter for pdf_id='{pdf_id_filter}'...")

        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter, # *** APPLY FILTER ***
            limit=limit,
            with_payload=True
        )
        logger.info(f"Retrieved {len(search_results)} results from Qdrant for pdf_id '{pdf_id_filter}'.")

        # Filter results to ensure they have valid text payload (redundant if only text stored, good practice)
        valid_results = [
            hit for hit in search_results
            if hit.payload and isinstance(hit.payload.get("text"), str) and hit.payload.get("text").strip()
        ]
        if len(valid_results) < len(search_results):
             logger.warning(f"Filtered out {len(search_results) - len(valid_results)} results lacking valid text payload (unexpected with filter).")
        return valid_results
    except Exception as e:
        logger.error(f"Error retrieving context from Qdrant: {e}", exc_info=True)
        return []

# format_context_for_llm remains the same
def format_context_for_llm(results):
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

# estimate_tokens remains the same
def estimate_tokens(text): return len(text.split())

# generate_rag_response remains the same (takes context string, doesn't need pdf_id directly)
def generate_rag_response(query, context_str, chat_history=None, system_instruction=None):
    if not llm: raise RuntimeError("LLM is not initialized.")
    if not system_instruction:
        system_instruction = ("...") # Default prompt from previous step
    history_str = ""
    if chat_history: # Format history (same as before)
        token_count = 0
        for turn in reversed(chat_history):
            turn_text = f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
            turn_tokens = estimate_tokens(turn_text)
            if token_count + turn_tokens > MAX_HISTORY_TOKENS: break
            history_str = turn_text + history_str; token_count += turn_tokens
        if history_str: history_str = f"Previous Conversation:\n---\n{history_str.strip()}\n---\n\n"
    if context_str and len(context_str) > MAX_CONTEXT_CHAR_LIMIT: # Truncate
        logger.warning(f"Context length ({len(context_str)}) exceeds limit ({MAX_CONTEXT_CHAR_LIMIT}), truncating.")
        context_str = context_str[:MAX_CONTEXT_CHAR_LIMIT] + "..."
    if not context_str:
         prompt_for_llm = f"{history_str}Instruction: {system_instruction}\n\nUser Question: {query}\n\nContext: [No relevant context provided]\n\nAssistant Answer:"
    else:
        prompt_for_llm = (f"{history_str}Instruction: {system_instruction}\n\nContext:\n---\n{context_str}\n---\n\nUser Question: {query}\n\nAssistant Answer (Cite sources like [1]):")
    logger.info(f"Sending request to LLM '{LLM_MODEL_NAME}'...")
    try:
        response = llm.generate_response(prompt_for_llm)
        logger.info("Received response from LLM.")
        response = response.split("Assistant Answer")[-1].strip(':').strip()
        return response
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        return "Sorry, an error occurred during LLM generation."

# --- Command Processing (modified to pass pdf_id_filter) ---

def detect_command_type(query):
    # Same detection logic as before
    query_lower = query.lower().strip()
    if query_lower.startswith("extract keywords") or query_lower.startswith("list keywords"): return "keywords"
    if any(term in query_lower for term in ["explain each topic", "explain all topics"]): return "explain_topics"
    if "list topics" in query_lower or "main topics" in query_lower or "key topics" in query_lower: return "topics"
    if query_lower.startswith("summarize"): return "summary"
    if query_lower.startswith("define "):
        match = re.match(r"define\s+(.+)", query, re.IGNORECASE)
        if match:
            term = match.group(1).strip().rstrip('?.!'); term = re.sub(r'[^\w\s-]', '', term);
            if term: return "definition", term
    if any(term in query_lower for term in ["create questions", "sample questions", "generate questions"]): return "questions"
    return "regular_query"

# *** MODIFIED process_command to accept and pass pdf_id_filter ***
def process_command(client, collection_name, query, chat_history, pdf_id_filter, command_type, command_details=None):
    logger.info(f"Processing command: {command_type} for PDF ID: {pdf_id_filter}")
    retrieval_query = query; system_instruction = None; limit = 5; query_for_llm = query
    # Define command specifics (same prompts as before)
    if command_type == "summary":
        retrieval_query = "Overall document content summary key points"; limit = 15
        system_instruction = "Summarize the text comprehensively based ONLY on the provided context..."; query_for_llm = "Summarize."
    elif command_type == "definition":
        term = command_details; limit = 7
        if not term: return {"answer": "Please specify term.", "sources": []}
        retrieval_query = f"Definition of '{term}'"; system_instruction = f"Based ONLY on context, define '{term}'..."; query_for_llm = f"Define '{term}'."
    elif command_type == "questions":
        retrieval_query = "Key points for question generation"; limit = 10
        system_instruction = "Based ONLY on context, generate 3-5 insightful questions..."; query_for_llm = "Generate questions."
    elif command_type == "topics":
         retrieval_query = "Main themes and topics"; limit = 15
         system_instruction = "Identify and list main topics ONLY from context..."; query_for_llm = "List main topics."
    elif command_type == "explain_topics":
         retrieval_query = "Detailed explanation of main topics"; limit = 15
         system_instruction = "Identify main topics in context and explain each briefly, based ONLY on context..."; query_for_llm = "Explain main topics."
    elif command_type == "keywords":
        retrieval_query = "Keywords and key entities"; limit = 10
        system_instruction = "Extract main keywords and named entities ONLY from the provided context..."; query_for_llm = "Extract keywords."
    else:
        logger.error(f"Unhandled command type '{command_type}'.")
        return process_regular_query_command(client, collection_name, query, chat_history, pdf_id_filter) # Fallback

    # Execute retrieval **with filter** and generation
    retrieved_context = retrieve_context(client, collection_name, retrieval_query, pdf_id_filter, limit=limit)
    context_str, sources = format_context_for_llm(retrieved_context)

    if not context_str:
        answer = f"Could not retrieve relevant context for command '{command_type}' from the specified document."
        if command_type == "definition": answer = f"Could not find context for '{command_details}' in the specified document."
        return {"answer": answer, "sources": []}

    logger.info(f"Generating LLM response for command '{command_type}'...")
    answer = generate_rag_response(query_for_llm, context_str, chat_history, system_instruction)
    return {"answer": answer, "sources": sources}


# *** MODIFIED process_regular_query_command to accept and pass pdf_id_filter ***
def process_regular_query_command(client, collection_name, query, chat_history, pdf_id_filter):
    """Handles a regular query using RAG, filtered by pdf_id."""
    logger.info(f"Processing regular query for PDF ID {pdf_id_filter}: {query[:50]}...")
    retrieved_context = retrieve_context(client, collection_name, query, pdf_id_filter, limit=CONTEXT_RETRIEVAL_LIMIT)
    context_str, sources = format_context_for_llm(retrieved_context)

    # Use the default RAG system prompt (includes citation request)
    answer = generate_rag_response(query, context_str, chat_history, system_prompt=None)
    return {"answer": answer, "sources": sources}

# --- Main Execution (modified to handle pdf_id) ---
def main():
    parser = argparse.ArgumentParser(description='Process a query for RAG using local LLM')
    parser.add_argument('query', type=str, help='The query to process')
    parser.add_argument('--collection_name', type=str, default=DEFAULT_COLLECTION, help='Qdrant collection name')
    # *** ADD pdf_id argument ***
    parser.add_argument('--pdf_id', required=True, help='MongoDB ID of the PDF to filter by')
    parser.add_argument('--history', type=str, default='[]', help='Chat history as a JSON string')
    args = parser.parse_args()

    if not embedding_model or not llm:
         logger.critical("Essential AI models did not load. Cannot process query.")
         result = {"answer": "Error: AI models failed to load.", "sources": []}
         print(json.dumps(result)); sys.exit(1)

    try: # Parse history
        chat_history = json.loads(args.history)
        if not isinstance(chat_history, list): raise ValueError("History must be a list.")
        # Add more validation if needed
    except Exception as e: logger.error(f"Invalid chat history: {e}"); chat_history = []

    result = {}
    try:
        qdrant_client = get_qdrant_client()
        command_info = detect_command_type(args.query)
        command_name = command_info[0] if isinstance(command_info, tuple) else command_info
        command_details = command_info[1] if isinstance(command_info, tuple) else None
        logger.info(f"Processing query for PDF '{args.pdf_id}' as command: {command_name}")

        if command_name == "regular_query":
             # *** Pass pdf_id to handler ***
             result = process_regular_query_command(qdrant_client, args.collection_name, args.query, chat_history, args.pdf_id)
        else:
             # *** Pass pdf_id to handler ***
             result = process_command(qdrant_client, args.collection_name, args.query, chat_history, args.pdf_id, command_name, command_details)

    except (ConnectionError, RuntimeError) as e:
         logger.error(f"Error: {e}", exc_info=True)
         result = {"answer": f"Error: {e}", "sources": []}
         print(json.dumps(result)); sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        result = {"answer": f"Sorry, an unexpected error occurred: {e}", "sources": []}
        print(json.dumps(result)); sys.exit(1)

    print(json.dumps(result)) # Output JSON result
    sys.exit(0)

if __name__ == "__main__":
    main()