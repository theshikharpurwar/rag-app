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
# Using environment variables for all connections
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
# Ollama runs on the HOST machine
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434")
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
    
    # Increase default retrieval limit for better coverage
    actual_limit = limit * 2  # Double the requested limit to get more potential matches
    
    try:
        # Process the query to improve retrieval quality
        query = query.strip()
        if len(query) < 10 and not query.endswith('?'):
            # Very short queries often need expansion
            logger.info(f"Expanding short query: '{query}'")
            query = f"Information about {query} in detail"
        
        # For queries that are clearly looking for "what is X" type of information
        if re.match(r'^(what|who|how|when|where|why|explain|describe|define)\s+', query.lower()):
            # Keep the query as is, it's already well-formed for semantic search
            pass
        elif not query.endswith('?') and len(query.split()) < 5:
            # If it's not a question and it's short, make it more detailed
            query = f"Find detailed information about {query}"
            
        logger.info(f"Processed query for retrieval: '{query}'")
        
        # Generate embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Create filter
        qdrant_filter = models.Filter(must=[models.FieldCondition(key="pdf_id", match=models.MatchValue(value=pdf_id_filter))])
        logger.info(f"Searching collection '{collection_name}' (limit={actual_limit}) with filter...")
        
        # Search with the expanded limit
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=actual_limit,
            with_payload=True,
            score_threshold=0.2  # Only return results with at least this similarity score
        )
        
        logger.info(f"Retrieved {len(search_results)} results from Qdrant for pdf_id '{pdf_id_filter}'.")
        
        # Advanced post-processing to select the most diverse and relevant context
        if len(search_results) > limit:
            # Filter out results with poor quality text
            valid_results = [
                hit for hit in search_results 
                if hit.payload and isinstance(hit.payload.get("text"), str) 
                and len(hit.payload.get("text", "").strip()) > 50  # Minimum useful text length
                and hit.score > 0.3  # Reasonable similarity threshold
            ]
            
            # If we still have more than the limit, select a diverse set
            if len(valid_results) > limit:
                # Sort by score first
                sorted_results = sorted(valid_results, key=lambda x: x.score, reverse=True)
                
                # Always include the top half results by score
                top_half = sorted_results[:limit//2]
                
                # For the remaining half, try to select results with different page numbers
                # to increase diversity of information
                remaining = sorted_results[limit//2:]
                selected_pages = set(r.payload.get("page") for r in top_half if r.payload.get("page"))
                diverse_selections = []
                
                for result in remaining:
                    page = result.payload.get("page")
                    if page and page not in selected_pages and len(diverse_selections) < (limit - len(top_half)):
                        diverse_selections.append(result)
                        selected_pages.add(page)
                
                # If we couldn't find enough diverse pages, just add the best remaining by score
                if len(diverse_selections) < (limit - len(top_half)):
                    needed = limit - len(top_half) - len(diverse_selections)
                    for result in remaining:
                        if result not in diverse_selections and needed > 0:
                            diverse_selections.append(result)
                            needed -= 1
                        if needed == 0:
                            break
                
                # Combine top results with diverse selections
                final_results = top_half + diverse_selections
                logger.info(f"Selected {len(final_results)} diverse results from {len(valid_results)} valid results")
                return final_results
            
            return valid_results[:limit]  # Return top valid results up to limit
        
        # If we have fewer than limit results, just return valid ones
        valid_results = [
            hit for hit in search_results 
            if hit.payload and isinstance(hit.payload.get("text"), str) 
            and hit.payload.get("text").strip()
        ]
        
        if len(valid_results) < len(search_results): 
            logger.warning(f"Filtered out {len(search_results) - len(valid_results)} results lacking valid text payload.")
        
        return valid_results
        
    except Exception as e: 
        logger.error(f"Qdrant retrieval error: {e}", exc_info=True)
        return []

def format_context_for_llm(results):
    """Formats retrieved context for the LLM prompt and extracts sources."""
    context_str = ""; sources = []
    if not results: return context_str, sources
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
                # Add more structured context with clear section headers
                context_str += f"[SOURCE {i+1}]\nDocument: {doc_name}\nPage: {page}\nRelevance Score: {score:.3f}\nContent:\n{text}\n\n"
                sources.append({
                    "id": i + 1, 
                    "page": page, 
                    "document": doc_name, 
                    "score": score
                })
        except Exception as e: 
            logger.warning(f"Failed to format hit {i}: {e}")
    
    return context_str.strip(), sources

def estimate_tokens(text): return len(text.split()) # Basic token estimate

def generate_rag_response(query, context_str, chat_history=None, system_instruction=None):
    """Generates a response using the LLM with context, history, and citation attempts."""
    if not llm: raise RuntimeError("LLM is not initialized.")
    
    if not system_instruction:
        system_instruction = (
            "You are a precise, accurate AI assistant answering questions based ONLY on the provided document sources. "
            "Follow these guidelines strictly:\n"
            "1. Use ONLY information from the provided [SOURCE X] sections.\n"
            "2. DO NOT use prior knowledge or make assumptions not supported by the sources.\n"
            "3. For each statement in your answer, cite the relevant source number in brackets like [1] or [2,3] at the end of sentences.\n"
            "4. If the provided sources don't contain the necessary information, state clearly: 'The provided sources don't contain information about [specific aspect]'.\n"
            "5. Be as specific, detailed, and comprehensive as possible while staying grounded in the sources.\n"
            "6. Resolve conflicts between sources by noting the conflicting information and citing both.\n"
            "7. When citing multiple sources, separate numbers with commas: [1,3,4] instead of [1][3][4].\n"
            "8. If a statement combines information from multiple sources, cite all relevant sources.\n"
            "9. Present the information as a cohesive, well-structured answer, rather than a list of facts.\n"
            "Your goal is to be helpful, accurate, and honest about what information is and isn't available in the provided sources."
        )
    
    # Format chat history
    history_str = ""
    if chat_history:
        token_count = 0
        for turn in reversed(chat_history):
            turn_text = f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
            turn_tokens = estimate_tokens(turn_text)
            if token_count + turn_tokens > MAX_HISTORY_TOKENS: break
            history_str = turn_text + history_str
            token_count += turn_tokens
        if history_str: history_str = f"Previous Conversation History:\n---\n{history_str.strip()}\n---\n\n"
    
    # Handle missing or excessive context
    if not context_str:
        logger.warning("No context provided to LLM.")
        prompt_for_llm = (
            f"{history_str}Instruction: {system_instruction}\n\n"
            f"User Question: {query}\n\n"
            f"Context: [No relevant sources found]\n\n"
            f"Assistant Answer: I don't have enough information in the provided sources to answer this question confidently."
        )
    else:
        # Handle excessive context length
        if len(context_str) > MAX_CONTEXT_CHAR_LIMIT:
            logger.warning(f"Context length ({len(context_str)}) exceeds limit ({MAX_CONTEXT_CHAR_LIMIT}), truncating.")
            # More intelligent truncation - keep complete source sections until we hit the limit
            sections = context_str.split("[SOURCE ")
            header = sections[0]
            sources = ["[SOURCE " + s for s in sections[1:]]
            
            truncated_context = header
            current_length = len(header)
            
            for section in sources:
                if current_length + len(section) + 3 > MAX_CONTEXT_CHAR_LIMIT:
                    # We can't fit this section, stop here
                    break
                truncated_context += section + "\n"
                current_length += len(section) + 1
            
            context_str = truncated_context.strip()
            logger.info(f"Intelligently truncated context to {len(context_str)} characters")

        # Build the final prompt with comprehensive instructions
        prompt_for_llm = (
            f"{history_str}Instruction: {system_instruction}\n\n"
            f"Here are the relevant sources to answer the question:\n"
            f"---\n{context_str}\n---\n\n"
            f"User Question: {query}\n\n"
            f"Assistant Answer (Be specific and cite sources like [1] or [2,3]):"
        )
    
    logger.info(f"Sending request to LLM '{LLM_MODEL_NAME}' at {OLLAMA_API_BASE}...")
    try:
        response = llm.generate_response(prompt_for_llm)
        logger.info("Received response from LLM.")
        
        # Clean up the response to remove any prefixes like "Assistant Answer:"
        response = re.sub(r'^Assistant Answer:?\s*', '', response.split("Assistant Answer")[-1].strip())
        return response.strip()
    except Exception as e: 
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        return "LLM generation error."
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
    logger.info(f"Processing command: {command_type} for PDF ID: {pdf_id_filter}")
    retrieval_query = query
    system_instruction = None
    limit = 5
    query_for_llm = query

    # Define specialized prompts and behavior for each command type
    if command_type == "summary":
        # For summarization, we want to gather more comprehensive context
        retrieval_query = "Document overview main points summary key topics"
        limit = 15
        system_instruction = (
            "You are an expert summarizer. Create a comprehensive summary of the document based ONLY on the provided sources. "
            "Focus on the main ideas, key points, and important details. "
            "Structure your summary with clear sections covering the major topics in the document. "
            "Include all significant information while omitting minor details. "
            "Cite sources for key information as [SOURCE NUMBER]. "
            "Do not include any information not present in the sources. "
            "If the sources appear incomplete or only cover part of the document, note this limitation."
        )
        query_for_llm = "Please provide a comprehensive summary of this document."
    
    elif command_type == "definition":
        term = command_details
        limit = 7
        retrieval_query = f"Define '{term}' meaning explanation concept"
        system_instruction = (
            f"You are a precise definitions expert. Provide a clear, accurate definition of '{term}' based ONLY on the provided sources. "
            f"Include the following in your response if available in the sources:\n"
            f"1. The formal definition of '{term}'\n"
            f"2. Key characteristics or components\n"
            f"3. Important context or related concepts\n"
            f"4. Examples if mentioned\n"
            f"Always cite your sources using [SOURCE NUMBER]. "
            f"If the sources don't contain a definition of '{term}', clearly state that the term is not defined in the provided sources."
        )
        query_for_llm = f"Define the term '{term}' based on the sources."
    
    elif command_type == "questions":
        limit = 10
        retrieval_query = "Generate detailed questions main topics important concepts document"
        system_instruction = (
            "You are an expert at generating insightful questions. Based ONLY on the provided sources, generate a set of thought-provoking questions that: "
            "1. Cover the main topics and concepts in the document\n"
            "2. Range from factual to analytical and conceptual\n"
            "3. Follow a logical progression from basic to advanced understanding\n"
            "4. Would help someone test their comprehension of the material\n"
            "For each question, cite the relevant source(s) as [SOURCE NUMBER]. "
            "Generate 5-10 questions depending on the breadth of the material in the sources."
        )
        query_for_llm = "Generate insightful questions about the content of this document."
    
    elif command_type == "topics":
        limit = 15
        retrieval_query = "List main topics subjects key concepts sections document"
        system_instruction = (
            "You are an expert at identifying and organizing topics. Based ONLY on the provided sources, create a structured list of the main topics covered in the document. "
            "For each topic:\n"
            "1. Provide a clear, concise title\n"
            "2. Include a brief (1-2 sentence) description\n"
            "3. Cite the relevant source(s) as [SOURCE NUMBER]\n"
            "Organize related topics together and present them in a logical sequence. "
            "Include only topics that are explicitly mentioned in the sources."
        )
        query_for_llm = "List and briefly describe the main topics covered in this document."
    
    elif command_type == "explain_topics":
        limit = 15
        retrieval_query = "Explain topics concepts main ideas document detailed"
        system_instruction = (
            "You are an expert at explaining complex topics. Based ONLY on the provided sources, provide detailed explanations of the main topics in the document. "
            "For each topic:\n"
            "1. Clearly identify the topic with a descriptive heading\n"
            "2. Provide a thorough explanation including key concepts, principles, and supporting details\n"
            "3. Explain relationships to other topics where relevant\n"
            "4. Cite all relevant sources using [SOURCE NUMBER]\n"
            "Organize your response in a logical flow that helps the reader build understanding progressively. "
            "Include only information explicitly stated in the sources."
        )
        query_for_llm = "Explain the main topics covered in this document in detail."
    
    elif command_type == "keywords":
        limit = 10
        retrieval_query = "Extract important keywords terms concepts document"
        system_instruction = (
            "You are an expert at identifying key terminology. Based ONLY on the provided sources, extract and explain the most important keywords or terms from the document. "
            "For each keyword/term:\n"
            "1. List the term in bold\n"
            "2. Provide a concise definition or explanation based on how it's used in the document\n"
            "3. Include context about why this term is significant\n"
            "4. Cite the relevant source(s) as [SOURCE NUMBER]\n"
            "Focus on specialized terminology, core concepts, and frequently referenced ideas. "
            "Organize terms thematically or alphabetically for easy reference."
        )
        query_for_llm = "Extract and explain the important keywords and terms from this document."
    
    else:
        return process_regular_query_command(client, collection_name, query, chat_history, pdf_id_filter)

    # Retrieve enhanced context with specialized query and higher limit
    retrieved_context = retrieve_context(client, collection_name, retrieval_query, pdf_id_filter, limit=limit)
    context_str, sources = format_context_for_llm(retrieved_context)
    
    if not context_str:
        answer = f"Could not retrieve context for command '{command_type}' from PDF '{pdf_id_filter}'."
        return {"answer": answer, "sources": []}
    
    # Generate response with specialized system instruction
    answer = generate_rag_response(query_for_llm, context_str, chat_history, system_instruction)
    return {"answer": answer, "sources": sources}

def process_regular_query_command(client, collection_name, query, chat_history, pdf_id_filter):
    """Process a regular query with enhanced context retrieval and response generation."""
    logger.info(f"Processing regular query for PDF ID {pdf_id_filter}: {query[:50]}...")
    
    # For regular queries, use a higher retrieval limit to get better coverage
    retrieval_limit = CONTEXT_RETRIEVAL_LIMIT * 2
    
    # Create a more focused system instruction for regular queries
    system_instruction = (
        "You are a precise, helpful AI assistant answering questions based ONLY on the provided document sources. "
        "Follow these guidelines strictly:\n"
        "1. Answer ONLY from the information in the provided sources. Do NOT use prior knowledge.\n"
        "2. Be thorough and detailed in your answer while staying directly relevant to the question.\n"
        "3. Cite specific sources for each part of your answer using [SOURCE NUMBER] notation.\n"
        "4. If the sources don't contain the information needed to answer fully, clearly state what's missing.\n"
        "5. If the question is unclear, interpret it reasonably based on the sources and explain your interpretation.\n"
        "6. Structure your answer logically with paragraphs for different aspects of the answer.\n"
        "7. If the sources contain conflicting information, note the conflicts and cite both sources.\n"
        "Your goal is to provide the most accurate, helpful answer possible using only the information available in the sources."
    )
    
    # Retrieve enhanced context with higher limit
    retrieved_context = retrieve_context(client, collection_name, query, pdf_id_filter, limit=retrieval_limit)
    context_str, sources = format_context_for_llm(retrieved_context)
    
    if not context_str:
        answer = f"I couldn't find relevant information to answer your question in the document (PDF ID: {pdf_id_filter}). Please try rephrasing your question or ask about a different topic covered in the document."
        return {"answer": answer, "sources": []}
    
    # Generate response with enhanced instruction
    answer = generate_rag_response(query, context_str, chat_history, system_instruction)
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