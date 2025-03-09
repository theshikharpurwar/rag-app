# D:\rag-app\python\local_llm.py

import sys
import json
import argparse
import logging
from qdrant_client import QdrantClient
import requests
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleEmbedder:
    """Simple embedder that uses Sentence Transformers"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logger.info(f"Initializing embedder with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def get_embedding(self, content, content_type="text"):
        try:
            if content_type.lower() == "text":
                if not content or content.strip() == "":
                    logger.warning("Empty text content provided, returning zero vector")
                    return [0.0] * 384  # Default size for all-MiniLM-L6-v2

                logger.info(f"Generating embedding for text: {content[:50]}...")
                embedding = self.model.encode(content)
                return embedding.tolist()
            else:
                logger.error(f"Unsupported content type: {content_type}")
                return [0.0] * 384
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384

def get_question_contexts(query, collection_name='documents', limit=5):
    """
    Get relevant contexts for a question from Qdrant

    Args:
        query (str): The question to search for
        collection_name (str): Name of the Qdrant collection
        limit (int): Maximum number of contexts to retrieve

    Returns:
        list: List of retrieved contexts
    """
    try:
        logger.info(f"Searching for query: {query}")
        logger.info(f"Using collection: {collection_name}")

        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)

        # Create an embedder to generate query vector
        embedder = SimpleEmbedder()

        # Generate embedding from the query text
        query_vector = embedder.get_embedding(query, "text")

        # Search for similar contexts using the query vector
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )

        # Extract relevant information from search results
        contexts = []
        for result in search_result:
            context = {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "source": result.payload.get("source", ""),
                "page": result.payload.get("page", 0),
                "type": result.payload.get("type", "text"),
                "image_path": result.payload.get("image_path", "")
            }
            contexts.append(context)

        logger.info(f"Found {len(contexts)} relevant contexts")
        return contexts

    except Exception as e:
        logger.error(f"Error getting contexts: {str(e)}")
        return []

def generate_summary(collection_name='documents'):
    """
    Generate a summary of all documents in the collection

    Args:
        collection_name (str): Name of the Qdrant collection

    Returns:
        str: The generated summary
    """
    try:
        logger.info(f"Generating summary for collection: {collection_name}")

        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)

        # Get all points in the collection (limited to 100 for practicality)
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False
        )

        points = scroll_result[0]

        # Group points by source
        sources = {}
        for point in points:
            source = point.payload.get("source", "unknown")
            if source not in sources:
                sources[source] = []

            # Only include text payloads
            if point.payload.get("type") == "text" and point.payload.get("text"):
                sources[source].append({
                    "page": point.payload.get("page", 0),
                    "text": point.payload.get("text", "")
                })

        # If no sources found, return a simple message
        if not sources:
            return "No documents found in the collection."

        # Create a summary of each source
        summary = "# Document Summary\n\n"

        for source, pages in sources.items():
            # Sort pages by page number
            pages.sort(key=lambda x: x["page"])

            summary += f"## {source}\n\n"
            summary += f"Contains {len(pages)} pages of content.\n\n"

            # Add a brief overview of each page (first few sentences)
            for page in pages:
                page_num = page["page"]
                page_text = page["text"]

                # Get first 200 characters for the overview
                page_overview = page_text[:200] + "..." if len(page_text) > 200 else page_text

                summary += f"**Page {page_num}**: {page_overview}\n\n"

        logger.info(f"Generated summary with {len(sources)} sources")
        return summary

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def generate_answer_with_ollama(query, contexts, model_name="phi"):
    """
    Generate an answer using Ollama

    Args:
        query (str): The query to answer
        contexts (list): List of contexts to use for answering
        model_name (str): Name of the Ollama model to use

    Returns:
        str: The generated answer
    """
    try:
        logger.info(f"Generating answer with Ollama model: {model_name}")

        if not contexts:
            return "I couldn't find any relevant information to answer your question."

        # Prepare the context text
        context_text = "\n\n".join([
            f"Context {i+1}:\n{ctx.get('text', '')}"
            for i, ctx in enumerate(contexts)
        ])

        # Prepare the prompt
        prompt = f"""
Based on the following contexts, please answer the query: "{query}"

{context_text}

Answer:
"""

        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1024
                }
            },
            stream=False  # We want a complete response
        )

        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"Error generating answer: Ollama API returned status code {response.status_code}"

        # Parse the response
        response_text = response.text
        responses = [json.loads(line) for line in response_text.strip().split('\n')]
        full_response = ''.join(r.get('response', '') for r in responses)

        logger.info(f"Generated answer length: {len(full_response)}")
        return full_response

    except Exception as e:
        logger.error(f"Error generating answer with Ollama: {str(e)}")
        return f"Error generating answer: {str(e)}"

def generate_answer(query, model_name, collection_name='documents'):
    """
    Generate an answer for a query - Always uses Phi model from Ollama

    Args:
        query (str): The query to answer
        model_name (str): Name of the LLM model (ignored, always uses phi)
        collection_name (str): Name of the Qdrant collection

    Returns:
        dict: The generated answer with metadata
    """
    try:
        logger.info(f"Generating answer for query: {query}")
        # Always use phi from Ollama
        model_to_use = "phi"
        logger.info(f"Using hardcoded model: {model_to_use}")

        # Check if this is a summary request
        if any(keyword in query.lower() for keyword in ["summary", "summarize", "summarize document"]):
            logger.info("Summary request detected")
            summary = generate_summary(collection_name)
            return {
                "answer": summary,
                "sources": [],
                "success": True
            }

        # Get relevant contexts
        contexts = get_question_contexts(query, collection_name)

        if not contexts:
            logger.warning("No relevant contexts found")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "success": True
            }

        # Generate answer using Ollama
        answer = generate_answer_with_ollama(query, contexts, model_to_use)

        # Extract sources for citation
        sources = []
        for context in contexts:
            source = {
                "source": context["source"],
                "page": context["page"],
                "score": context["score"]
            }
            if source not in sources:
                sources.append(source)

        result = {
            "answer": answer,
            "sources": sources,
            "success": True
        }

        logger.info(f"Generated answer with {len(sources)} sources")
        return result

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return {
            "answer": f"I encountered an error while trying to answer your question: {str(e)}",
            "sources": [],
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Generate answers using local LLM')
    parser.add_argument('query', help='The query to answer')
    parser.add_argument('--model_name', default='phi', help='Name of the LLM model (ignored, always uses phi)')
    parser.add_argument('--collection_name', default='documents', help='Name of the Qdrant collection')

    args = parser.parse_args()

    result = generate_answer(args.query, args.model_name, args.collection_name)
    print(json.dumps(result))

if __name__ == "__main__":
    main()