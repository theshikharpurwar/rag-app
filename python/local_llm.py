import sys
import json
import logging
import requests
import argparse
from qdrant_client import QdrantClient

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_answer(query, collection_name="documents", model_name="phi"):
    try:
        # Connect to Qdrant
        client = QdrantClient("localhost", port=6333)

        # Generate embedding for the query using the Ollama API
        logger.info(f"Generating embedding for query: {query}")
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "all-minilm", "prompt": query}
        )

        if response.status_code != 200:
            logger.error(f"Failed to generate embedding: {response.text}")
            return {"answer": "Error generating query embedding", "sources": []}

        query_embedding = response.json()["embedding"]

        # Search for relevant content
        logger.info(f"Searching for relevant content in collection: {collection_name}")
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5  # Retrieve top 5 most relevant chunks
        )

        # Check if we found any relevant content
        if not search_results:
            logger.info("No relevant content found")
            return {
                "answer": "I couldn't find any relevant information in the uploaded document to answer your question.",
                "sources": []
            }

        # Extract text and metadata from search results
        contexts = []
        sources = []

        for result in search_results:
            if result.score < 0.6:  # Filter out low relevance results
                continue

            payload = result.payload
            text = payload.get("text", "")
            page_num = payload.get("page_num", "unknown")
            doc_name = payload.get("doc_name", "unknown")

            if text:
                contexts.append(f"Content: {text}")
                sources.append({"page": page_num, "document": doc_name, "score": round(result.score, 2)})

        if not contexts:
            return {
                "answer": "While I found some related content, it doesn't appear to directly answer your question.",
                "sources": []
            }

        # Combine all contexts into a single string
        context_text = "\n\n".join(contexts)

        # Prepare prompt for the LLM
        is_summary = any(term in query.lower() for term in ["summarize", "summary", "summarization", "overview"])

        if is_summary:
            prompt = f"""
You are an AI assistant tasked with providing accurate information based ONLY on the provided document content.

DOCUMENT CONTENT:
{context_text}

QUERY: {query}

INSTRUCTIONS:
1. Answer ONLY based on the information in the document content provided above.
2. If the content appears in a list or bullet point format, PRESERVE that format in your answer.
3. If the information to answer the query is not in the provided content, state "I don't have enough information to answer this question."
4. DO NOT make up or hallucinate any information not present in the document content.
5. Keep your answer clear, concise, and directly relevant to the query.
6. Provide a comprehensive summary based on all the provided content.

YOUR ANSWER:
"""
        else:
            prompt = f"""
You are an AI assistant tasked with providing accurate information based ONLY on the provided document content.

DOCUMENT CONTENT:
{context_text}

QUERY: {query}

INSTRUCTIONS:
1. Answer ONLY based on the information in the document content provided above.
2. If the content appears in a list or bullet point format, PRESERVE that format in your answer.
3. If the information to answer the query is not in the provided content, state "I don't have enough information to answer this question."
4. DO NOT make up or hallucinate any information not present in the document content.
5. Keep your answer clear, concise, and directly relevant to the query.

YOUR ANSWER:
"""

        # Generate answer using Ollama
        logger.info(f"Generating answer using model: {model_name}")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )

        if response.status_code != 200:
            logger.error(f"Failed to generate answer: {response.text}")
            return {"answer": "Error generating answer", "sources": []}

        answer = response.json().get("response", "")

        # Format sources more clearly for citation
        if sources:
            return {
                "answer": answer,
                "sources": sources
            }
        else:
            return {
                "answer": answer,
                "sources": []
            }

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return {"answer": f"Error: {str(e)}", "sources": []}

def main():
    parser = argparse.ArgumentParser(description="Generate answers to queries based on document content")
    parser.add_argument("query", help="The query to answer")
    parser.add_argument("--collection_name", default="documents", help="Name of the Qdrant collection")
    parser.add_argument("--model_name", default="phi", help="Name of the model to use")

    args = parser.parse_args()

    result = generate_answer(args.query, args.collection_name, args.model_name)
    print(json.dumps(result))

if __name__ == "__main__":
    main()