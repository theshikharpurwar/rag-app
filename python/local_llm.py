# D:\rag-app\python\local_llm.py

import sys
import json
import argparse
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from embeddings.embed_factory import EmbeddingModelFactory
from llm.llm_factory import LLMFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_similar(client, collection_name, query_embedding, limit=5):
    """
    Search for similar documents in Qdrant

    Args:
        client: QdrantClient instance
        collection_name (str): Name of the collection
        query_embedding (list): Query embedding vector
        limit (int): Maximum number of results

    Returns:
        list: List of search results
    """
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        logger.info(f"Found {len(search_result)} results")
        return search_result
    except Exception as e:
        logger.error(f"Error searching for similar documents: {str(e)}")
        return []

def generate_answer(query, collection_name="documents", model_name="phi"):
    """
    Generate an answer based on the query using RAG

    Args:
        query (str): User query
        collection_name (str): Name of the Qdrant collection
        model_name (str): Name of the language model

    Returns:
        dict: Answer and sources
    """
    try:
        logger.info(f"Generating answer for: '{query}'")
        logger.info(f"Using model: {model_name}, collection: {collection_name}")

        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)

        try:
            # Initialize the embedder
            embedder = EmbeddingModelFactory.get_embedder()

            # Generate query embedding
            query_embedding = embedder.get_embedding(query, input_type="text")

            # Search for similar documents
            search_result = search_similar(client, collection_name, query_embedding)

        except Exception as e:
            logger.error(f"Error during embedding or search: {str(e)}")
            return {
                "answer": f"Error during search: {str(e)}",
                "sources": []
            }

        if not search_result:
            logger.warning("No relevant content found")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }

        try:
            # Initialize the LLM
            llm = LLMFactory.get_llm(model_name)

            # Generate answer with sources
            result = llm.generate_with_sources(query, search_result)
            return result

        except Exception as e:
            logger.error(f"Error generating answer with LLM: {str(e)}")

            # Fallback: Generate a simple response based on retrieved content
            answer = "Based on the document, I found these relevant sections:\n\n"
            sources = []

            for i, doc in enumerate(search_result[:3]):
                payload = doc.payload
                content = payload.get("text", "No text available")
                page_num = payload.get("page_num", "Unknown")
                filename = payload.get("filename", "Unknown document")

                answer += f"From page {page_num}: {content[:300]}...\n\n"

                sources.append({
                    "page": page_num,
                    "filename": filename,
                    "text": content[:200] + ("..." if len(content) > 200 else "")
                })

            return {
                "answer": answer,
                "sources": sources
            }

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": []
        }

def generate_summary(collection_name="documents", model_name="phi"):
    """
    Generate a summary of the document

    Args:
        collection_name (str): Name of the Qdrant collection
        model_name (str): Name of the language model

    Returns:
        dict: Summary and sources
    """
    try:
        logger.info("Summary request detected, fetching all content")

        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)

        # Scroll through all text points in the collection
        scroll_result = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="text")
                    )
                ]
            ),
            limit=100
        )

        points = scroll_result[0]
        logger.info(f"Found {len(points)} text points in the collection")

        if not points:
            logger.warning("No text content found in the collection")
            return {
                "answer": "No document content found to summarize.",
                "sources": []
            }

        try:
            # Initialize the LLM
            llm = LLMFactory.get_llm(model_name)

            # Generate summary with sources
            result = llm.generate_with_sources("Please provide a comprehensive summary of this document.", points)
            return result

        except Exception as e:
            logger.error(f"Error generating summary with LLM: {str(e)}")

            # Fallback: Generate a simple summary based on content
            answer = "Here's what I found in the document:\n\n"
            sources = []

            for i, point in enumerate(points[:10]):  # Limit to first 10 points
                payload = point.payload
                text = payload.get("text", "No text available")
                page_num = payload.get("page_num", "Unknown")
                filename = payload.get("filename", "Unknown document")

                answer += f"From page {page_num}: {text[:200]}...\n\n"

                sources.append({
                    "page": page_num,
                    "filename": filename,
                    "text": text[:200] + ("..." if len(text) > 200 else "")
                })

            return {
                "answer": answer,
                "sources": sources
            }

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return {
            "answer": f"Error generating summary: {str(e)}",
            "sources": []
        }

def main():
    try:
        parser = argparse.ArgumentParser(description="Generate answers using RAG")
        parser.add_argument("query", help="User query")
        parser.add_argument("--collection_name", default="documents", help="Name of the Qdrant collection")
        parser.add_argument("--model_name", default="phi", help="Name of the language model")

        args = parser.parse_args()

        # Check if the query is asking for a summary
        if args.query.lower() in ["summarize", "summary", "summarize document", "give me a summary", "give full summary"]:
            result = generate_summary(
                collection_name=args.collection_name,
                model_name=args.model_name
            )
        else:
            # Generate answer for normal queries
            result = generate_answer(
                args.query,
                collection_name=args.collection_name,
                model_name=args.model_name
            )

        # Print the result as JSON for parsing by Node.js
        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        error_result = {
            "answer": f"An error occurred: {str(e)}",
            "sources": []
        }
        print(json.dumps(error_result, ensure_ascii=False))

if __name__ == "__main__":
    main()