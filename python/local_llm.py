# python/local_llm.py
import sys
import os
import json
import logging
from PIL import Image
from qdrant_client import QdrantClient, models
from embeddings.embed_factory import EmbeddingModelFactory
from llm.llm_factory import LLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_document(question, pdf_id, collection_name, images_dir,
                   embed_model_name, embed_model_path,
                   llm_model_name, llm_model_path,
                   llm_params=None):
    """Query a document with a question and generate an answer"""

    # Load embedding model for query
    if isinstance(llm_params, str):
        llm_params = json.loads(llm_params)

    try:
        embedder = EmbeddingModelFactory.create_model(embed_model_name, embed_model_path)
        logger.info(f"Embedding model {embed_model_name} loaded successfully")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise

    # Generate embedding for the query
    query_embedding = embedder.get_query_embedding(question)

    # Connect to Qdrant and search for relevant pages
    try:
        client = QdrantClient(url="http://localhost:6333")
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="pdf_id", match=models.MatchValue(value=pdf_id))]
            ),
            limit=3  # Get top 3 most relevant pages
        )
        logger.info(f"Found {len(search_result)} relevant pages")
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}")
        raise

    if not search_result:
        return "No relevant content found for this question."

    # Get image paths from search results
    image_paths = [hit.payload["image_path"] for hit in search_result]

    # Load LLM model
    try:
        llm = LLMFactory.create_model(llm_model_name, llm_model_path, llm_params)
        logger.info(f"LLM model {llm_model_name} loaded successfully")
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        raise

    # Generate response
    try:
        response = llm.generate_response(question, image_paths)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 10:
        print(json.dumps({
            "error": "Usage: local_llm.py <question> <pdf_id> <collection_name> <images_dir> <embed_model_name> <embed_model_path> <llm_model_name> <llm_model_path> [<llm_params>]"
        }))
        sys.exit(1)

    question = sys.argv[1]
    pdf_id = sys.argv[2]
    collection_name = sys.argv[3]
    images_dir = sys.argv[4]
    embed_model_name = sys.argv[5]
    embed_model_path = sys.argv[6]
    llm_model_name = sys.argv[7]
    llm_model_path = sys.argv[8]
    llm_params = sys.argv[9] if len(sys.argv) > 9 else "{}"

    try:
        response = query_document(
            question, pdf_id, collection_name, images_dir,
            embed_model_name, embed_model_path,
            llm_model_name, llm_model_path, llm_params
        )
        print(response)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)