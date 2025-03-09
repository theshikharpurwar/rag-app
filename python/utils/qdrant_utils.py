# D:\rag-app\python\qdrant_utils.py

import argparse
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_collection(collection_name="documents", vector_size=384):
    """Reset a Qdrant collection by recreating it."""
    try:
        client = QdrantClient(host="localhost", port=6333)
        logger.info(f"Connected to Qdrant server")

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name in collection_names:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)

        # Create new collection
        logger.info(f"Creating new collection: {collection_name} with vector size: {vector_size}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

        logger.info(f"Collection {collection_name} reset successfully")
        print(f"Collection {collection_name} reset successfully")
        return True

    except Exception as e:
        logger.error(f"Error resetting collection: {str(e)}")
        print(f"Error resetting collection: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant collection utilities")
    parser.add_argument("action", choices=["reset_collection", "clear"], help="Action to perform")
    parser.add_argument("--collection_name", default="documents", help="Name of the collection")
    parser.add_argument("--vector_size", type=int, default=384, help="Size of the vectors")

    args = parser.parse_args()

    # Allow 'clear' as an alias for 'reset_collection'
    if args.action in ["reset_collection", "clear"]:
        success = reset_collection(args.collection_name, args.vector_size)
        sys.exit(0 if success else 1)