# FILE: python/utils/qdrant_utils.py

import argparse
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
DEFAULT_COLLECTION = "documents"
DEFAULT_VECTOR_SIZE = 384 # Match all-MiniLM-L6-v2
# --- End Configuration ---


def reset_collection(collection_name=DEFAULT_COLLECTION, vector_size=DEFAULT_VECTOR_SIZE):
    """Reset a Qdrant collection by recreating it."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
        logger.info(f"Connected to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")

        # Check if collection exists and delete if it does
        try:
             client.get_collection(collection_name=collection_name)
             logger.info(f"Deleting existing collection: {collection_name}")
             client.delete_collection(collection_name=collection_name, timeout=60)
             logger.info(f"Collection {collection_name} deleted.")
        except Exception as e:
             if "Not found" in str(e) or "doesn't exist" in str(e) or ("status_code=404" in str(e)) or ("CollectionNotFoundException" in str(e)):
                  logger.info(f"Collection {collection_name} does not exist, no need to delete.")
             else:
                  logger.warning(f"Could not confirm deletion status for {collection_name}: {e}")

        # Create new collection
        logger.info(f"Creating new collection: {collection_name} with vector size: {vector_size}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            timeout=60
        )
        logger.info(f"Collection {collection_name} reset successfully with vector size {vector_size}")
        print(f"Collection {collection_name} reset successfully")
        return True

    except Exception as e:
        logger.error(f"Error resetting collection {collection_name}: {str(e)}", exc_info=True)
        print(f"Error resetting collection {collection_name}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant collection utilities")
    parser.add_argument("action", choices=["reset_collection", "clear"], help="Action to perform")
    parser.add_argument("--collection_name", default=DEFAULT_COLLECTION, help=f"Name of the collection (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--vector_size", type=int, default=DEFAULT_VECTOR_SIZE, help=f"Size of the vectors (default: {DEFAULT_VECTOR_SIZE})")

    args = parser.parse_args()

    if args.action in ["reset_collection", "clear"]:
        vector_size_to_use = args.vector_size
        if args.vector_size != DEFAULT_VECTOR_SIZE:
             logger.warning(f"Using non-default vector size: {args.vector_size}. Ensure this matches your embedding model!")
        success = reset_collection(args.collection_name, vector_size_to_use)
        sys.exit(0 if success else 1)