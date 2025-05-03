# D:\rag-app\python\cleanup_qdrant.py

import sys
import logging
import os
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_collection(collection_name):
    """
    Clear all points from a Qdrant collection

    Args:
        collection_name (str): Name of the collection

    Returns:
        bool: True if successful
    """
    try:
        logger.info(f"Clearing collection: {collection_name}")

        # Initialize Qdrant client
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
        client = QdrantClient(qdrant_host, port=qdrant_port)

        # Check if collection exists
        try:
            collections = client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)

            if not collection_exists:
                logger.info(f"Collection {collection_name} does not exist, nothing to clear")
                return True

            # Delete all points from the collection
            client.delete(
                collection_name=collection_name,
                points_selector=None  # This deletes all points
            )

            logger.info(f"Collection {collection_name} cleared successfully")
            return True

        except UnexpectedResponse as e:
            if "Not found" in str(e):
                logger.info(f"Collection {collection_name} does not exist, nothing to clear")
                return True
            else:
                raise

    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cleanup_qdrant.py <collection_name>")
        sys.exit(1)

    collection_name = sys.argv[1]
    success = clear_collection(collection_name)

    if success:
        print(f"Collection {collection_name} cleared successfully")
        sys.exit(0)
    else:
        print(f"Failed to clear collection {collection_name}")
        sys.exit(1)