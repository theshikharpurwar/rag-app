# D:\rag-app\python\utils\qdrant_utils.py

import logging
import json
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_collection(collection_name="documents", host="localhost", port=6333):
    """Clear a Qdrant collection by deleting and recreating it

    Args:
        collection_name (str): Name of the collection to clear
        host (str): Qdrant server host
        port (int): Qdrant server port

    Returns:
        dict: Status information about the operation
    """
    client = QdrantClient(host=host, port=port)
    logger.info(f"Attempting to clear Qdrant collection: {collection_name}")

    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if collection_exists:
            # Get collection info to preserve configuration
            collection_info = client.get_collection(collection_name)
            vector_size = collection_info.config.params.vectors.size
            distance = collection_info.config.params.vectors.distance

            # Delete the collection
            client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")

            # Recreate with same parameters
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": vector_size,
                    "distance": distance
                }
            )
            logger.info(f"Recreated collection: {collection_name} with vector size {vector_size}")

            return {
                "success": True,
                "message": f"Collection {collection_name} cleared and recreated",
                "details": {
                    "vector_size": vector_size,
                    "distance": distance
                }
            }
        else:
            logger.info(f"Collection {collection_name} does not exist, creating new")

            # Create new collection with default parameters
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": 512,  # Default size for CLIP model
                    "distance": "Cosine"
                }
            )

            return {
                "success": True,
                "message": f"Created new collection: {collection_name}",
                "details": {
                    "vector_size": 512,
                    "distance": "Cosine"
                }
            }
    except Exception as e:
        error_msg = f"Error clearing Qdrant collection: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg
        }

if __name__ == "__main__":
    # This allows running the script directly for testing
    import sys
    collection = "documents" if len(sys.argv) < 2 else sys.argv[1]
    result = clear_collection(collection)
    print(json.dumps(result))