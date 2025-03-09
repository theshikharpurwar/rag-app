import sys
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_collection(collection_name):
    try:
        logger.info(f"Clearing collection: {collection_name}")
        client = QdrantClient("localhost", port=6333)

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name in collection_names:
            # Delete the entire collection instead of trying to delete points
            client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection {collection_name}")

            # Recreate the collection with the same parameters
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"Successfully recreated collection {collection_name}")
            print(f"Successfully cleared collection {collection_name}")
            return True
        else:
            logger.info(f"Collection {collection_name} does not exist, nothing to delete")
            print(f"Collection {collection_name} does not exist")
            return True
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        print(f"Failed to clear collection {collection_name}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qdrant_utils.py <collection_name>")
        sys.exit(1)

    collection_name = sys.argv[1]
    success = clear_collection(collection_name)
    if not success:
        sys.exit(1)
    sys.exit(0)