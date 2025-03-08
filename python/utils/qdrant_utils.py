# D:\rag-app\python\utils\qdrant_utils.py

import sys
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_collection(collection_name, vector_size=384):
    """
    Reset a Qdrant collection by recreating it

    Args:
        collection_name (str): Name of the collection
        vector_size (int): Size of the vectors

    Returns:
        bool: True if successful
    """
    try:
        logger.info(f"Resetting collection: {collection_name}")
        
        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)
        
        # Delete collection if it exists
        if collection_exists:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)
        
        # Create new collection
        logger.info(f"Creating new collection: {collection_name} with vector size {vector_size}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        
        logger.info(f"Collection {collection_name} reset successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error resetting collection: {str(e)}")
        return False

def clear_collection(collection_name):
    """
    Clear all points from a Qdrant collection without deleting the collection

    Args:
        collection_name (str): Name of the collection

    Returns:
        bool: True if successful
    """
    try:
        logger.info(f"Clearing collection: {collection_name}")
        
        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)
        
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python qdrant_utils.py <command> <collection_name> [vector_size]")
        sys.exit(1)
    
    command = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "documents"
    vector_size = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    
    if command == "reset":
        success = reset_collection(collection_name, vector_size)
        if success:
            print(f"Collection {collection_name} reset successfully")
        else:
            print(f"Failed to reset collection {collection_name}")
            sys.exit(1)
    elif command == "clear":
        success = clear_collection(collection_name)
        if success:
            print(f"Collection {collection_name} cleared successfully")
        else:
            print(f"Failed to clear collection {collection_name}")
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()