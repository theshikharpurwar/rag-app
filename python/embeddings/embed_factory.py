# D:\rag-app\python\embeddings\embed_factory.py

import logging
import sys
import os

# Add the parent directory to the path so we can import the local_embed module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from embeddings.local_embed import LocalEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModelFactory:
    """
    Factory class for creating embedding model instances
    """

    @staticmethod
    def get_embedder(model_name=None, **kwargs):
        """
        Get an embedder instance based on model name

        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments for specific models

        Returns:
            Embedder instance
        """
        # Default to all-MiniLM-L6-v2 if not specified
        if not model_name or model_name.lower() == "local":
            model_name = "all-MiniLM-L6-v2"

        logger.info(f"Creating local embedder with model {model_name}")
        return LocalEmbedder(model_name=model_name)