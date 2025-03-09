# D:\rag-app\python\embeddings\embed_factory.py

import logging
from .local_embed import LocalEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedder(model_name=None, **kwargs):
    """
    Factory function to get an embedder instance

    Args:
        model_name (str, optional): Name of the model to use
        **kwargs: Additional arguments for the embedder

    Returns:
        object: An embedder instance
    """
    logger.info(f"Getting embedder instance for model: {model_name}")

    # Use a default model if none provided
    if not model_name:
        model_name = 'all-MiniLM-L6-v2'
        logger.info(f"No model specified, using default: {model_name}")

    # Currently we only support local sentence-transformers models
    return LocalEmbedder(model_name=model_name)