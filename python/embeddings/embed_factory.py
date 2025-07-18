# D:\rag-app\python\embeddings\embed_factory.py

import logging
from .local_embed import LocalEmbedder
from .nomic_embed import NomicEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedder(model_name=None, embedder_type="local", **kwargs):
    """
    Factory function to get an embedder instance

    Args:
        model_name (str, optional): Name of the model to use
        embedder_type (str): Type of embedder ("local", "nomic")
        **kwargs: Additional arguments for the embedder

    Returns:
        object: An embedder instance
    """
    logger.info(f"Getting embedder instance for model: {model_name}, type: {embedder_type}")

    if embedder_type == "nomic":
        # For Nomic embedders, we use local models - no API key needed
        device = kwargs.get('device')
        cache_dir = kwargs.get('cache_dir')
        return NomicEmbedder(device=device, cache_dir=cache_dir)
    else:
        # Use a default model if none provided for local embedders
        if not model_name:
            model_name = 'all-MiniLM-L6-v2'
            logger.info(f"No model specified, using default: {model_name}")
        
        return LocalEmbedder(model_name=model_name)