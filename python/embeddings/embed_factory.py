# D:\rag-app\python\embeddings\embed_factory.py

import logging
from .ollama_embed import OllamaEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedder(model_name=None, embedder_type="ollama", **kwargs):
    """
    Factory function to get an embedder instance

    Args:
        model_name (str, optional): Name of the model to use
        embedder_type (str): Type of embedder ("ollama")
        **kwargs: Additional arguments for the embedder

    Returns:
        object: An embedder instance
    """
    logger.info(f"Getting embedder instance for model: {model_name}, type: {embedder_type}")

    # Use Ollama embedder (eliminates heavy ML dependencies)
    if not model_name:
        model_name = 'nomic-embed-text:v1.5'
        logger.info(f"No model specified, using default: {model_name}")
    
    return OllamaEmbedder(model_name=model_name)