# D:\rag-app\python\embeddings\embed_factory.py

import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedder(model_name, model_path=None, **kwargs):
    """
    Factory function to get the appropriate embedder based on model name.

    Args:
        model_name (str): Name of the model to use ('clip' or 'colpali')
        model_path (str, optional): Path to the model. Defaults to None.
        **kwargs: Additional arguments to pass to the embedder.

    Returns:
        Embedder: An instance of the appropriate embedder class.
    """
    logger.info(f"Creating embedder for model: {model_name} from {model_path}")

    if model_name.lower() == 'clip':
        from embeddings.clip_embed import ClipEmbedder
        return ClipEmbedder(model_path, **kwargs)
    elif model_name.lower() == 'colpali':
        from embeddings.colpali_embed import ColpaliEmbedder
        return ColpaliEmbedder(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")