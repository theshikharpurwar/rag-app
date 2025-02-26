# python/embeddings/embed_factory.py
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModelFactory:
    """Factory for creating embedding models"""

    @staticmethod
    def create_model(model_name, model_path, params=None):
        """Create and return the specified embedding model"""
        if params is None:
            params = {}

        model_name = model_name.lower()

        if model_name == 'colpali':
            from .colpali_embed import ColPaliEmbedder
            return ColPaliEmbedder(model_path, params)
        elif model_name == 'sentence_transformer':
            from .sentence_transformer_embed import SentenceTransformerEmbedder
            return SentenceTransformerEmbedder(model_path, params)
        # Add more models as needed
        else:
            logger.error(f"Unsupported embedding model: {model_name}")
            raise ValueError(f"Unsupported embedding model: {model_name}")