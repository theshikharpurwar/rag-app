import os
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModelFactory:
    """Factory class for creating embedding model instances"""

    @staticmethod
    def get_embedder(model_path: str, **kwargs: Any):
        """
        Get an embedder instance based on the model path

        Args:
            model_path: Path or identifier for the embedding model
            **kwargs: Additional arguments to pass to the embedder

        Returns:
            An embedder instance
        """
        logger.info(f"Creating embedder for model: {model_path}")

        # Determine which embedder to use based on the model path
        if "clip" in model_path.lower():
            from embeddings.clip_embed import ClipEmbedder
            return ClipEmbedder(model_path=model_path, **kwargs)
        elif "colpali" in model_path.lower():
            from embeddings.colpali_embed import ColpaliEmbedder
            return ColpaliEmbedder(model_path=model_path, **kwargs)
        else:
            # Default to CLIP embedder
            logger.warning(f"Unknown model type for {model_path}, defaulting to CLIP embedder")
            from embeddings.clip_embed import ClipEmbedder
            return ClipEmbedder(model_path="openai/clip-vit-base-patch32", **kwargs)

# For backward compatibility
def get_embedder(model_path: str, **kwargs: Any):
    """Legacy function for backward compatibility"""
    return EmbeddingModelFactory.get_embedder(model_path, **kwargs)