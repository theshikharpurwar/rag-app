# python/embeddings/embed_factory.py

import importlib
import os
import sys

def get_embedder(model_name, model_path, **kwargs):
    """
    Factory function to get the appropriate embedder based on model name

    Args:
        model_name (str): Name of the embedding model (e.g., 'colpali')
        model_path (str): Path or identifier for the model
        **kwargs: Additional parameters for the embedder

    Returns:
        An embedder instance that can embed images and text
    """
    try:
        # Try to import the specific embedder module
        if model_name == 'colpali':
            from python.embeddings.colpali_embed import ColpaliEmbedder
            return ColpaliEmbedder(model_path, **kwargs)
        else:
            # Try dynamic import
            module_name = f"python.embeddings.{model_name}_embed"
            module = importlib.import_module(module_name)

            # Get the embedder class
            embedder_class = getattr(module, f"{model_name.capitalize()}Embedder")

            # Create and return an instance
            return embedder_class(model_path, **kwargs)
    except (ImportError, AttributeError) as e:
        # If specific embedder not found, use a default embedder
        print(f"Warning: Could not load embedder for {model_name}, using default ColpaliEmbedder")
        from python.embeddings.colpali_embed import ColpaliEmbedder
        return ColpaliEmbedder(model_path, **kwargs)