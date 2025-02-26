# python/llm/llm_factory.py
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory for creating LLM models"""

    @staticmethod
    def create_model(model_name, model_path, params=None):
        """Create and return the specified LLM model"""
        if params is None:
            params = {}

        model_name = model_name.lower()

        if model_name == 'qwen':
            from .qwen_llm import QwenVLModel
            return QwenVLModel(model_path, params)
        elif model_name == 'llama':
            from .llama_llm import LlamaVLModel
            return LlamaVLModel(model_path, params)
        # Add more models as needed
        else:
            logger.error(f"Unsupported LLM model: {model_name}")
            raise ValueError(f"Unsupported LLM model: {model_name}")