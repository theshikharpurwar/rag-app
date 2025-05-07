# D:\rag-app\python\llm\llm_factory.py

import logging
import os
from .ollama_llm import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_llm(model_name=None):
    """
    Factory function to get an LLM instance

    Args:
        model_name (str, optional): Name of the model to use. If None, will use LLM_MODEL env var or default.

    Returns:
        object: An LLM instance
    """
    # Use provided model name, or get from env, or use default
    model_name = model_name or os.environ.get('LLM_MODEL', 'gemma3:1b')
    logger.info(f"Getting LLM instance for model: {model_name}")

    # Currently we only support Ollama models
    return OllamaLLM(model_name=model_name)