# D:\rag-app\python\llm\llm_factory.py

import logging
from .ollama_llm import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_llm(model_name):
    """
    Factory function to get an LLM instance

    Args:
        model_name (str): Name of the model to use

    Returns:
        object: An LLM instance
    """
    logger.info(f"Getting LLM instance for model: {model_name}")

    # Currently we only support Ollama models
    return OllamaLLM(model_name=model_name)