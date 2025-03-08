# D:\rag-app\python\llm\llm_factory.py

import logging
from .ollama_llm import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Factory class for creating LLM instances
    """

    @staticmethod
    def get_llm(model_name=None, **kwargs):
        """
        Get an LLM instance based on model name

        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments for specific models

        Returns:
            LLM instance
        """
        if not model_name:
            model_name = "phi"

        # Always use Ollama LLM with the specified model
        logger.info(f"Creating Ollama LLM with model {model_name}")
        return OllamaLLM(model_name=model_name)