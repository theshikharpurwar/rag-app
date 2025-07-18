# D:\rag-app\python\embeddings\ollama_embed.py

import logging
import requests
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaEmbedder:
    """
    Embedder that uses Ollama API for generating embeddings
    This eliminates the need for heavy ML libraries like torch, transformers, etc.
    """

    def __init__(self, model_name="nomic-embed-text:v1.5", api_base=None):
        """
        Initialize the OllamaEmbedder

        Args:
            model_name (str): Name of the Ollama embedding model to use
            api_base (str, optional): Base URL for Ollama API. Defaults to http://localhost:11434
        """
        self.model_name = model_name
        
        # Get the API base URL from the environment or use the provided one
        ollama_host = os.environ.get('OLLAMA_HOST_URL', 'http://localhost:11434')
        self.api_base = api_base if api_base else f"{ollama_host}/api"
        
        logger.info(f"Initializing OllamaEmbedder with model: {self.model_name}")
        logger.info(f"Using Ollama API at: {self.api_base}")

        # Verify that Ollama is running and the model is available
        try:
            # Extract the base URL without the /api suffix for the tags endpoint
            base_url = self.api_base.rsplit('/api', 1)[0]
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
                    logger.info(f"You may need to run: ollama pull {self.model_name}")
                else:
                    logger.info(f"Model {self.model_name} is available")
            else:
                logger.warning(f"Could not check available models. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking Ollama API: {str(e)}")
            logger.warning(f"Make sure Ollama is running at {self.api_base}")

    def encode_text(self, texts, task_type="search_document"):
        """
        Generate embeddings for a list of texts using Ollama API

        Args:
            texts (list): List of text strings to embed
            task_type (str): Task type for the embeddings ("search_document" or "search_query")

        Returns:
            list: List of embedding vectors
        """
        if not texts:
            return []

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        
        for text in texts:
            try:
                # Prepare the request payload for Ollama embeddings API
                payload = {
                    "model": self.model_name,
                    "prompt": text,
                    "options": {
                        "task": task_type
                    }
                }

                # Make the API request
                response = requests.post(
                    f"{self.api_base}/embeddings",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding', [])
                    
                    if embedding:
                        embeddings.append(embedding)
                        logger.debug(f"Generated embedding of size {len(embedding)} for text: {text[:50]}...")
                    else:
                        logger.error(f"Empty embedding returned for text: {text[:50]}...")
                        embeddings.append([0.0] * 768)  # Fallback zero vector
                else:
                    logger.error(f"Ollama API error {response.status_code}: {response.text}")
                    embeddings.append([0.0] * 768)  # Fallback zero vector

            except Exception as e:
                logger.error(f"Error generating embedding for text '{text[:50]}...': {str(e)}")
                embeddings.append([0.0] * 768)  # Fallback zero vector

        logger.info(f"Generated {len(embeddings)} embeddings using Ollama")
        return embeddings

    def encode_single_text(self, text, task_type="search_document"):
        """
        Generate embedding for a single text

        Args:
            text (str): Text string to embed
            task_type (str): Task type for the embedding

        Returns:
            list: Embedding vector
        """
        embeddings = self.encode_text([text], task_type)
        return embeddings[0] if embeddings else [0.0] * 768
