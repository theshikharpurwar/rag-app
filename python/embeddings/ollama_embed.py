# D:\rag-app\python\embeddings\ollama_embed.py

import logging
import os
import requests

from config.models import DEFAULT_VECTOR_SIZE, EMBED_BATCH_SIZE, OLLAMA_KEEP_ALIVE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _zero_vectors(count: int, dim: int) -> list[list[float]]:
    return [[0.0] * dim for _ in range(count)]


def _pad_embeddings(vectors: list, target_len: int, dim: int) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(target_len):
        if i < len(vectors) and vectors[i]:
            row = list(vectors[i])
            if len(row) == dim:
                out.append(row)
            else:
                logger.warning(
                    "Embedding length %s != expected %s at index %s; using zero vector",
                    len(row),
                    dim,
                    i,
                )
                out.append([0.0] * dim)
        else:
            out.append([0.0] * dim)
    return out


class OllamaEmbedder:
    """
    Embedder that uses Ollama API for generating embeddings
    This eliminates the need for heavy ML libraries like torch, transformers, etc.
    """

    def __init__(
        self,
        model_name="nomic-embed-text:v1.5",
        api_base=None,
        batch_size=None,
        zero_vector_dim=None,
        keep_alive=None,
    ):
        """
        Initialize the OllamaEmbedder

        Args:
            model_name (str): Name of the Ollama embedding model to use
            api_base (str, optional): Base URL for Ollama API. Defaults to http://localhost:11434
            batch_size (int, optional): Max texts per /api/embed request (default from EMBED_BATCH_SIZE)
            zero_vector_dim (int, optional): Dimension for fallback zero vectors (default DEFAULT_VECTOR_SIZE)
            keep_alive (str, optional): Ollama keep_alive for embed requests (default OLLAMA_KEEP_ALIVE)
        """
        self.model_name = model_name
        self.batch_size = int(batch_size if batch_size is not None else EMBED_BATCH_SIZE)
        self.zero_vector_dim = int(zero_vector_dim if zero_vector_dim is not None else DEFAULT_VECTOR_SIZE)
        self.keep_alive = keep_alive if keep_alive is not None else OLLAMA_KEEP_ALIVE

        ollama_host = os.environ.get('OLLAMA_HOST_URL', 'http://localhost:11434')
        self.api_base = api_base if api_base else f"{ollama_host}/api"

        logger.info(f"Initializing OllamaEmbedder with model: {self.model_name}")
        logger.info(f"Using Ollama API at: {self.api_base}")
        logger.info(f"Embed batch size: {self.batch_size}")

        try:
            base_url = self.api_base.rsplit('/api', 1)[0]
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                model_found = (
                    self.model_name in available_models or
                    f"{self.model_name}:latest" in available_models
                )
                if not model_found:
                    logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
                    logger.info(f"You may need to run: ollama pull {self.model_name}")
                else:
                    logger.info(f"Model {self.model_name} is available")
            else:
                logger.warning(f"Could not check available models. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking Ollama API: {str(e)}")
            logger.warning(f"Make sure Ollama is running at {self.api_base}")

    def _encode_legacy(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Per-item POST /api/embeddings (legacy) for older Ollama without /api/embed."""
        embeddings: list[list[float]] = []
        for text in texts:
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": text,
                    "keep_alive": self.keep_alive,
                    "options": {
                        "task": task_type
                    },
                }
                response = requests.post(
                    f"{self.api_base}/embeddings",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding', [])
                    if embedding:
                        embeddings.append(embedding)
                        logger.debug(f"Generated embedding of size {len(embedding)} for text: {text[:50]}...")
                    else:
                        logger.error(f"Empty embedding returned for text: {text[:50]}...")
                        embeddings.append([0.0] * self.zero_vector_dim)
                else:
                    logger.error(f"Ollama API error {response.status_code}: {response.text}")
                    embeddings.append([0.0] * self.zero_vector_dim)
            except Exception as e:
                logger.error(f"Error generating embedding for text '{text[:50]}...': {str(e)}")
                embeddings.append([0.0] * self.zero_vector_dim)
        return embeddings

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

        all_embeddings: list[list[float]] = []
        dim = self.zero_vector_dim

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                timeout = max(30, 5 * len(batch))
                resp = requests.post(
                    f"{self.api_base}/embed",
                    json={
                        "model": self.model_name,
                        "input": batch,
                        "keep_alive": self.keep_alive,
                        "options": {"task": task_type},
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                if resp.status_code == 404:
                    logger.warning(
                        "/api/embed returned 404; falling back to legacy /api/embeddings for this batch"
                    )
                    all_embeddings.extend(self._encode_legacy(batch, task_type))
                    continue
                resp.raise_for_status()
                result = resp.json()
                vectors = result.get("embeddings") or []
                if len(vectors) != len(batch):
                    logger.error(
                        "Ollama /api/embed returned %s vectors for %s inputs; padding",
                        len(vectors),
                        len(batch),
                    )
                    vectors = _pad_embeddings(vectors, len(batch), dim)
                all_embeddings.extend(vectors)
            except Exception as e:
                logger.error(f"Batch embed failed ({len(batch)} items): {e}")
                all_embeddings.extend(_zero_vectors(len(batch), dim))

        logger.info(f"Generated {len(all_embeddings)} embeddings using Ollama")
        return all_embeddings

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
        return embeddings[0] if embeddings else [0.0] * self.zero_vector_dim
