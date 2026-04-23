# D:\rag-app\python\llm\ollama_llm.py

import logging
import requests
import json
import os

from config.models import OLLAMA_KEEP_ALIVE, OLLAMA_NUM_BATCH, OLLAMA_NUM_CTX

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaLLM:
    """
    Class to generate text responses using the Ollama API
    """

    def __init__(self, model_name=None, api_base=None, num_batch=None, num_ctx=None, keep_alive=None):
        """
        Initialize the OllamaLLM with a model

        Args:
            model_name (str): Name of the Ollama model to use
            api_base (str, optional): Base URL for Ollama API. Defaults to http://localhost:11434/api
            num_batch (int, optional): Ollama options.num_batch (default from OLLAMA_NUM_BATCH)
            num_ctx (int | None, optional): Ollama options.num_ctx; omit when None or 0 (model default)
            keep_alive (str, optional): Top-level keep_alive for /api/chat (default OLLAMA_KEEP_ALIVE)
        """
        # Get model from environment variable or use the provided one or default to phi2
        self.model_name = model_name or os.environ.get('LLM_MODEL', 'phi2')
        self.num_batch = int(num_batch if num_batch is not None else OLLAMA_NUM_BATCH)
        if num_ctx is not None:
            self.num_ctx = int(num_ctx) if int(num_ctx) > 0 else None
        else:
            self.num_ctx = OLLAMA_NUM_CTX
        self.keep_alive = keep_alive if keep_alive is not None else OLLAMA_KEEP_ALIVE
        logger.info(f"Initializing OllamaLLM with model: {self.model_name}")

        # Get the API base URL from the environment or use the provided one
        # The OLLAMA_HOST_URL env var is expected to end with the base URL (e.g., http://host.docker.internal:11434)
        ollama_host = os.environ.get('OLLAMA_HOST_URL', 'http://localhost:11434')
        self.api_base = api_base if api_base else f"{ollama_host}/api"
        logger.info(f"Using Ollama API at: {self.api_base}")

        # Verify that Ollama is running and the model is available
        try:
            # Extract the base URL without the /api suffix for the tags endpoint
            base_url = self.api_base.rsplit('/api', 1)[0]
            response = requests.get(f"{base_url}/api/tags")
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

    def generate_response(self, prompt, context=None, max_tokens=1000, temperature=0.7, messages=None):
        """
        Generate a response using the Ollama /api/chat endpoint with structured messages.
        Uses role-based message format that chat-tuned models (like qwen2.5vl) understand best.

        Args:
            prompt (str): Used as the final user message if messages is None
            context (list, optional): Unused, kept for API compatibility
            max_tokens (int, optional): Maximum number of tokens to generate
            temperature (float, optional): Sampling temperature
            messages (list, optional): Full conversation as [{"role": ..., "content": ...}]

        Returns:
            str: The generated response
        """
        if not prompt and not messages:
            logger.warning("Empty prompt provided")
            return "Please provide a question or prompt."

        # Use provided messages or build a simple single-turn conversation
        chat_messages = messages if messages else [{"role": "user", "content": prompt}]

        logger.info(f"Sending {len(chat_messages)} message(s) to /api/chat...")

        try:
            options = {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_batch": self.num_batch,
            }
            if self.num_ctx is not None:
                options["num_ctx"] = self.num_ctx
            payload = {
                "model": self.model_name,
                "messages": chat_messages,
                "stream": True,
                "options": options,
                "keep_alive": self.keep_alive,
            }

            logger.info(f"Sending request to: {self.api_base}/chat")
            response = requests.post(f"{self.api_base}/chat", json=payload, timeout=120, stream=True)

            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        full_response += content
                        if chunk.get("done", False):
                            break

                logger.info(f"Successfully generated response: {full_response[:50]}...")
                return full_response
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Sorry, I encountered an error: {error_msg}"

        except requests.exceptions.Timeout:
            error_msg = f"Request to Ollama API timed out at {self.api_base}"
            logger.error(error_msg)
            return f"Sorry, the Ollama API request timed out. Please ensure Ollama is running at {self.api_base.split('/api')[0]}."

        except requests.exceptions.ConnectionError:
            error_msg = f"Could not connect to Ollama API at {self.api_base}"
            logger.error(error_msg)
            return f"Sorry, I could not connect to the Ollama API. Please ensure Ollama is running at {self.api_base.split('/api')[0]}."

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"Sorry, I encountered an error: {error_msg}"

    def generate_answer(self, query, retrieved_contexts):
        """
        Generate an answer for a query using retrieved contexts

        Args:
            query (str): The query to answer
            retrieved_contexts (list): List of retrieved contexts

        Returns:
            str: The generated answer
        """
        logger.info(f"Generating answer for query: {query}")

        if not retrieved_contexts or len(retrieved_contexts) == 0:
            logger.warning("No contexts provided for the query")
            return "I couldn't find any relevant information to answer your question."

        # Prepare the prompt with the retrieved contexts
        context_text = "\n\n".join([
            f"Context {i+1}:\n{ctx.get('text', '')}"
            for i, ctx in enumerate(retrieved_contexts)
        ])

        prompt = f"""
Based on the following contexts, please answer the query: "{query}"

{context_text}

Answer:
"""

        response = self.generate_response(prompt)
        logger.info(f"Generated answer: {response[:100]}...")

        return response