# D:\rag-app\python\llm\ollama_llm.py

import logging
import requests
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaLLM:
    """
    Class to generate text responses using the Ollama API
    """

    def __init__(self, model_name='phi2', api_base=None):
        """
        Initialize the OllamaLLM with a model

        Args:
            model_name (str): Name of the Ollama model to use
            api_base (str, optional): Base URL for Ollama API. Defaults to http://localhost:11434/api
        """
        logger.info(f"Initializing OllamaLLM with model: {model_name}")
        self.model_name = model_name
        self.api_base = api_base if api_base else "http://localhost:11434/api"
        logger.info(f"Using Ollama API at: {self.api_base}")

        # Verify that Ollama is running and the model is available
        try:
            # Extract the base URL without the /api suffix for the tags endpoint
            base_url = self.api_base.rsplit('/api', 1)[0]
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                if model_name not in available_models:
                    logger.warning(f"Model {model_name} not found in available models: {available_models}")
                    logger.info(f"You may need to run: ollama pull {model_name}")
                else:
                    logger.info(f"Model {model_name} is available")
            else:
                logger.warning(f"Could not check available models. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking Ollama API: {str(e)}")
            logger.warning(f"Make sure Ollama is running at {self.api_base}")

    def generate_response(self, prompt, context=None, max_tokens=1000, temperature=0.7):
        """
        Generate a response for the provided prompt

        Args:
            prompt (str): The prompt to generate a response for
            context (list, optional): Additional context for the prompt
            max_tokens (int, optional): Maximum number of tokens to generate
            temperature (float, optional): Sampling temperature

        Returns:
            str: The generated response
        """
        if not prompt:
            logger.warning("Empty prompt provided")
            return "Please provide a question or prompt."

        logger.info(f"Generating response for prompt: {prompt[:50]}...")

        try:
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }

            # Make the request to the Ollama API
            logger.info(f"Sending request to: {self.api_base}/generate")
            response = requests.post(f"{self.api_base}/generate", json=payload, timeout=60)

            if response.status_code == 200:
                # Extract the response text
                response_text = response.text

                # Parse the response
                responses = [json.loads(line) for line in response_text.strip().split('\n')]
                full_response = ''.join(r.get('response', '') for r in responses)

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