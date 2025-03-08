# D:\rag-app\python\llm\mistral_llm.py

import os
import logging
import json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MistralLLM:
    """Mistral AI implementation for generating text responses"""

    def __init__(self, model_name="mistral-large-latest", api_key=None, temperature=0.7, max_tokens=1024):
        """
        Initialize the Mistral LLM

        Args:
            model_name (str): Mistral model name
            api_key (str): Mistral API key
            temperature (float): Temperature for generation
            max_tokens (int): Maximum tokens to generate
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("Mistral API key not provided")

        logger.info(f"Initializing Mistral LLM with model: {model_name}")
        self.client = MistralClient(api_key=self.api_key)

    def generate(self, prompt, context=None, system_prompt=None):
        """
        Generate text based on prompt and context

        Args:
            prompt (str): The user query
            context (str): Additional context (e.g., retrieved documents)
            system_prompt (str): System prompt to guide model behavior

        Returns:
            str: Generated text response
        """
        try:
            # Build messages
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append(ChatMessage(role="system", content=system_prompt))
            else:
                # Default system prompt for RAG
                default_system_prompt = (
                    "You are a helpful AI assistant. Your task is to answer questions based on the provided context. "
                    "If the question cannot be answered using the information provided, say 'I couldn't find any relevant information.' "
                    "Do not make up information that is not supported by the provided context."
                )
                messages.append(ChatMessage(role="system", content=default_system_prompt))

            # Add context to the user message if provided
            user_content = prompt
            if context:
                user_content = f"Context information is below:\n\n{context}\n\nQuestion: {prompt}\n\nAnswer:"

            messages.append(ChatMessage(role="user", content=user_content))

            logger.info(f"Generating response with Mistral for prompt: {prompt[:100]}...")

            # Generate response
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract response text
            generated_text = response.choices[0].message.content
            logger.info(f"Generated response: {generated_text[:100]}...")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text with Mistral LLM: {str(e)}")
            return f"Error generating response: {str(e)}"

    def generate_with_sources(self, query, results, num_results=3):
        """
        Generate a response based on search results with source citations

        Args:
            query (str): User query
            results (list): List of search results with text and metadata
            num_results (int): Number of top results to include

        Returns:
            dict: Response with answer and sources
        """
        try:
            # Limit to top results
            top_results = results[:num_results] if len(results) > num_results else results

            # Extract context from results
            context_parts = []
            sources = []

            for i, result in enumerate(top_results):
                result_text = result.get('text', result.get('payload', {}).get('text', ''))
                if result_text:
                    context_parts.append(f"[{i+1}] {result_text}")

                    # Collect source information
                    metadata = result.get('metadata', result.get('payload', {}))
                    source_info = {
                        'id': i+1,
                        'text': result_text[:100] + "..." if len(result_text) > 100 else result_text
                    }

                    # Add page number if available
                    if 'page_num' in metadata:
                        source_info['page'] = metadata['page_num']

                    # Add file name if available
                    if 'file_name' in metadata:
                        source_info['file'] = metadata['file_name']

                    sources.append(source_info)

            # Combine context
            combined_context = "\n\n".join(context_parts)

            # Create prompt with instructions to cite sources
            system_prompt = (
                "You are a helpful AI assistant. Answer the user's question based on the provided context information. "
                "Each piece of context is labeled with a number [1], [2], etc. When you use information from the context, "
                "cite the source by including its number in brackets at the end of the sentence or paragraph. "
                "If the question cannot be answered using the information provided, say 'I couldn't find any relevant information.' "
                "Do not make up information that is not supported by the provided context."
            )

            # Generate response
            generated_text = self.generate(query, combined_context, system_prompt)

            # Return formatted response with sources
            return {
                "answer": generated_text,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error generating response with sources: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": []
            }

    @staticmethod
    def format_results_for_llm(results):
        """Helper to format Qdrant results for the LLM"""
        context_elements = []

        for i, result in enumerate(results):
            if hasattr(result, 'payload') and result.payload:
                text = result.payload.get('text', '')
                page_num = result.payload.get('page_num', 'unknown')
                context_elements.append(f"[Document {i+1}, Page {page_num}]: {text}")

        return "\n\n".join(context_elements)