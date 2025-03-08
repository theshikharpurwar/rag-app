# D:\rag-app\python\llm\ollama_llm.py

import logging
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaLLM:
    """
    Class for generating text responses using Ollama's local models.
    """

    def __init__(self, model_name="phi", **kwargs):
        """
        Initialize the Ollama LLM

        Args:
            model_name (str): Name of the Ollama model
        """
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api"
        self.temperature = 0.7
        self.max_tokens = 1024
        logger.info(f"Initialized Ollama LLM with model: {model_name}")

    def generate(self, prompt, context=None, conversation_history=None):
        """
        Generate a response based on the prompt and context

        Args:
            prompt (str): The query or instruction
            context (str, optional): Additional context for the query
            conversation_history (list, optional): List of previous messages

        Returns:
            str: Generated response
        """
        try:
            # Create system prompt with context if provided
            system_content = "You are a helpful assistant."
            if context:
                system_content += " Answer the question based on this context:\n\n" + context

            # Build message list
            messages = [{"role": "system", "content": system_content}]

            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            # Add the current prompt
            messages.append({"role": "user", "content": prompt})

            logger.info(f"Generating response for prompt: {prompt[:50]}...")

            # Call Ollama API
            response = requests.post(
                f"{self.api_base}/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            )

            if response.status_code == 200:
                answer = response.json().get("message", {}).get("content", "")
                logger.info(f"Generated response with {len(answer)} characters")
                return answer
            else:
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def generate_with_sources(self, query, retrieved_documents):
        """
        Generate a response based on retrieved documents with source citations

        Args:
            query (str): The user's query
            retrieved_documents (list): List of documents from vector search

        Returns:
            dict: Response with answer and sources
        """
        try:
            if not retrieved_documents or len(retrieved_documents) == 0:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }

            # Extract the content from documents
            contexts = []
            sources = []

            for i, doc in enumerate(retrieved_documents):
                payload = doc.payload
                content = payload.get("text", "No text available")
                page_num = payload.get("page_num", "Unknown")
                filename = payload.get("filename", "Unknown document")

                context_text = f"Document {i+1} (Page {page_num} from {filename}):\n{content}\n\n"
                contexts.append(context_text)

                sources.append({
                    "page": page_num,
                    "filename": filename,
                    "text": content[:200] + ("..." if len(content) > 200 else "")
                })

            # Combine contexts
            combined_context = "\n".join(contexts)

            # Create system prompt
            system_content = f"""You are a helpful assistant. Answer the question based on these documents:

{combined_context}

If the documents don't contain the information needed to answer the question, say "I couldn't find relevant information to answer your question."
Cite the page numbers and document names when providing information."""

            # Create messages
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]

            logger.info(f"Generating response with sources for query: {query}")

            # Call Ollama API
            response = requests.post(
                f"{self.api_base}/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.3,  # Lower temperature for more factual answers
                    "max_tokens": 1024
                }
            )

            if response.status_code == 200:
                answer = response.json().get("message", {}).get("content", "")
                logger.info(f"Generated response with {len(answer)} characters and {len(sources)} sources")

                return {
                    "answer": answer,
                    "sources": sources
                }
            else:
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                logger.error(error_msg)

                return {
                    "answer": error_msg,
                    "sources": []
                }

        except Exception as e:
            logger.error(f"Error generating response with sources: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": []
            }