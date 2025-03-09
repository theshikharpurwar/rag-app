# D:\rag-app\python\embeddings\local_embed.py

import logging
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalEmbedder:
    """
    Class to generate embeddings using local sentence-transformers models
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the LocalEmbedder with a sentence-transformers model
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        logger.info(f"Initializing LocalEmbedder with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Successfully loaded model {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            # Fall back to a smaller model if loading fails
            fallback_model = 'all-MiniLM-L6-v2'
            logger.info(f"Falling back to model: {fallback_model}")
            self.model = SentenceTransformer(fallback_model)
            self.model_name = fallback_model
    
    def get_embedding(self, content, content_type="text"):
        """
        Generate embedding for the provided content
        
        Args:
            content (str or PIL.Image.Image): The content to embed
            content_type (str): Type of content ('text' or 'image')
            
        Returns:
            list: The generated embedding as a list of floats
        """
        try:
            if content_type.lower() == "text":
                if not content or content.strip() == "":
                    logger.warning("Empty text content provided, returning zero vector")
                    # Return zero vector of the same dimension as the model's output
                    return [0.0] * self.model.get_sentence_embedding_dimension()
                
                # Generate text embedding
                logger.info(f"Generating embedding for text: {content[:50]}...")
                embedding = self.model.encode(content)
                
            elif content_type.lower() == "image":
                if content is None:
                    logger.warning("None image content provided, returning zero vector")
                    return [0.0] * self.model.get_sentence_embedding_dimension()
                
                # Generate image embedding
                logger.info("Generating embedding for image...")
                if isinstance(content, str):
                    # If content is a file path
                    try:
                        image = Image.open(content).convert('RGB')
                        embedding = self.model.encode(image)
                    except Exception as e:
                        logger.error(f"Error opening image from path {content}: {str(e)}")
                        return [0.0] * self.model.get_sentence_embedding_dimension()
                else:
                    # If content is already a PIL image
                    embedding = self.model.encode(content)
            else:
                logger.error(f"Unsupported content type: {content_type}")
                return [0.0] * self.model.get_sentence_embedding_dimension()
            
            # Convert to list and normalize if needed
            embedding_list = embedding.tolist()
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector in case of error
            return [0.0] * self.model.get_sentence_embedding_dimension()