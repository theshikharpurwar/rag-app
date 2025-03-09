# D:\rag-app\python\embeddings\local_embed.py

import logging
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from torchvision import transforms

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

    def get_embedding(self, content, content_type='text'):
        """Generate embeddings for text or images."""
        try:
            if content_type == 'text':
                # Get text embedding
                embedding = self.model.encode([content])[0]
                return embedding.tolist()
            elif content_type == 'image':
                # For images, need to convert PIL Image to RGB mode if needed
                if hasattr(content, 'convert'):  # Check if it's a PIL Image
                    # Convert to RGB mode if it's not already
                    if content.mode != 'RGB':
                        content = content.convert('RGB')

                    # Simple approach: use a text description of the image
                    # Most sentence transformer models aren't designed for images
                    dummy_text = "image content placeholder"
                    embedding = self.model.encode([dummy_text])[0]
                    return embedding.tolist()
                else:
                    raise ValueError(f"Unsupported image type: {type(content)}")
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector of the correct dimension as fallback
            return [0.0] * 384  # assuming 384-dimension embeddings