# D:\rag-app\python\embeddings\local_embed.py

import logging
import numpy as np
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalEmbedder:
    """
    Class for generating embeddings using local sentence-transformers models.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", **kwargs):
        """
        Initialize the local embedder

        Args:
            model_name (str): Name of the sentence-transformers model
        """
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Local embedding model loaded successfully")

    def get_embedding(self, input_data, input_type="text"):
        """
        Generate embedding for text or images

        Args:
            input_data: Text string or image (PIL.Image or path)
            input_type (str): Type of input ("text" or "image")

        Returns:
            list: Embedding vector
        """
        try:
            if input_type == "text":
                return self._get_text_embedding(input_data)
            elif input_type == "image":
                return self._get_image_embedding(input_data)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _get_text_embedding(self, text):
        """
        Generate embedding for text

        Args:
            text (str): Text to embed

        Returns:
            list: Embedding vector
        """
        if not text or text.strip() == "":
            logger.warning("Empty text provided for embedding, using placeholder")
            text = "empty document"

        logger.info(f"Generating text embedding with length: {len(text)}")

        try:
            # Generate embedding
            embedding = self.model.encode(text)

            # Convert to list if it's a numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            # Return a zero vector as fallback (adjust dimension as needed)
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2

    def _get_image_embedding(self, image_input):
        """
        Generate embedding for image by describing it as text
        since sentence-transformers doesn't directly support images.

        Args:
            image_input: PIL.Image object or path to image file

        Returns:
            list: Embedding vector
        """
        # For images, we'll create a simple text description
        # and generate an embedding for that text
        try:
            if isinstance(image_input, str):
                # Image path provided
                logger.info(f"Loading image from path: {image_input}")
                with Image.open(image_input) as img:
                    width, height = img.size
                    image_desc = f"An image with dimensions {width}x{height}"
            elif isinstance(image_input, Image.Image):
                # PIL Image object provided
                width, height = image_input.size
                image_desc = f"An image with dimensions {width}x{height}"
            else:
                image_desc = "An unknown image"

            logger.info(f"Using text description for image: {image_desc}")
            return self._get_text_embedding(image_desc)

        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2