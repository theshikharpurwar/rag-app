# D:\rag-app\python\embeddings\clip_embed.py

import logging
import torch
import clip
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClipEmbedder:
    """Class for generating CLIP embeddings for text and images."""

    def __init__(self, model_path="ViT-B/32"):
        """Initialize the CLIP embedder with the specified model."""
        logger.info(f"Initializing CLIP embedder with model: {model_path}")

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using CLIP model: {model_path}")

        # Load CLIP model
        self.model, self.preprocess = clip.load(model_path, device=self.device)
        logger.info(f"CLIP model loaded on {self.device}")

    def get_embedding(self, input_data):
        """
        Generate embeddings for text or image input.

        Args:
            input_data: Either a string (for text) or PIL Image object (for images)

        Returns:
            Embedding vector as numpy array
        """
        try:
            with torch.no_grad():
                if isinstance(input_data, str):
                    # Generate text embedding
                    text_inputs = clip.tokenize([input_data]).to(self.device)
                    text_features = self.model.encode_text(text_inputs)
                    embedding = text_features[0].cpu().numpy()
                elif isinstance(input_data, Image.Image):
                    # Generate image embedding
                    image = self.preprocess(input_data).unsqueeze(0).to(self.device)
                    image_features = self.model.encode_image(image)
                    embedding = image_features[0].cpu().numpy()
                else:
                    raise ValueError("Input must be either a string or PIL Image")

                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise