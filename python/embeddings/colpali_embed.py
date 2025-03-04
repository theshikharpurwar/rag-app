# D:\rag-app\python\embeddings\colpali_embed.py

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColpaliEmbedder:
    def __init__(self, model_path="vidore/colpali-v1.2", device=None, **kwargs):
        self.model_path = model_path

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Colpali model from {model_path} on {self.device}")

        # Load model and processor
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)

        logger.info(f"Colpali model loaded successfully")

    def embed_image(self, image):
        """
        Generate embedding for an image

        Args:
            image: PIL Image or path to image file

        Returns:
            numpy array: Image embedding
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Generate embedding
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                image_features = outputs.last_hidden_state.mean(dim=1)  # Pool the output

            # Normalize embedding
            image_embedding = image_features / image_features.norm(dim=1, keepdim=True)

            # Convert to numpy array
            return image_embedding.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error embedding image: {str(e)}")
            raise

    def embed_text(self, text):
        """
        Generate embedding for text

        Args:
            text: Text string

        Returns:
            numpy array: Text embedding
        """
        try:
            # Process text
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)

            # Generate embedding
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                text_features = outputs.last_hidden_state.mean(dim=1)  # Pool the output

            # Normalize embedding
            text_embedding = text_features / text_features.norm(dim=1, keepdim=True)

            # Convert to numpy array
            return text_embedding.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise