# D:\rag-app\python\embeddings\clip_embed.py

import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClipEmbedder:
    """Class for generating embeddings using CLIP model"""

    def __init__(self, model_path="openai/clip-vit-base-patch32", device=None):
        """
        Initialize the CLIP embedder

        Args:
            model_path: Path or identifier for the CLIP model
            device: Device to run the model on (None for auto-detection)
        """
        self.model_path = model_path

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading CLIP model from {model_path} on {self.device}")

        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)

        logger.info(f"CLIP model loaded successfully")

    def embed_image(self, image):
        """
        Generate embedding for an image

        Args:
            image: PIL Image or path to image

        Returns:
            numpy array containing the embedding
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize embedding
        image_embedding = image_features / image_features.norm(dim=1, keepdim=True)

        # Convert to numpy array
        return image_embedding.cpu().numpy()[0]

    def embed_text(self, text):
        """
        Generate embedding for text

        Args:
            text: Text string

        Returns:
            numpy array containing the embedding
        """
        # Process text
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)

        # Generate embedding
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize embedding
        text_embedding = text_features / text_features.norm(dim=1, keepdim=True)

        # Convert to numpy array
        return text_embedding.cpu().numpy()[0]