# python/embeddings/colpali_embed.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ColpaliEmbedder:
    def __init__(self, model_path, **kwargs):
        """
        Initialize the embedder with a CLIP model

        Args:
            model_path (str): Path or identifier for the model
            **kwargs: Additional parameters for the model
        """
        # Use a standard CLIP model instead of PaliGemma
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        # Print info about the model being used
        print(f"Using CLIP model: {self.model_name} instead of {model_path}")

    def embed_image(self, image):
        """
        Compute embedding for an image

        Args:
            image (PIL.Image): Image to embed

        Returns:
            numpy.ndarray: Embedding vector
        """
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")

        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize the features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Convert to numpy array
        embedding = image_features.squeeze().cpu().numpy()

        return embedding

    def embed_text(self, text):
        """
        Compute embedding for text

        Args:
            text (str): Text to embed

        Returns:
            numpy.ndarray: Embedding vector
        """
        # Process the text
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)

        # Get text features
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize the features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Convert to numpy array
        embedding = text_features.squeeze().cpu().numpy()

        return embedding