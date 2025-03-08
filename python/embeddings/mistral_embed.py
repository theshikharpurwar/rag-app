# D:\rag-app\python\embeddings\mistral_embed.py

import os
import logging
import base64
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
from mistralai.client import MistralClient
from mistralai.models.embeddings import EmbeddingRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MistralEmbedder:
    """Mistral AI implementation for generating embeddings"""

    def __init__(self, model_name="mistral-embed", api_key=None):
        """
        Initialize the Mistral embedder

        Args:
            model_name (str): Mistral embedding model name
            api_key (str): Mistral API key
        """
        self.model_name = model_name
        self.api_key = api_key

        if not self.api_key:
            raise ValueError("Mistral API key not provided")

        logger.info(f"Initializing Mistral embedder with model: {model_name}")
        self.client = MistralClient(api_key=self.api_key)

    def get_embedding(self, input_data, input_type="text"):
        """
        Generate embeddings for text or image using Mistral AI

        Args:
            input_data: Text string or image path/bytes
            input_type (str): 'text' or 'image'

        Returns:
            np.array: Embedding vector
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
        """Generate embedding for text"""
        if not text or text.strip() == "":
            logger.warning("Empty text provided, returning zero vector")
            # Return a zero vector with the expected dimension
            return np.zeros(1024)  # Mistral embeddings are typically 1024D

        logger.info(f"Generating text embedding with Mistral (length: {len(text)})")

        # Get embedding from Mistral API
        response = self.client.embeddings(
            model=self.model_name,
            input=text
        )

        # Extract the embedding vector
        embedding = np.array(response.data[0].embedding)
        return embedding

    def _get_image_embedding(self, image_input):
        """Generate embedding for image"""
        # Handle different image input types
        if isinstance(image_input, str):
            # Input is a file path
            if os.path.isfile(image_input):
                logger.info(f"Loading image from path: {image_input}")
                if image_input.lower().endswith('.pdf'):
                    image_bytes = self._extract_image_from_pdf(image_input)
                else:
                    with open(image_input, "rb") as img_file:
                        image_bytes = img_file.read()
            else:
                raise FileNotFoundError(f"Image file not found: {image_input}")
        elif isinstance(image_input, bytes):
            # Input is already bytes
            image_bytes = image_input
        elif isinstance(image_input, Image.Image):
            # Input is PIL Image
            buffer = BytesIO()
            image_input.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        logger.info("Generating image embedding with Mistral")

        # Get embedding from Mistral API
        # For image embeddings with Mistral
        request = EmbeddingRequest(
            model=self.model_name,
            input=[{"type": "image", "data": base64_image}]
        )

        response = self.client.embeddings(request=request)

        # Extract the embedding vector
        embedding = np.array(response.data[0].embedding)
        return embedding

    def _extract_image_from_pdf(self, pdf_path, page_num=0):
        """Extract image from the first page of a PDF"""
        try:
            logger.info(f"Extracting image from PDF: {pdf_path}, page {page_num}")
            doc = fitz.open(pdf_path)

            if page_num >= len(doc):
                logger.warning(f"Page {page_num} not found in PDF, using first page")
                page_num = 0

            page = doc[page_num]

            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))

            # Convert to PIL Image and then to bytes
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")

            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error extracting image from PDF: {str(e)}")
            raise