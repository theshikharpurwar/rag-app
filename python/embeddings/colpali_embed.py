# python/embeddings/colpali_embed.py
import logging
import torch
from colpali_engine.models import ColPali, ColPaliProcessor

logger = logging.getLogger(__name__)

class ColPaliEmbedder:
    """ColPali-based embedding model"""

    def __init__(self, model_path, params=None):
        self.model_path = model_path
        self.params = params or {}
        self._load_model()

    def _load_model(self):
        """Load the model and processor"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading ColPali model from {self.model_path} on {device}")

            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self.model = ColPali.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True
            )
            self.processor = ColPaliProcessor.from_pretrained(self.model_path)
            logger.info("ColPali model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ColPali model: {e}")
            raise

    def get_embedding_dimension(self):
        """Return the embedding dimension"""
        return 128  # ColPali's default dimension

    def get_query_embedding(self, query):
        """Generate embedding for text query"""
        try:
            with torch.no_grad():
                query_input = self.processor.process_queries([query]).to(self.model.device)
                query_embedding = self.model(**query_input)
            return query_embedding[0].cpu().float().numpy().tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    def get_image_embedding(self, image):
        """Generate embedding for image"""
        try:
            with torch.no_grad():
                image_input = self.processor.process_images(image).to(self.model.device)
                image_embedding = self.model(**image_input)
            return image_embedding.cpu().float().numpy().tolist()
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            raise