# Local Nomic embedding support for multimodal RAG (no API required)

import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
import os
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class NomicEmbedder:
    """
    Local embedder for Nomic models supporting both text and vision embeddings
    Loads models locally from Hugging Face without requiring API calls
    """
    
    def __init__(self, device: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize local Nomic embedder
        
        Args:
            device: Device to run models on ('cpu', 'cuda', or None for auto-detect)
            cache_dir: Directory to cache downloaded models
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing Nomic embedder on device: {self.device}")
        
        # Model names
        self.text_model_name = "nomic-ai/nomic-embed-text-v1.5"
        self.vision_model_name = "nomic-ai/nomic-embed-vision-v1.5"
        
        # Cache directory
        self.cache_dir = cache_dir
          # Track which loading method worked for text model
        self._use_sentence_transformers_text = True
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load the Nomic models locally"""
        try:
            logger.info("Loading Nomic text model locally...")
            # Load text model using sentence-transformers for better compatibility
            self.text_model = SentenceTransformer(
                self.text_model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            
            # Set max sequence length if not set
            if hasattr(self.text_model, 'max_seq_length'):
                if self.text_model.max_seq_length is None:
                    self.text_model.max_seq_length = 8192  # Nomic text models support long sequences
            
            logger.info(f"✓ Text model loaded: {self.text_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            # Fallback: try loading with transformers directly
            try:
                logger.info("Trying alternative text model loading...")
                self.text_tokenizer = AutoTokenizer.from_pretrained(
                    self.text_model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                self.text_model = AutoModel.from_pretrained(
                    self.text_model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                ).to(self.device)
                self.text_model.eval()
                self._use_sentence_transformers_text = False
                logger.info("✓ Text model loaded with transformers")
            except Exception as e2:
                logger.error(f"Failed to load text model with transformers: {e2}")
                raise RuntimeError(f"Could not load text model: {e2}")
        
        try:
            logger.info("Loading Nomic vision model locally...")
            # Load vision model
            self.vision_processor = AutoImageProcessor.from_pretrained(
                self.vision_model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.vision_model = AutoModel.from_pretrained(
                self.vision_model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            ).to(self.device)
            self.vision_model.eval()
            
            logger.info(f"✓ Vision model loaded: {self.vision_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise RuntimeError(f"Could not load vision model: {e}")
        
        # Test models
        self._test_models()
    
    def _test_models(self):
        """Test that models work correctly"""
        try:
            # Test text model
            test_text_embedding = self.encode_text(["test"], task_type="search_query")
            if test_text_embedding and len(test_text_embedding) > 0:
                text_dim = len(test_text_embedding[0])
                logger.info(f"✓ Text model test successful. Dimension: {text_dim}")
            else:
                raise Exception("Text model test failed")
            
            # Test vision model
            test_img = Image.new('RGB', (224, 224), color='white')
            test_vision_embedding = self.encode_images([test_img])
            if test_vision_embedding and len(test_vision_embedding) > 0:
                vision_dim = len(test_vision_embedding[0])
                logger.info(f"✓ Vision model test successful. Dimension: {vision_dim}")
                
                # Check if dimensions match (should be same for Nomic models)
                if text_dim == vision_dim:
                    logger.info(f"✓ Models share embedding space (dim: {text_dim})")
                else:
                    logger.warning(f"⚠️ Dimension mismatch: text={text_dim}, vision={vision_dim}")
            else:
                raise Exception("Vision model test failed")
                
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
            raise RuntimeError(f"Model testing failed: {e}")    
    def encode_text(self, texts: List[str], task_type: str = "search_document") -> Optional[List[List[float]]]:
        """
        Encode text using local nomic-embed-text-v1.5
        
        Args:
            texts: List of text strings to embed
            task_type: Either "search_document" or "search_query"
        
        Returns:
            List of embeddings (list of floats for each text)
        """
        if not texts:
            return []
        
        try:
            # Add the appropriate prefix based on task type
            prefixed_texts = []
            for text in texts:
                if task_type == "search_document":
                    prefixed_texts.append(f"search_document: {text}")
                elif task_type == "search_query":
                    prefixed_texts.append(f"search_query: {text}")
                else:
                    # For other tasks, use the text as-is
                    prefixed_texts.append(text)
            
            # Use sentence-transformers if available
            if hasattr(self, 'text_model') and hasattr(self.text_model, 'encode'):
                with torch.no_grad():
                    embeddings = self.text_model.encode(
                        prefixed_texts,
                        convert_to_tensor=False,
                        normalize_embeddings=True,
                        batch_size=32
                    )
                    return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            
            # Fallback: use transformers directly
            elif hasattr(self, 'text_tokenizer'):
                embeddings = []
                with torch.no_grad():
                    for text in prefixed_texts:
                        # Tokenize
                        inputs = self.text_tokenizer(
                            text,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=8192
                        ).to(self.device)
                        
                        # Get embeddings
                        outputs = self.text_model(**inputs)
                        # Use [CLS] token or mean pooling
                        if hasattr(outputs, 'last_hidden_state'):
                            embedding = outputs.last_hidden_state.mean(dim=1)
                        else:
                            embedding = outputs.pooler_output
                        
                        # Normalize
                        embedding = F.normalize(embedding, p=2, dim=1)
                        embeddings.append(embedding.cpu().numpy().flatten().tolist())
                
                return embeddings
            else:
                logger.error("No text model available")
                return None
                
        except Exception as e:
            logger.error(f"Error encoding text locally: {e}")
            return None
    
    def encode_images(self, images: List[Union[Image.Image, str]]) -> Optional[List[List[float]]]:
        """
        Encode images using local nomic-embed-vision-v1.5
        
        Args:
            images: List of PIL Images or base64 encoded image strings
        
        Returns:
            List of embeddings (list of floats for each image)
        """
        if not images:
            return []
        
        try:
            # Convert all inputs to PIL Images
            pil_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    pil_images.append(img)
                elif isinstance(img, str):
                    # Assume it's base64 encoded
                    import base64
                    import io
                    img_data = base64.b64decode(img)
                    pil_img = Image.open(io.BytesIO(img_data))
                    pil_images.append(pil_img)
                else:
                    logger.error(f"Unsupported image type: {type(img)}")
                    continue
            
            if not pil_images:
                logger.warning("No valid images to encode")
                return []
            
            # Process images and get embeddings
            embeddings = []
            with torch.no_grad():
                for img in pil_images:
                    # Ensure RGB format
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Process image
                    inputs = self.vision_processor(
                        images=img,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Get embeddings
                    outputs = self.vision_model(**inputs)
                    
                    # Extract embedding (typically from pooler_output or last_hidden_state)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embedding = outputs.pooler_output
                    elif hasattr(outputs, 'last_hidden_state'):
                        # Use mean pooling if no pooler output
                        embedding = outputs.last_hidden_state.mean(dim=1)
                    else:
                        # Fallback: use the first output
                        embedding = list(outputs.values())[0]
                        if len(embedding.shape) > 2:
                            embedding = embedding.mean(dim=1)
                    
                    # Normalize
                    embedding = F.normalize(embedding, p=2, dim=1)
                    embeddings.append(embedding.cpu().numpy().flatten().tolist())
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding images locally: {e}")
            return None
    
    def get_text_dimension(self) -> int:
        """Get the dimension of text embeddings"""
        test_embedding = self.encode_text(["test"])
        if test_embedding and len(test_embedding) > 0:
            return len(test_embedding[0])
        return 768  # Default dimension for Nomic models
    
    def get_vision_dimension(self) -> int:
        """Get the dimension of vision embeddings"""
        # Create a small test image
        test_img = Image.new('RGB', (100, 100), color='white')
        test_embedding = self.encode_images([test_img])
        if test_embedding and len(test_embedding) > 0:
            return len(test_embedding[0])
        return 768  # Default dimension for Nomic models
