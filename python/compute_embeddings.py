#!/usr/bin/env python
# compute_embeddings.py

import os
import sys
import json
import logging
import numpy as np
from PIL import Image
import torch
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import embedding module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.embeddings.embed_factory import get_embedder

def convert_pdf_to_images(pdf_path, output_dir):
    """Convert PDF to images using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        image_paths = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            image_path = os.path.join(output_dir, f"page_{page_num+1}.png")
            pix.save(image_path)
            image_paths.append(image_path)
        
        return image_paths, len(doc)
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise

def compute_embeddings(image_paths, model_name, model_path, model_params=None):
    """Compute embeddings for images using the specified model"""
    try:
        if model_params is None:
            model_params = {}
        
        # Get embedder based on model name
        embedder = get_embedder(model_name, model_path, **model_params)
        
        # Compute embeddings for each image
        embeddings = []
        for img_path in image_paths:
            img = Image.open(img_path)
            embedding = embedder.embed_image(img)
            embeddings.append(embedding)
        
        return embeddings
    except Exception as e:
        logger.error(f"Error computing embeddings: {str(e)}")
        raise

def store_embeddings(embeddings, collection_name, pdf_name, image_paths):
    """Store embeddings in Qdrant"""
    try:
        # Connect to Qdrant
        client = QdrantClient(url="http://localhost:6333")
        
        # Prepare points for insertion
        points = []
        for i, (embedding, img_path) in enumerate(zip(embeddings, image_paths)):
            # Convert embedding to list if it's a numpy array or tensor
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy().tolist()
            
            points.append(models.PointStruct(
                id=f"{pdf_name}_{i}",
                vector=embedding,
                payload={
                    "pdf_name": pdf_name,
                    "page_num": i + 1,
                    "image_path": img_path
                }
            ))
        
        # Insert points into collection
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Stored {len(points)} embeddings in collection {collection_name}")
        return len(points)
    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}")
        raise

def main():
    """Main function to process PDF and compute embeddings"""
    if len(sys.argv) < 4:
        logger.error("Usage: python compute_embeddings.py <pdf_path> <collection_name> <output_dir> [model_name] [model_path] [model_params_json]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    collection_name = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Default model settings
    model_name = "colpali"
    model_path = "vidore/colpali-v1.2"
    model_params = {}
    
    # Override with command line arguments if provided
    if len(sys.argv) > 4:
        model_name = sys.argv[4]
    if len(sys.argv) > 5:
        model_path = sys.argv[5]
    if len(sys.argv) > 6:
        try:
            model_params = json.loads(sys.argv[6])
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON for model parameters: {sys.argv[6]}")
    
    try:
        # Process PDF
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images
        image_paths, page_count = convert_pdf_to_images(pdf_path, output_dir)
        logger.info(f"Converted PDF to {len(image_paths)} images")
        
        # Compute embeddings
        embeddings = compute_embeddings(image_paths, model_name, model_path, model_params)
        logger.info(f"Computed {len(embeddings)} embeddings")
        
        # Store embeddings in Qdrant
        pdf_name = os.path.basename(pdf_path)
        stored_count = store_embeddings(embeddings, collection_name, pdf_name, image_paths)
        
        # Return result as JSON
        result = {
            "success": True,
            "pdf_name": pdf_name,
            "pageCount": page_count,
            "embeddingCount": len(embeddings),
            "storedCount": stored_count
        }
        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(result))
        sys.exit(1)

if __name__ == "__main__":
    main()