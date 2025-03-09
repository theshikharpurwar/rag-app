# D:\rag-app\python\compute_embeddings.py

import os
import sys
import json
import argparse
import logging
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleEmbedder:
    """Simple embedder that uses Sentence Transformers"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logger.info(f"Initializing embedder with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def get_embedding(self, content, content_type="text"):
        try:
            if content_type.lower() == "text":
                if not content or content.strip() == "":
                    logger.warning("Empty text content provided, returning zero vector")
                    return [0.0] * 384  # Default size for all-MiniLM-L6-v2
                
                logger.info(f"Generating embedding for text: {content[:50]}...")
                embedding = self.model.encode(content)
                return embedding.tolist()
                
            elif content_type.lower() == "image":
                if content is None:
                    logger.warning("None image content provided, returning zero vector")
                    return [0.0] * 384
                
                logger.info("Generating embedding for image...")
                if isinstance(content, str):
                    # If content is a file path
                    try:
                        image = Image.open(content).convert('RGB')
                        embedding = self.model.encode(image)
                        return embedding.tolist()
                    except Exception as e:
                        logger.error(f"Error opening image from path {content}: {str(e)}")
                        return [0.0] * 384
                else:
                    # If content is already a PIL image
                    embedding = self.model.encode(content)
                    return embedding.tolist()
                    
            else:
                logger.error(f"Unsupported content type: {content_type}")
                return [0.0] * 384
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384

def process_pdf(pdf_path, collection_name='documents'):
    """
    Process a PDF file, extract text and images, compute embeddings, and store in Qdrant

    Args:
        pdf_path (str): Path to the PDF file
        collection_name (str): Name of the Qdrant collection

    Returns:
        dict: Processing results with success flag and details
    """
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"Using collection: {collection_name}")

        # Get the embedder (hardcoded to all-MiniLM-L6-v2)
        embedder = SimpleEmbedder()

        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)

        # Ensure collection exists
        try:
            collections = client.get_collections().collections
            collection_exists = any(collection.name == collection_name for collection in collections)

            if not collection_exists:
                logger.info(f"Creating collection: {collection_name}")
                # Vector size is 384 for all-MiniLM-L6-v2
                vector_size = 384

                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {collection_name} with vector size {vector_size}")
            else:
                logger.info(f"Collection {collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            return {"success": False, "error": f"Error ensuring collection exists: {str(e)}"}

        # Open PDF document
        document = fitz.open(pdf_path)
        num_pages = len(document)
        logger.info(f"PDF has {num_pages} pages")

        # Store embeddings
        stored_count = 0

        # Process each page
        for page_num, page in enumerate(document):
            # Get page text
            page_text = page.get_text()

            if page_text.strip():
                # Compute text embedding
                text_embedding = embedder.get_embedding(page_text, "text")

                # Store text and embedding in Qdrant
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=page_num,
                            vector=text_embedding,
                            payload={
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "text": page_text,
                                "type": "text"
                            }
                        )
                    ]
                )
                stored_count += 1
                logger.info(f"Stored text embedding for page {page_num+1}")

            # Extract images from the page
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                img_index_in_doc = img_info[0]
                base_image = document.extract_image(img_index_in_doc)
                image_bytes = base_image["image"]

                try:
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))

                    # Compute image embedding
                    image_embedding = embedder.get_embedding(image, "image")

                    # Generate a unique ID for the image
                    image_id = int(f"{page_num+1}{img_index+1}")

                    # Save image to file
                    image_dir = os.path.join(os.path.dirname(pdf_path), "images")
                    os.makedirs(image_dir, exist_ok=True)
                    image_filename = os.path.join(image_dir, f"{os.path.basename(pdf_path)}_{page_num+1}_{img_index+1}.png")
                    image.save(image_filename)

                    # Store image embedding in Qdrant
                    client.upsert(
                        collection_name=collection_name,
                        points=[
                            models.PointStruct(
                                id=image_id,
                                vector=image_embedding,
                                payload={
                                    "source": os.path.basename(pdf_path),
                                    "page": page_num + 1,
                                    "image_path": image_filename,
                                    "type": "image"
                                }
                            )
                        ]
                    )
                    stored_count += 1
                    logger.info(f"Stored image embedding for page {page_num+1}, image {img_index+1}")

                except Exception as e:
                    logger.error(f"Error processing image {img_index+1} on page {page_num+1}: {str(e)}")

        document.close()

        result = {
            "success": True,
            "filename": os.path.basename(pdf_path),
            "page_count": num_pages,
            "embeddings_count": stored_count,
            "collection": collection_name
        }

        logger.info(f"Successfully processed PDF with {stored_count} embeddings")
        return result

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Process PDF and compute embeddings')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--collection_name', default='documents', help='Name of the Qdrant collection')

    args = parser.parse_args()

    result = process_pdf(args.pdf_path, args.collection_name)
    print(json.dumps(result))

if __name__ == "__main__":
    main()