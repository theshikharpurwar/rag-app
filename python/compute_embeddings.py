# D:\rag-app\python\compute_embeddings.py

import os
import sys
import json
import argparse
import logging
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.http import models
from embeddings.embed_factory import EmbeddingModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_collection_exists(client, collection_name, vector_size=384):
    """
    Ensure that the collection exists in Qdrant

    Args:
        client: QdrantClient instance
        collection_name (str): Name of the collection
        vector_size (int): Size of the vectors

    Returns:
        bool: True if collection exists or was created
    """
    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if not collection_exists:
            logger.info(f"Creating collection {collection_name} with vector size {vector_size}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection {collection_name} created successfully")
        else:
            logger.info(f"Collection {collection_name} already exists")

        return True
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {str(e)}")
        return False

def extract_from_pdf(pdf_path):
    """
    Extract text and images from a PDF file

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        list: List of dictionaries with page text and images
    """
    try:
        logger.info(f"Extracting content from PDF: {pdf_path}")

        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []

        # Extract the filename from the path
        filename = os.path.basename(pdf_path)

        # Open the PDF
        doc = fitz.open(pdf_path)
        logger.info(f"PDF opened successfully with {len(doc)} pages")

        pages = []

        for page_idx, page in enumerate(doc):
            # Extract text
            text = page.get_text()

            # Extract images
            image_list = []

            # Get the page images
            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Convert to PIL Image
                    image = Image.open(BytesIO(image_bytes))

                    # Add the image to the list
                    image_list.append({
                        "image": image,
                        "index": img_idx
                    })
                except Exception as e:
                    logger.error(f"Error extracting image {img_idx} from page {page_idx}: {str(e)}")

            # Add the page data
            pages.append({
                "page_num": page_idx + 1,
                "text": text,
                "images": image_list,
                "filename": filename
            })

        logger.info(f"Extracted content from {len(pages)} pages")
        return pages

    except Exception as e:
        logger.error(f"Error extracting from PDF: {str(e)}")
        return []

def store_embeddings(client, collection_name, pages, embedder):
    """
    Generate and store embeddings for PDF pages

    Args:
        client: QdrantClient instance
        collection_name (str): Name of the collection
        pages (list): List of page data
        embedder: Embedder instance

    Returns:
        int: Number of points stored
    """
    try:
        logger.info(f"Storing embeddings for {len(pages)} pages")

        points = []

        for page_idx, page in enumerate(pages):
            try:
                # Generate embedding for text
                text = page["text"]
                if text and text.strip():
                    text_embedding = embedder.get_embedding(text, input_type="text")

                    # Convert to list if it's a numpy array
                    if isinstance(text_embedding, np.ndarray):
                        text_embedding = text_embedding.tolist()

                    # Create a point for the text
                    point_id = page_idx * 100  # Text points have IDs like 0, 100, 200...

                    points.append(models.PointStruct(
                        id=point_id,
                        vector=text_embedding,
                        payload={
                            "type": "text",
                            "text": text,
                            "page_num": page["page_num"],
                            "filename": page["filename"]
                        }
                    ))

                # Generate embeddings for images
                for img_idx, img_data in enumerate(page["images"]):
                    image = img_data["image"]

                    # Skip very small images
                    if image.width < 50 or image.height < 50:
                        continue

                    # Generate embedding for the image
                    image_embedding = embedder.get_embedding(image, input_type="image")

                    # Convert to list if it's a numpy array
                    if isinstance(image_embedding, np.ndarray):
                        image_embedding = image_embedding.tolist()

                    # Create a point for the image
                    point_id = page_idx * 100 + img_idx + 1  # Image points have IDs like 1, 2, 101, 102...

                    # Convert PIL image to base64 for storage
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = f"data:image/jpeg;base64,{BytesIO(buffered.getvalue()).read().hex()}"

                    points.append(models.PointStruct(
                        id=point_id,
                        vector=image_embedding,
                        payload={
                            "type": "image",
                            "image_data": img_str,
                            "width": image.width,
                            "height": image.height,
                            "page_num": page["page_num"],
                            "filename": page["filename"]
                        }
                    ))
            except Exception as e:
                logger.error(f"Error processing page {page_idx}: {str(e)}")

        # Store points in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            logger.info(f"Stored batch of {len(batch)} points")

        logger.info(f"Successfully stored {len(points)} points")
        return len(points)

    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}")
        return 0

def process_pdf(pdf_path, collection_name="documents", model_name="all-MiniLM-L6-v2"):
    """
    Process a PDF file and store embeddings in Qdrant

    Args:
        pdf_path (str): Path to the PDF file
        collection_name (str): Name of the Qdrant collection
        model_name (str): Name of the embedding model

    Returns:
        dict: Result of processing
    """
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"Using model: {model_name}")

        try:
            # Initialize the embedder
            embedder = EmbeddingModelFactory.get_embedder(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {str(e)}")
            return {
                "success": False,
                "message": f"Error initializing embedder: {str(e)}",
                "page_count": 0,
                "points_stored": 0
            }

        # Initialize Qdrant client
        client = QdrantClient("localhost", port=6333)

        # Ensure collection exists - use 384 for all-MiniLM-L6-v2
        vector_size = 384  # Default for sentence-transformers models
        ensure_collection_exists(client, collection_name, vector_size)

        # Extract content from PDF
        pages = extract_from_pdf(pdf_path)

        if not pages:
            return {
                "success": False,
                "message": "Failed to extract content from PDF",
                "page_count": 0,
                "points_stored": 0
            }

        # Store embeddings
        points_stored = store_embeddings(client, collection_name, pages, embedder)

        return {
            "success": True,
            "message": "PDF processed successfully",
            "page_count": len(pages),
            "points_stored": points_stored
        }

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {
            "success": False,
            "message": f"Error processing PDF: {str(e)}",
            "page_count": 0,
            "points_stored": 0
        }

def main():
    try:
        parser = argparse.ArgumentParser(description="Process PDF and store embeddings in Qdrant")
        parser.add_argument("pdf_path", help="Path to the PDF file")
        parser.add_argument("--collection_name", default="documents", help="Name of the Qdrant collection")
        parser.add_argument("--model_name", default="all-MiniLM-L6-v2", help="Name of the embedding model")

        args = parser.parse_args()

        result = process_pdf(
            args.pdf_path,
            collection_name=args.collection_name,
            model_name=args.model_name
        )

        # Print the result as JSON for parsing by Node.js
        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        error_result = {
            "success": False,
            "message": f"Error: {str(e)}",
            "page_count": 0,
            "points_stored": 0
        }
        print(json.dumps(error_result, ensure_ascii=False))

if __name__ == "__main__":
    main()