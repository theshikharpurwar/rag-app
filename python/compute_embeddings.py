# python/compute_embeddings.py
import sys
import os
import json
import logging
from pdf2image import convert_from_path
from PIL import Image
from embeddings.embed_factory import EmbeddingModelFactory
from qdrant_client import QdrantClient, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf(pdf_path, collection_name, images_dir, model_name, model_path, model_params=None):
    """Process a PDF file, convert to images, generate embeddings, and store in Qdrant"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create directory for storing images
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)

    pdf_id = os.path.basename(pdf_path)
    logger.info(f"Processing PDF: {pdf_path}")

    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=200)
        logger.info(f"Converted PDF to {len(images)} images")
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        if "poppler" in str(e).lower():
            logger.error("Error: Poppler not found. Make sure poppler is installed and in PATH")
        raise

    # Load embedding model using factory
    if model_params:
        if isinstance(model_params, str):
            model_params = json.loads(model_params)
    else:
        model_params = {}

    try:
        embedder = EmbeddingModelFactory.create_model(model_name, model_path, model_params)
        logger.info(f"Embedding model {model_name} loaded successfully")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise

    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")
    vector_dim = embedder.get_embedding_dimension()

    # Ensure collection exists with correct dimensions
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection {collection_name} with dimension {vector_dim}")
    except Exception as e:
        logger.error(f"Error creating Qdrant collection: {e}")
        raise

    # Process each page
    points = []
    saved_images = []

    for i, image in enumerate(images):
        try:
            # Save image
            image_filename = f"page_{i}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            image.save(image_path, "JPEG")
            saved_images.append(image_path)
            logger.info(f"Saved image to {image_path}")

            # Generate embedding
            embedding = embedder.get_image_embedding(image)

            # Create Qdrant point
            point = models.PointStruct(
                id=f"{os.path.basename(images_dir)}_{i}",
                vector=embedding,
                payload={
                    "pdf_id": pdf_id,
                    "page_num": i,
                    "image_path": image_path
                }
            )
            points.append(point)
            logger.info(f"Generated embedding for page {i}")
        except Exception as e:
            logger.error(f"Error processing page {i}: {e}")

    # Store embeddings in Qdrant
    if points:
        try:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Stored {len(points)} embeddings in Qdrant")
        except Exception as e:
            logger.error(f"Error storing embeddings in Qdrant: {e}")
            raise
    else:
        raise Exception("No embeddings generated")

    # Return results
    return {
        "message": "PDF processed successfully",
        "pageCount": len(images),
        "savedImages": saved_images
    }

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(json.dumps({
            "error": "Usage: compute_embeddings.py <pdf_path> <collection_name> <images_dir> <model_name> <model_path> [<model_params>]"
        }))
        sys.exit(1)

    pdf_path = sys.argv[1]
    collection_name = sys.argv[2]
    images_dir = sys.argv[3]
    model_name = sys.argv[4]
    model_path = sys.argv[5]
    model_params = sys.argv[6] if len(sys.argv) > 6 else "{}"

    try:
        result = process_pdf(pdf_path, collection_name, images_dir, model_name, model_path, model_params)
        print(json.dumps(result))
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)