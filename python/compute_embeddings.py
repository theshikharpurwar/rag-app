# FILE: python/compute_embeddings.py
# (Includes ALL fixes: Qdrant check, 're' import, UUID Point IDs, stores pdf_id)

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
import re    # <--- FIX: Import 're' module
import uuid  # <--- FIX: Import 'uuid' module for generating valid IDs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_SIZE = 384
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
DEFAULT_COLLECTION = 'documents'
IMAGE_SAVE_DIR_RELATIVE = "images"
RENDERING_DPI = 150
# --- End Configuration ---

class SimpleEmbedder:
    """Simple embedder that uses Sentence Transformers"""
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        logger.info(f"Initializing embedder with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            test_embedding = self.model.encode("test")
            actual_size = len(test_embedding)
            if actual_size != VECTOR_SIZE:
                 logger.warning(f"Model {model_name} output dimension ({actual_size}) does not match configured VECTOR_SIZE ({VECTOR_SIZE}).")
            logger.info(f"Embedding model {model_name} loaded successfully (Dim: {actual_size}).")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise ImportError(f"Could not load embedding model {model_name}") from e

    def get_embedding(self, content, content_type="text"):
        try:
            if content_type.lower() == "text":
                if not content or not content.strip():
                    logger.warning("Empty text content provided, returning zero vector")
                    return [0.0] * VECTOR_SIZE
                embedding = self.model.encode(content)
                return embedding.tolist()
            elif content_type.lower() == "image":
                if content is None:
                    logger.warning("None image content provided, returning zero vector")
                    return [0.0] * VECTOR_SIZE
                try:
                    # Attempt direct encode (will likely fail for text models but keeps original logic flow)
                    embedding = self.model.encode(content)
                    return embedding.tolist()
                except Exception as img_embed_err:
                    logger.warning(f"Failed to directly embed image with {self.model_name}: {img_embed_err}. Using placeholder text.")
                    placeholder_text = "image content"
                    embedding = self.model.encode(placeholder_text)
                    return embedding.tolist()
            else:
                logger.error(f"Unsupported content type: {content_type}")
                return [0.0] * VECTOR_SIZE
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * VECTOR_SIZE

# Using process_pdf function name, includes pdf_id argument
def process_pdf(pdf_path, pdf_id, collection_name=DEFAULT_COLLECTION):
    """Process PDF, extract text & images, compute embeddings, store in Qdrant with pdf_id."""
    if not pdf_id:
        logger.error("Missing pdf_id for processing.")
        return {"success": False, "error": "PDF ID not provided to embedding script."}

    try:
        logger.info(f"Processing PDF: {pdf_path} (ID: {pdf_id}) into collection: {collection_name}")
        embedder = SimpleEmbedder()
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

        # --- CORRECTED Qdrant Collection Check (Includes vector size fix) ---
        try:
            logger.info(f"Checking collection '{collection_name}'...")
            collections = client.get_collections().collections
            collection_info = next((c for c in collections if c.name == collection_name), None)

            if collection_info:
                full_collection_info = client.get_collection(collection_name=collection_name)
                try:
                    existing_size = -1 # Default to invalid size
                    if hasattr(full_collection_info, 'config') and hasattr(full_collection_info.config, 'params') and hasattr(full_collection_info.config.params, 'vectors'):
                        vectors_config = full_collection_info.config.params.vectors
                        # Handle both single default vector config and named vector dict
                        if isinstance(vectors_config, models.VectorParams):
                             existing_size = vectors_config.size
                        elif isinstance(vectors_config, dict):
                             # Assuming default unnamed vector params if it's a dict
                             if '' in vectors_config and isinstance(vectors_config[''], models.VectorParams):
                                  existing_size = vectors_config[''].size
                             elif models.DEFAULT_VECTOR_NAME in vectors_config and isinstance(vectors_config[models.DEFAULT_VECTOR_NAME], models.VectorParams):
                                  existing_size = vectors_config[models.DEFAULT_VECTOR_NAME].size
                             else: # Check if any key holds VectorParams (for older/custom named vectors)
                                 for key in vectors_config:
                                     if isinstance(vectors_config[key], models.VectorParams):
                                         existing_size = vectors_config[key].size
                                         logger.info(f"Using size from named vector config '{key}'")
                                         break
                        if existing_size == -1: logger.warning(f"Could not determine vector size format. Recreating.")
                    else: logger.warning(f"Could not find vector config structure. Recreating.")

                    if existing_size != -1 and existing_size != VECTOR_SIZE:
                        logger.warning(f"Collection '{collection_name}' size mismatch ({existing_size}!={VECTOR_SIZE}). Recreating.")
                        client.delete_collection(collection_name=collection_name, timeout=60)
                        collection_info = None # Force recreation
                    elif existing_size == VECTOR_SIZE:
                        logger.info(f"Collection '{collection_name}' exists with correct vector size {VECTOR_SIZE}.")
                    # else existing_size remained -1, handled below by collection_info being None

                except AttributeError as ae:
                    logger.warning(f"Could not access vector config attributes for '{collection_name}': {ae}. Recreating.")
                    client.delete_collection(collection_name=collection_name, timeout=60)
                    collection_info = None
            # Create if it doesn't exist or was deleted
            if not collection_info:
                logger.info(f"Creating collection: {collection_name} size {VECTOR_SIZE}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
                    timeout=60
                )
        except Exception as e:
            logger.error(f"Error setting up Qdrant collection: {e}", exc_info=True)
            return {"success": False, "error": f"Qdrant collection setup failed: {e}"}
        # --- END CORRECTED Qdrant Check ---

        document = fitz.open(pdf_path)
        num_pages = len(document)
        logger.info(f"PDF has {num_pages} pages")

        points_to_upsert = []
        embeddings_count = 0
        pdf_base_name = os.path.basename(pdf_path)
        pdf_dir = os.path.dirname(pdf_path)
        image_output_dir = os.path.join(pdf_dir, IMAGE_SAVE_DIR_RELATIVE)
        os.makedirs(image_output_dir, exist_ok=True)
        logger.info(f"Image output directory: {image_output_dir}")

        # Process each page
        for page_num, page in enumerate(document):
            # Process Text
            page_text = page.get_text("text").strip()
            if page_text:
                try:
                    text_embedding = embedder.get_embedding(page_text, "text")
                    if text_embedding != [0.0] * VECTOR_SIZE:
                        payload = {
                            "pdf_id": pdf_id, # Store the PDF ID
                            "source": pdf_base_name,
                            "page": page_num + 1,
                            "text": page_text,
                            "type": "text"
                        }
                        # Use UUID for point ID
                        point_id = str(uuid.uuid4())
                        points_to_upsert.append( models.PointStruct(id=point_id, vector=text_embedding, payload=payload) )
                        embeddings_count += 1
                    else: logger.warning(f"Failed text embed page {page_num+1}")
                except Exception as text_err: logger.error(f"Error text page {page_num+1}: {text_err}")

            # Process Images
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                try:
                    img_index_in_doc = img_info[0]
                    base_image = document.extract_image(img_index_in_doc)
                    if not base_image: continue
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image = Image.open(io.BytesIO(image_bytes))
                    image_embedding = embedder.get_embedding(image, "image")

                    if image_embedding != [0.0] * VECTOR_SIZE:
                        # Use 're' module correctly for safe filename
                        safe_pdf_base = re.sub(r'[^\w\-_\.]', '_', os.path.splitext(pdf_base_name)[0])
                        image_filename = f"{safe_pdf_base}_page_{page_num + 1}_img_{img_index + 1}.png"
                        image_save_path = os.path.join(image_output_dir, image_filename)
                        image.convert("RGB").save(image_save_path, "PNG")

                        payload = {
                            "pdf_id": pdf_id, # Store the PDF ID
                            "source": pdf_base_name,
                            "page": page_num + 1,
                            "image_path": image_save_path, # Store relative path maybe? Needs careful handling on retrieval
                            "type": "image"
                        }
                        # Use UUID for point ID
                        point_id = str(uuid.uuid4())
                        points_to_upsert.append( models.PointStruct(id=point_id, vector=image_embedding, payload=payload) )
                        embeddings_count += 1
                    else: logger.warning(f"Failed image embed page {page_num+1} img {img_index+1}")
                except Exception as img_err: logger.error(f"Error image page {page_num+1} img {img_index+1}: {img_err}", exc_info=False)

        try: document.close()
        except Exception as close_err: logger.error(f"Error closing PDF: {close_err}")

        # Batch upsert
        if points_to_upsert:
            logger.info(f"Upserting {len(points_to_upsert)} points for PDF {pdf_id}...")
            try:
                batch_size = 100
                for i in range(0, len(points_to_upsert), batch_size):
                    batch = points_to_upsert[i:i+batch_size]
                    client.upsert(collection_name=collection_name, points=batch, wait=True)
                logger.info(f"Upsert successful for {len(points_to_upsert)} points (PDF ID: {pdf_id}).")
            except Exception as e:
                 logger.error(f"Qdrant upsert failed for PDF {pdf_id}: {e}", exc_info=True)
                 error_detail = str(e)
                 if hasattr(e, 'http_body'): error_detail = getattr(e, 'http_body', str(e)) # Get specific Qdrant error if available
                 return {"success": False, "error": f"Qdrant upsert failed: {error_detail}"}
        else: logger.warning("No text or image content found/embedded.")

        result = {"success": True, "filename": pdf_base_name, "page_count": num_pages, "embeddings_count": embeddings_count, "collection": collection_name}
        logger.info(f"Successfully processed PDF: {pdf_base_name} (ID: {pdf_id})")
        return result

    except Exception as e:
        logger.error(f"Critical Error processing PDF {pdf_path} (ID: {pdf_id}): {e}", exc_info=True)
        if 'document' in locals() and document and not document.is_closed:
             try: document.close()
             except: pass
        return {"success": False, "error": f"General error: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description='Compute embeddings for PDF (Text & Image w/ Text Model), store in Qdrant.')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--pdf_id', required=True, help='MongoDB ID of the PDF document')
    parser.add_argument('--collection_name', default=DEFAULT_COLLECTION, help='Name of the Qdrant collection')
    args = parser.parse_args()

    result = process_pdf(args.pdf_path, args.pdf_id, args.collection_name)
    print(json.dumps(result)) # Output result as JSON for backend

if __name__ == "__main__":
    main()