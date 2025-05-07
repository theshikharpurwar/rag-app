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
BATCH_SIZE = 10  # Process in batches for performance
TEXT_CHUNK_SIZE = 500  # Characters per chunk
TEXT_CHUNK_OVERLAP = 100  # Character overlap between chunks
SKIP_IMAGES = os.environ.get("SKIP_IMAGES", "false").lower() == "true"  # Option to skip images
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
            
            # Enable batching for better performance
            self.model.max_seq_length = 256  # Limit sequence length for faster processing
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
            
    def get_embeddings_batch(self, texts):
        """Process multiple texts in a batch for better performance"""
        if not texts:
            return []
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [[0.0] * VECTOR_SIZE for _ in texts]

def chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=TEXT_CHUNK_OVERLAP):
    """
    Split text into smaller chunks with overlap for better semantic search.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # Find the end of the chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to break at a sentence or paragraph
        if end < text_len:
            # Look for paragraph breaks first (double newline)
            paragraph_break = text.find('\n\n', end - 50, end + 50)
            if paragraph_break != -1 and paragraph_break - start >= 100:
                end = paragraph_break
            else:
                # Try to find sentence breaks (period followed by space)
                sentence_break = text.rfind('. ', end - 50, end + 50)
                if sentence_break != -1 and sentence_break - start >= 100:
                    end = sentence_break + 1  # Include the period
                else:
                    # If no good break, try space
                    space = text.rfind(' ', end - 20, end + 20)
                    if space != -1:
                        end = space
        
        # Extract the chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move the start position for the next chunk, with overlap
        start = end - overlap if end < text_len else text_len
        
    return chunks

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

        # Lists for batch processing
        text_chunks = []  
        chunk_metadata = []

        # Process each page
        for page_num, page in enumerate(document):
            # Extract page text
            page_text = page.get_text("text").strip()
            if page_text:
                # Split text into smaller chunks for better retrieval
                chunks = chunk_text(page_text)
                
                # For small pages, just use the whole text
                if not chunks and page_text:
                    chunks = [page_text]
                
                # Add chunks to batch processing lists
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        text_chunks.append(chunk)
                        chunk_metadata.append({
                            "page_num": page_num,
                            "page": page_num + 1,
                            "source": pdf_base_name,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        })
                
                # Process in batches
                if len(text_chunks) >= BATCH_SIZE:
                    process_text_batch(text_chunks, chunk_metadata, embedder, pdf_id, points_to_upsert)
                    embeddings_count += len(text_chunks)
                    text_chunks = []
                    chunk_metadata = []

            # Process Images (skip if configured)
            if not SKIP_IMAGES:
                process_page_images(page, page_num, document, embedder, pdf_id, pdf_base_name, 
                                  image_output_dir, points_to_upsert)
                                  
        # Process any remaining text chunks
        if text_chunks:
            process_text_batch(text_chunks, chunk_metadata, embedder, pdf_id, points_to_upsert)
            embeddings_count += len(text_chunks)

        try: document.close()
        except Exception as close_err: logger.error(f"Error closing PDF: {close_err}")

        # Batch upsert with improved chunking
        if points_to_upsert:
            logger.info(f"Upserting {len(points_to_upsert)} points for PDF {pdf_id}...")
            try:
                batch_size = 50  # Optimal batch size for Qdrant
                for i in range(0, len(points_to_upsert), batch_size):
                    batch = points_to_upsert[i:i+batch_size]
                    client.upsert(collection_name=collection_name, points=batch, wait=True)
                    logger.info(f"Upserted batch {i//batch_size + 1}/{(len(points_to_upsert)-1)//batch_size + 1} with {len(batch)} points")
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

def process_text_batch(text_chunks, chunk_metadata, embedder, pdf_id, points_to_upsert):
    """Process a batch of text chunks for better performance"""
    if not text_chunks:
        return
    
    # Get embeddings in a batch for better performance
    embeddings = embedder.get_embeddings_batch(text_chunks)
    
    # Create points from embeddings
    for i, (text, metadata, embedding) in enumerate(zip(text_chunks, chunk_metadata, embeddings)):
        if embedding != [0.0] * VECTOR_SIZE:
            # Create a more descriptive payload that helps with retrieval
            payload = {
                "pdf_id": pdf_id,
                "source": metadata["source"],
                "page": metadata["page"],
                "text": text,
                "type": "text",
                "chunk_index": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 1)
            }
            # Use UUID for point ID
            point_id = str(uuid.uuid4())
            points_to_upsert.append(models.PointStruct(id=point_id, vector=embedding, payload=payload))
        else:
            logger.warning(f"Failed text embed page {metadata['page']} chunk {metadata.get('chunk_index', 0)}")

def process_page_images(page, page_num, document, embedder, pdf_id, pdf_base_name, image_output_dir, points_to_upsert):
    """Process images from a page"""
    try:
        # Use a faster image extraction method
        image_list = page.get_images(full=True)
        image_count = 0
        
        for img_index, img_info in enumerate(image_list):
            # Only process larger, more meaningful images
            xref = img_info[0]
            
            try:
                base_image = document.extract_image(xref)
                if not base_image: continue
                image_bytes = base_image["image"]
                
                # Create image and check size/quality
                image = Image.open(io.BytesIO(image_bytes))
                
                # Skip very small images that aren't informative
                if image.width < 100 or image.height < 100:
                    continue
                    
                # Skip images that are too large (likely full-page backgrounds)
                if image.width > 2000 and image.height > 2000:
                    # Could be a page scan - keep it but resize for efficiency
                    image = image.resize((1024, int(1024 * image.height / image.width)), Image.LANCZOS)
                    
                # Get embedding
                image_embedding = embedder.get_embedding(image, "image")

                if image_embedding != [0.0] * VECTOR_SIZE:
                    # Use 're' module correctly for safe filename
                    safe_pdf_base = re.sub(r'[^\w\-_\.]', '_', os.path.splitext(pdf_base_name)[0])
                    image_filename = f"{safe_pdf_base}_page_{page_num + 1}_img_{img_index + 1}.png"
                    image_save_path = os.path.join(image_output_dir, image_filename)
                    
                    # Use a lower quality for faster saves
                    image.convert("RGB").save(image_save_path, "PNG", optimize=True)

                    # Create a more descriptive payload
                    payload = {
                        "pdf_id": pdf_id,
                        "source": pdf_base_name,
                        "page": page_num + 1,
                        "image_path": image_save_path,
                        "type": "image",
                        "width": image.width,
                        "height": image.height
                    }
                    # Use UUID for point ID
                    point_id = str(uuid.uuid4())
                    points_to_upsert.append(models.PointStruct(id=point_id, vector=image_embedding, payload=payload))
                    image_count += 1
            except Exception as img_err: 
                logger.error(f"Error with image page {page_num+1} img {img_index+1}: {img_err}", exc_info=False)
                
        if image_count > 0:
            logger.info(f"Processed {image_count} images from page {page_num+1}")
    except Exception as e:
        logger.error(f"Error processing images on page {page_num+1}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Compute embeddings for PDF (Text & Image w/ Text Model), store in Qdrant.')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--pdf_id', required=True, help='MongoDB ID of the PDF document')
    parser.add_argument('--collection_name', default=DEFAULT_COLLECTION, help='Name of the Qdrant collection')
    parser.add_argument('--skip_images', action='store_true', help='Skip processing images to improve speed')
    args = parser.parse_args()

    # Override global setting if explicitly set in command line
    global SKIP_IMAGES
    if args.skip_images:
        SKIP_IMAGES = True
        logger.info("Image processing disabled via command line flag")

    result = process_pdf(args.pdf_path, args.pdf_id, args.collection_name)
    print(json.dumps(result)) # Output result as JSON for backend

if __name__ == "__main__":
    main()