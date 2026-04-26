# FILE: python/compute_embeddings.py
# Using centralized model configuration

import os
import sys
import json
import argparse
import logging
import time
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from embeddings.ollama_embed import OllamaEmbedder # Use Ollama embedder instead of heavy ML libraries
import re    # <--- FIX: Import 're' module
import uuid  # <--- FIX: Import 'uuid' module for generating valid IDs

# Import centralized configuration
from config import (
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    EMBED_BATCH_SIZE,
    DEFAULT_VECTOR_SIZE,
    QDRANT_HOST,
    QDRANT_PORT,
    DEFAULT_COLLECTION,
    OLLAMA_HOST_URL,
    OLLAMA_API_BASE,
    TEXT_CHUNK_SIZE,
    TEXT_CHUNK_OVERLAP,
    IMAGE_SAVE_DIR_RELATIVE,
    RENDERING_DPI,
    # Phase 2
    ENABLE_KNOWLEDGE_GRAPH,
    ENABLE_COMMUNITY_SUMMARIES,
    KG_EXTRACTION_MODE,
    KG_NP_EXTRACTOR,
    KG_TRIPLE_BATCH_SIZE,
    KG_EXTRACT_CONCURRENCY,
    KG_EXTRACT_MAX_CHARS,
    KG_STORAGE_PRETTY,
    KG_LLM_MODEL,
    INDICES_DIR,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print current configuration for debugging to stderr instead of stdout
import sys
def print_config_to_stderr():
    """Print the current model configuration for debugging to stderr."""
    print("=" * 60, file=sys.stderr)
    print("🔧 CURRENT RAG APPLICATION CONFIGURATION", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"LLM Model:        {LLM_MODEL_NAME}", file=sys.stderr)
    print(f"Embedding Model:  {EMBEDDING_MODEL_NAME}", file=sys.stderr)
    print(f"Vector Size:      {DEFAULT_VECTOR_SIZE}", file=sys.stderr)
    print(f"Ollama Host:      {OLLAMA_HOST_URL}", file=sys.stderr)
    print(f"Qdrant Host:      {QDRANT_HOST}:{QDRANT_PORT}", file=sys.stderr)
    print(f"Collection:       {DEFAULT_COLLECTION}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

print_config_to_stderr()

# --- Configuration (now imported from central config) ---
VECTOR_SIZE = DEFAULT_VECTOR_SIZE
BATCH_SIZE = 10  # Process in batches for performance
SKIP_IMAGES = os.environ.get("SKIP_IMAGES", "false").lower() == "true"  # Option to skip images
USE_DOCLING = os.environ.get("USE_DOCLING", "false").lower() == "true"  # Use Docling for better RAG extraction
# --- End Configuration ---

# Remove or comment out the SimpleEmbedder class
# class SimpleEmbedder:
#     """Simple embedder that uses Sentence Transformers"""
#     def __init__(self, model_name=EMBEDDING_MODEL_NAME):
#         logger.info(f"Initializing embedder with model: {model_name}")
#         try:
#             self.model = SentenceTransformer(model_name)
#             self.model_name = model_name
#             test_embedding = self.model.encode("test")
#             actual_size = len(test_embedding)
#             if actual_size != VECTOR_SIZE:
#                  logger.warning(f"Model {model_name} output dimension ({actual_size}) does not match configured VECTOR_SIZE ({VECTOR_SIZE}).")
#             logger.info(f"Embedding model {model_name} loaded successfully (Dim: {actual_size}).")
#             
#             # Enable batching for better performance
#             self.model.max_seq_length = 256  # Limit sequence length for faster processing
#         except Exception as e:
#             logger.error(f"Failed to load embedding model {model_name}: {e}")
#             raise ImportError(f"Could not load embedding model {model_name}") from e
# 
#     def get_embedding(self, content, content_type="text"):
#         try:
#             if content_type.lower() == "text":
#                 if not content or not content.strip():
#                     logger.warning("Empty text content provided, returning zero vector")
#                     return [0.0] * VECTOR_SIZE
#                 embedding = self.model.encode(content)
#                 return embedding.tolist()
#             elif content_type.lower() == "image":
#                 if content is None:
#                     logger.warning("None image content provided, returning zero vector")
#                     return [0.0] * VECTOR_SIZE
#                 try:
#                     # Attempt direct encode (will likely fail for text models but keeps original logic flow)
#                     embedding = self.model.encode(content)
#                     return embedding.tolist()
#                 except Exception as img_embed_err:
#                     logger.warning(f"Failed to directly embed image with {self.model_name}: {img_embed_err}. Using placeholder text.")
#                     placeholder_text = "image content"
#                     embedding = self.model.encode(placeholder_text)
#                     return embedding.tolist()
#             else:
#                 logger.error(f"Unsupported content type: {content_type}")
#                 return [0.0] * VECTOR_SIZE
#         except Exception as e:
#             logger.error(f"Error generating embedding: {str(e)}")
#             return [0.0] * VECTOR_SIZE
#             
#     def get_embeddings_batch(self, texts):
#         """Process multiple texts in a batch for better performance"""
#         if not texts:
#             return []
#         try:
#             embeddings = self.model.encode(texts, show_progress_bar=False)
#             return [emb.tolist() for emb in embeddings]
#         except Exception as e:
#             logger.error(f"Error generating batch embeddings: {str(e)}")
#             return [[0.0] * VECTOR_SIZE for _ in texts]

def chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=TEXT_CHUNK_OVERLAP):
    """
    Split text into smaller chunks with overlap for better semantic search.
    Improved to respect semantic boundaries (paragraphs, sentences) for better retrieval.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    # First, try to split by paragraphs for better semantic coherence
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_len = len(para)
        
        # If adding this paragraph doesn't exceed chunk_size, add it
        if current_size + para_len + 2 <= chunk_size:  # +2 for \n\n
            if current_chunk:
                current_chunk += "\n\n" + para
                current_size += para_len + 2
            else:
                current_chunk = para
                current_size = para_len
        else:
            # Current paragraph would make chunk too large
            if current_chunk:
                # Save current chunk
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                if current_size > overlap:
                    # Extract last 'overlap' characters for continuity
                    overlap_text = current_chunk[-overlap:]
                    # Try to start from a sentence boundary
                    sentence_start = overlap_text.find('. ')
                    if sentence_start != -1:
                        overlap_text = overlap_text[sentence_start + 2:]
                    current_chunk = overlap_text.strip() + "\n\n" + para
                    current_size = len(current_chunk)
                else:
                    current_chunk = para
                    current_size = para_len
            else:
                # This paragraph is too large on its own, split it by sentences
                if para_len > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sent_chunk = ""
                    sent_size = 0
                    
                    for sent in sentences:
                        if sent_size + len(sent) + 1 <= chunk_size:
                            sent_chunk += (" " if sent_chunk else "") + sent
                            sent_size += len(sent) + 1
                        else:
                            if sent_chunk:
                                chunks.append(sent_chunk.strip())
                            
                            # Handle very long sentences
                            if len(sent) > chunk_size:
                                # Split at word boundaries
                                words = sent.split()
                                word_chunk = ""
                                for word in words:
                                    if len(word_chunk) + len(word) + 1 <= chunk_size:
                                        word_chunk += (" " if word_chunk else "") + word
                                    else:
                                        if word_chunk:
                                            chunks.append(word_chunk)
                                        word_chunk = word
                                if word_chunk:
                                    sent_chunk = word_chunk
                                    sent_size = len(word_chunk)
                            else:
                                sent_chunk = sent
                                sent_size = len(sent)
                    
                    if sent_chunk:
                        current_chunk = sent_chunk
                        current_size = sent_size
                else:
                    current_chunk = para
                    current_size = para_len
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Post-process: ensure no chunk is too small (merge tiny chunks)
    final_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # If this chunk is very small and we have a previous chunk, try to merge
        if len(chunk) < 100 and final_chunks and len(final_chunks[-1]) + len(chunk) + 2 <= chunk_size:
            final_chunks[-1] += "\n\n" + chunk
        else:
            final_chunks.append(chunk)
    
    return final_chunks if final_chunks else [text.strip()] if text.strip() else []


def extract_with_docling(pdf_path):
    """
    Extract text from PDF using Docling (IBM) for RAG-optimized extraction.
    Returns list of (page_num, text, chunk_index) tuples.
    Uses Docling's HybridChunker which is purpose-built for RAG pipelines.
    OCR is disabled (do_ocr=False) — Tesseract is not in the container.
    For scanned PDFs, consider using DeepSeek-OCR separately.
    Falls back to empty list on any error so PyMuPDF fallback can take over.
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.chunking import HybridChunker

        # Disable OCR — Tesseract not available in container.
        # Works perfectly for text-based PDFs (research papers, reports, etc.)
        pipeline_options = PdfPipelineOptions(do_ocr=False)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        logger.info("[Docling] Converting PDF (OCR disabled)...")
        result = converter.convert(pdf_path)

        logger.info("[Docling] Chunking document with HybridChunker...")
        chunker = HybridChunker()
        chunks = list(chunker.chunk(result.document))
        logger.info(f"[Docling] Produced {len(chunks)} chunks")

        extracted = []
        for i, chunk in enumerate(chunks):
            text = chunk.text.strip()
            if not text:
                continue
            # Extract page number from provenance metadata if available
            page_num = 1  # default
            try:
                items = chunk.meta.doc_items if hasattr(chunk.meta, 'doc_items') else []
                if items and hasattr(items[0], 'prov') and items[0].prov:
                    page_num = items[0].prov[0].page_no
            except Exception:
                pass
            extracted.append((page_num, text, i))

        return extracted

    except ImportError:
        logger.warning("[Docling] Not installed. Falling back to PyMuPDF.")
        return []
    except Exception as e:
        logger.error(f"[Docling] Extraction failed: {e}. Falling back to PyMuPDF.")
        return []


def _kg_mode_log_line() -> str:
    if KG_EXTRACTION_MODE == "llm_triples":
        return (
            f"[KG] mode=llm_triples model={KG_LLM_MODEL} batch={KG_TRIPLE_BATCH_SIZE} "
            f"concurrency={KG_EXTRACT_CONCURRENCY} max_chars={KG_EXTRACT_MAX_CHARS} "
            f"storage={'pretty' if KG_STORAGE_PRETTY else 'compact'}"
        )
    if KG_EXTRACTION_MODE == "noun_phrase_cooccurrence":
        return (
            f"[KG] mode=noun_phrase_cooccurrence extractor={KG_NP_EXTRACTOR} "
            f"storage={'pretty' if KG_STORAGE_PRETTY else 'compact'}"
        )
    return "[KG] mode=disabled (graph retrieval will be skipped)"


def _write_kg_audit_artifact(pdf_id: str, audit_data: dict) -> str | None:
    try:
        diagnostics_dir = os.path.join(INDICES_DIR, "diagnostics")
        os.makedirs(diagnostics_dir, exist_ok=True)
        path = os.path.join(diagnostics_dir, f"{pdf_id}_kg_audit.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2, default=str)
        logger.info("[KG] Audit artifact saved: %s", path)
        return path
    except Exception as e:
        logger.warning("[KG] Failed to write audit artifact for %s: %s", pdf_id, e)
        return None


# Using process_pdf function name, includes pdf_id argument
def process_pdf(pdf_path, pdf_id, collection_name=DEFAULT_COLLECTION, reset=False):
    """Process PDF, extract text & images, compute embeddings, store in Qdrant with pdf_id."""
    if not pdf_id:
        logger.error("Missing pdf_id for processing.")
        return {"success": False, "error": "PDF ID not provided to embedding script."}

    try:
        logger.info(f"Processing PDF: {pdf_path} (ID: {pdf_id}) for collection: {collection_name}")
        logger.info(_kg_mode_log_line())
        # Use Ollama embedder (no heavy ML dependencies)
        embedder = OllamaEmbedder(model_name=EMBEDDING_MODEL_NAME, batch_size=EMBED_BATCH_SIZE)
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

        if reset:
            pdf_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="pdf_id",
                        match=models.MatchValue(value=pdf_id),
                    )
                ]
            )
            if collection_info:
                try:
                    pre = client.count(
                        collection_name=collection_name,
                        count_filter=pdf_filter,
                        exact=True,
                    ).count
                    if pre > 0:
                        client.delete(
                            collection_name=collection_name,
                            points_selector=models.FilterSelector(filter=pdf_filter),
                            wait=True,
                        )
                        logger.info(
                            f"[Reset] Deleted {pre} existing points for pdf_id={pdf_id}"
                        )
                    else:
                        logger.info(f"[Reset] No existing points for pdf_id={pdf_id}")
                except Exception as e:
                    logger.warning(
                        f"[Reset] pdf_id-scoped delete failed (continuing): {e}"
                    )
            else:
                logger.info(
                    "[Reset] Fresh or recreated collection — no pdf_id-scoped delete needed"
                )

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

        # ── Docling path ──────────────────────────────────────────────────────
        if USE_DOCLING:
            logger.info("[Docling] USE_DOCLING=true — using Docling for text extraction")
            docling_chunks = extract_with_docling(pdf_path)

            if docling_chunks:
                text_chunks = [text for (_, text, _) in docling_chunks]
                chunk_metadata = [
                    {
                        "page_num": page_num - 1,
                        "page": page_num,
                        "source": pdf_base_name,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(docling_chunks),
                        "extractor": "docling"
                    }
                    for (page_num, _, chunk_idx) in docling_chunks
                ]
                # Embed all Docling chunks
                for i in range(0, len(text_chunks), BATCH_SIZE):
                    batch_texts = text_chunks[i:i + BATCH_SIZE]
                    batch_meta = chunk_metadata[i:i + BATCH_SIZE]
                    process_text_batch(batch_texts, batch_meta, embedder, pdf_id, points_to_upsert)
                    embeddings_count += len(batch_texts)

                num_pages = max(m["page"] for m in chunk_metadata) if chunk_metadata else 0
                logger.info(f"[Docling] Embedded {embeddings_count} chunks across {num_pages} pages")
            else:
                logger.warning("[Docling] No chunks returned — falling back to PyMuPDF")
                USE_DOCLING_ACTIVE = False  # signal fallback
            docling_succeeded = bool(docling_chunks)
        else:
            docling_succeeded = False

        # ── PyMuPDF4LLM path (default or fallback) ───────────────────────────
        if not USE_DOCLING or not docling_succeeded:
            md_pages = None
            extractor_label = "pymupdf4llm"
            try:
                logger.info("[PyMuPDF4LLM] Extracting text as Markdown (preserves tables, headers)...")
                # pymupdf4llm prints recommendations to stdout which corrupts JSON output.
                # Redirect stdout → stderr during its execution.
                _real_stdout = sys.stdout
                sys.stdout = sys.stderr
                try:
                    import pymupdf4llm
                    # page_chunks=True → one dict per page: {"text": "...", "metadata": {"page": 0, ...}}
                    md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
                finally:
                    sys.stdout = _real_stdout
            except ImportError as imp_err:
                # Graceful fallback: use plain PyMuPDF (fitz) text extraction so the
                # pipeline (and evaluation harness) still works without pymupdf4llm.
                logger.warning(
                    f"[PyMuPDF4LLM] Not available ({imp_err}); falling back to fitz plain-text extraction."
                )
                extractor_label = "pymupdf_fitz"
                md_pages = []
                _fallback_doc = fitz.open(pdf_path)
                try:
                    for _pn, _page in enumerate(_fallback_doc):
                        md_pages.append({
                            "text": _page.get_text() or "",
                            "metadata": {"page": _pn},
                        })
                finally:
                    try:
                        _fallback_doc.close()
                    except Exception:
                        pass

            num_pages = len(md_pages)
            logger.info(f"[{extractor_label}] Got {num_pages} page(s)")

            text_chunks = []
            chunk_metadata = []

            for _idx, page_data in enumerate(md_pages):
                meta = page_data.get("metadata", {}) or {}
                # Newer pymupdf4llm exposes 1-indexed `page_number`; older versions
                # used 0-indexed `page`. Fall back to the enumeration index so the
                # page number is never silently 0/1 for every chunk.
                if meta.get("page_number") is not None:
                    page_num = int(meta["page_number"]) - 1  # normalise to 0-indexed
                elif meta.get("page") is not None:
                    page_num = int(meta["page"])
                else:
                    page_num = _idx
                page_text = page_data["text"].strip()

                if page_text:
                    chunks = chunk_text(page_text)
                    if not chunks and page_text:
                        chunks = [page_text]
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            text_chunks.append(chunk)
                            chunk_metadata.append({
                                "page_num": page_num,
                                "page": page_num + 1,
                                "source": pdf_base_name,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "extractor": extractor_label
                            })
                    if len(text_chunks) >= BATCH_SIZE:
                        process_text_batch(text_chunks, chunk_metadata, embedder, pdf_id, points_to_upsert)
                        embeddings_count += len(text_chunks)
                        text_chunks = []
                        chunk_metadata = []

            # Image processing still needs fitz (pymupdf4llm doesn't extract raw images)
            if not SKIP_IMAGES:
                document = fitz.open(pdf_path)
                for page_num, page in enumerate(document):
                    process_page_images(page, page_num, document, embedder, pdf_id, pdf_base_name,
                                      image_output_dir, points_to_upsert)
                try: document.close()
                except Exception as close_err: logger.error(f"Error closing PDF: {close_err}")

            if text_chunks:
                process_text_batch(text_chunks, chunk_metadata, embedder, pdf_id, points_to_upsert)
                embeddings_count += len(text_chunks)

        # ── Upsert to Qdrant ──────────────────────────────────────────────────
        if points_to_upsert:
            logger.info(f"Upserting {len(points_to_upsert)} points for PDF {pdf_id}...")
            try:
                batch_size = 50
                for i in range(0, len(points_to_upsert), batch_size):
                    batch = points_to_upsert[i:i+batch_size]
                    client.upsert(collection_name=collection_name, points=batch, wait=True)
                    logger.info(f"Upserted batch {i//batch_size + 1}/{(len(points_to_upsert)-1)//batch_size + 1} with {len(batch)} points")
                logger.info(f"Upsert successful for {len(points_to_upsert)} points (PDF ID: {pdf_id}).")
            except Exception as e:
                logger.error(f"Qdrant upsert failed for PDF {pdf_id}: {e}", exc_info=True)
                error_detail = str(e)
                if hasattr(e, 'http_body'): error_detail = getattr(e, 'http_body', str(e))
                return {"success": False, "error": f"Qdrant upsert failed: {error_detail}"}
        else:
            logger.warning("No text or image content found/embedded.")

        # ── Phase 2: Build BM25 Index & Knowledge Graph ───────────────────────
        all_text_chunks = [p.payload for p in points_to_upsert if hasattr(p, 'payload') and p.payload.get('type') == 'text']

        bm25_path = None
        kg_path = None

        if all_text_chunks:
            os.makedirs(INDICES_DIR, exist_ok=True)

            # ── BM25 Index (always built — lightweight) ───────────────────────
            try:
                from retrieval.bm25_search import BM25Index
                bm25 = BM25Index()
                bm25_chunks = [{"text": c.get("text", ""), "pdf_id": c.get("pdf_id", pdf_id),
                                "page": c.get("page", 0), "source": c.get("source", ""),
                                "chunk_index": c.get("chunk_index", 0)} for c in all_text_chunks]
                bm25.build_index(bm25_chunks)
                bm25_path = os.path.join(INDICES_DIR, f"{pdf_id}_bm25.pkl")
                bm25.save(bm25_path)
                logger.info(f"[BM25] Index saved for PDF {pdf_id}: {bm25_path}")
            except Exception as e:
                logger.error(f"[BM25] Failed to build index for PDF {pdf_id}: {e}", exc_info=True)

            # ── Knowledge Graph (gated by ENABLE_KNOWLEDGE_GRAPH) ─────────────
            if ENABLE_KNOWLEDGE_GRAPH and KG_EXTRACTION_MODE != "disabled":
                try:
                    from llm.ollama_llm import OllamaLLM
                    from retrieval.entity_extractor import EntityExtractor, NounPhraseCooccurrenceExtractor
                    from retrieval.knowledge_graph import KnowledgeGraph

                    logger.info(f"[KG] Starting entity extraction for PDF {pdf_id} ({len(all_text_chunks)} chunks)...")
                    extraction_started = time.time()
                    kg_llm = None
                    if KG_EXTRACTION_MODE == "llm_triples":
                        kg_llm = OllamaLLM(model_name=KG_LLM_MODEL, api_base=OLLAMA_API_BASE)
                        extractor = EntityExtractor(
                            kg_llm,
                            batch_size=KG_TRIPLE_BATCH_SIZE,
                            max_chars=KG_EXTRACT_MAX_CHARS,
                            concurrency=KG_EXTRACT_CONCURRENCY,
                        )
                    elif KG_EXTRACTION_MODE == "noun_phrase_cooccurrence":
                        extractor = NounPhraseCooccurrenceExtractor(extractor_name=KG_NP_EXTRACTOR)
                    else:
                        raise ValueError(f"Unsupported KG_EXTRACTION_MODE: {KG_EXTRACTION_MODE}")

                    triples, sources, extraction_audit = extractor.extract_from_chunks(bm25_chunks)
                    extraction_elapsed = time.time() - extraction_started

                    if triples:
                        kg = KnowledgeGraph()
                        kg.set_metadata(
                            kg_mode=KG_EXTRACTION_MODE,
                            np_extractor=KG_NP_EXTRACTOR if KG_EXTRACTION_MODE == "noun_phrase_cooccurrence" else None,
                            storage_pretty=KG_STORAGE_PRETTY,
                            kg_llm_model=KG_LLM_MODEL if KG_EXTRACTION_MODE == "llm_triples" else None,
                            triple_batch_size=KG_TRIPLE_BATCH_SIZE,
                            extract_concurrency=KG_EXTRACT_CONCURRENCY,
                            extract_max_chars=KG_EXTRACT_MAX_CHARS,
                        )
                        build_started = time.time()
                        kg.build_from_triples(triples, sources)
                        build_elapsed = time.time() - build_started
                        community_started = time.time()
                        kg.detect_communities()
                        community_elapsed = time.time() - community_started
                        if ENABLE_COMMUNITY_SUMMARIES:
                            logger.info(f"[KG] Generating community summaries for PDF {pdf_id}...")
                            summary_embedder = OllamaEmbedder(
                                model_name=EMBEDDING_MODEL_NAME,
                                batch_size=EMBED_BATCH_SIZE,
                            )
                            n_sum = len(kg.generate_community_summaries(kg_llm, embedder=summary_embedder)) if kg_llm else 0
                            logger.info(f"[KG] Generated {n_sum} community summaries for PDF {pdf_id}")
                        kg_path = os.path.join(INDICES_DIR, f"{pdf_id}_graph.json")
                        save_started = time.time()
                        kg.save(kg_path)
                        save_elapsed = time.time() - save_started
                        graph_json_size_bytes = os.path.getsize(kg_path) if os.path.exists(kg_path) else 0
                        summary = kg.get_summary()
                        logger.info(f"[KG] Graph saved for PDF {pdf_id}: {summary['nodes']} nodes, "
                                     f"{summary['edges']} edges, {summary['communities']} communities")
                        audit_payload = {
                            "pdf_id": pdf_id,
                            "kg_mode": KG_EXTRACTION_MODE,
                            "np_extractor": KG_NP_EXTRACTOR if KG_EXTRACTION_MODE == "noun_phrase_cooccurrence" else None,
                            "kg_llm_model": KG_LLM_MODEL if KG_EXTRACTION_MODE == "llm_triples" else None,
                            "storage_pretty": KG_STORAGE_PRETTY,
                            "chunk_count": len(bm25_chunks),
                            "graph_summary": summary,
                            "graph_json_size_bytes": graph_json_size_bytes,
                            "timings": {
                                "extraction_seconds": extraction_elapsed,
                                "graph_build_seconds": build_elapsed,
                                "community_detection_seconds": community_elapsed,
                                "save_seconds": save_elapsed,
                            },
                            "extraction_audit": extraction_audit,
                        }
                        _write_kg_audit_artifact(pdf_id, audit_payload)
                    else:
                        logger.warning(f"[KG] No triples extracted for PDF {pdf_id}")
                        _write_kg_audit_artifact(
                            pdf_id,
                            {
                                "pdf_id": pdf_id,
                                "kg_mode": KG_EXTRACTION_MODE,
                                "chunk_count": len(bm25_chunks),
                                "graph_summary": {"nodes": 0, "edges": 0, "communities": 0},
                                "graph_json_size_bytes": 0,
                                "timings": {
                                    "extraction_seconds": extraction_elapsed,
                                    "graph_build_seconds": 0.0,
                                    "community_detection_seconds": 0.0,
                                    "save_seconds": 0.0,
                                },
                                "extraction_audit": extraction_audit,
                            },
                        )
                except Exception as e:
                    logger.error(f"[KG] Failed to build knowledge graph for PDF {pdf_id}: {e}", exc_info=True)
            else:
                logger.info("[KG] Knowledge graph disabled")

        result = {
            "success": True,
            "filename": pdf_base_name,
            "page_count": num_pages if 'num_pages' in locals() else 0,
            "embeddings_count": embeddings_count,
            "collection": collection_name,
            "bm25_index": bm25_path,
            "knowledge_graph": kg_path,
        }
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

    # 5. Use the correct task_type for embedding documents
    embeddings = embedder.encode_text(text_chunks, task_type="search_document")

    if not embeddings:
        logger.error("Failed to generate embeddings for the batch.")
        return

    # Create points from embeddings
    for i, (text, metadata, embedding) in enumerate(zip(text_chunks, chunk_metadata, embeddings)):
        if embedding and len(embedding) == VECTOR_SIZE:
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
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Delete existing Qdrant points for this pdf_id before upsert (idempotent re-ingest)',
    )
    args = parser.parse_args()

    # Override global setting if explicitly set in command line
    global SKIP_IMAGES
    if args.skip_images:
        SKIP_IMAGES = True
        logger.info("Image processing disabled via command line flag")

    result = process_pdf(
        args.pdf_path, args.pdf_id, args.collection_name, reset=args.reset
    )
    print(json.dumps(result)) # Output result as JSON for backend

if __name__ == "__main__":
    main()
