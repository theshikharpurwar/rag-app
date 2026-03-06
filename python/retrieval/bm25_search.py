# python/retrieval/bm25_search.py
"""
BM25 Keyword Search for hybrid retrieval.

Uses rank_bm25 (BM25Okapi) to build a sparse keyword index at ingestion time.
The index is persisted as a pickle file and loaded at query time.
"""

import logging
import os
import pickle
import re
import string

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace tokenizer with lowercasing and punctuation removal.
    No external NLP library required.
    """
    text = text.lower()
    # Remove punctuation except hyphens (useful for compound terms)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    # Filter very short tokens (single chars except meaningful ones)
    tokens = [t for t in tokens if len(t) > 1 or t in ('a', 'i')]
    return tokens


class BM25Index:
    """
    BM25 keyword search index built from document chunks.

    Build once at ingestion time, persist, then load at query time.
    Supports filtering by pdf_id since the index stores per-chunk metadata.
    """

    def __init__(self):
        self.bm25 = None
        self.chunks = []       # list of dicts: {"text": ..., "pdf_id": ..., "page": ..., ...}
        self.tokenized = []    # parallel list of tokenized texts
        self._is_built = False

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from a list of chunk dicts.

        Each chunk dict must have at least: {"text": str}
        Optional metadata: {"pdf_id": str, "page": int, "source": str, "chunk_index": int}
        """
        if not chunks:
            logger.warning("[BM25] No chunks provided, index will be empty")
            self._is_built = True
            return

        self.chunks = chunks
        self.tokenized = [_tokenize(c.get("text", "")) for c in chunks]

        # Filter out empty tokenizations
        valid_indices = [i for i, t in enumerate(self.tokenized) if len(t) > 0]
        if not valid_indices:
            logger.warning("[BM25] All chunks produced empty tokenizations")
            self._is_built = True
            return

        valid_tokenized = [self.tokenized[i] for i in valid_indices]
        self.bm25 = BM25Okapi(valid_tokenized)
        # Map BM25 internal indices back to our chunk indices
        self._valid_indices = valid_indices
        self._is_built = True

        logger.info(f"[BM25] Built index with {len(valid_indices)} chunks "
                     f"({len(set(c.get('pdf_id', '') for c in chunks))} document(s))")

    def search(self, query: str, pdf_id: str = None, top_k: int = 10) -> list[dict]:
        """
        Search the BM25 index for the given query.

        Args:
            query: Natural language query string
            pdf_id: Optional filter — only return chunks from this document
            top_k: Max results to return

        Returns:
            List of dicts with keys: text, page, source, score, chunk_index
        """
        if not self._is_built or self.bm25 is None:
            logger.warning("[BM25] Index not built or empty, returning empty results")
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores for all indexed chunks
        scores = self.bm25.get_scores(query_tokens)

        # Map scores back to original chunk indices and apply pdf_id filter
        results = []
        for bm25_idx, score in enumerate(scores):
            if score <= 0:
                continue
            original_idx = self._valid_indices[bm25_idx]
            chunk = self.chunks[original_idx]

            # Filter by pdf_id if specified
            if pdf_id and chunk.get("pdf_id") != pdf_id:
                continue

            results.append({
                "text": chunk.get("text", ""),
                "page": chunk.get("page", "N/A"),
                "source": chunk.get("source", "Unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "score": float(score),
                "retrieval_method": "bm25",
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"[BM25] Query '{query[:50]}...' returned {len(results[:top_k])} results")
        return results[:top_k]

    def save(self, path: str) -> None:
        """Persist the BM25 index to a pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'tokenized': self.tokenized,
                'valid_indices': getattr(self, '_valid_indices', []),
            }, f)
        logger.info(f"[BM25] Index saved to {path}")

    def load(self, path: str) -> bool:
        """Load a BM25 index from a pickle file. Returns True if successful."""
        if not os.path.exists(path):
            logger.warning(f"[BM25] Index file not found: {path}")
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']
            self.tokenized = data['tokenized']
            self._valid_indices = data.get('valid_indices', list(range(len(self.chunks))))
            self._is_built = True
            logger.info(f"[BM25] Index loaded from {path} ({len(self.chunks)} chunks)")
            return True
        except Exception as e:
            logger.error(f"[BM25] Failed to load index from {path}: {e}")
            return False
