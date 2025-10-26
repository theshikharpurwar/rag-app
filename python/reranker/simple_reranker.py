"""
Lightweight reranker using a small cross-encoder model.
This improves retrieval quality by reordering the top-k results.
"""

import logging
from sentence_transformers import CrossEncoder
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleReranker:
    """
    Lightweight reranker using ms-marco-MiniLM-L-6-v2 cross-encoder.
    This is much smaller and faster than bge-reranker-base.
    
    Model size: ~80MB
    Speed: Very fast, suitable for real-time use
    """
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a lightweight cross-encoder model.
        
        Args:
            model_name (str): Name of the cross-encoder model to use
        """
        self.model_name = model_name
        logger.info(f"Loading reranker model: {model_name}")
        
        try:
            self.model = CrossEncoder(model_name, max_length=512)
            logger.info(f"✓ Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
    
    def rerank(self, query, documents, top_k=5):
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query (str): The search query
            documents (list): List of documents (dicts with 'text' field or QdrantSearchResult objects)
            top_k (int): Number of top documents to return after reranking
            
        Returns:
            list: Reranked documents (subset of top_k most relevant)
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        if len(documents) <= top_k:
            logger.info(f"Only {len(documents)} documents, no reranking needed")
            return documents
        
        try:
            # Extract text from documents
            # Handle both dict format and Qdrant search result format
            texts = []
            for doc in documents:
                if hasattr(doc, 'payload') and isinstance(doc.payload, dict):
                    # Qdrant search result format
                    texts.append(doc.payload.get('text', ''))
                elif isinstance(doc, dict):
                    # Dictionary format
                    texts.append(doc.get('text', ''))
                else:
                    logger.warning(f"Unknown document format: {type(doc)}")
                    texts.append(str(doc))
            
            # Create query-document pairs
            pairs = [[query, text] for text in texts]
            
            # Score all pairs
            logger.info(f"Reranking {len(pairs)} documents for query: '{query[:50]}...'")
            scores = self.model.predict(pairs)
            
            # Sort by score (descending) and get top_k
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top_k documents
            reranked = [doc for doc, score in scored_docs[:top_k]]
            
            # Log score distribution
            top_score = scored_docs[0][1] if scored_docs else 0
            bottom_score = scored_docs[-1][1] if scored_docs else 0
            logger.info(f"Reranking complete. Top score: {top_score:.3f}, Bottom score: {bottom_score:.3f}")
            logger.info(f"Returning top {len(reranked)} documents")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            # Fallback: return original top_k documents
            return documents[:top_k]
    
    def rerank_with_scores(self, query, documents, top_k=5):
        """
        Rerank documents and return them with their scores.
        
        Args:
            query (str): The search query
            documents (list): List of documents
            top_k (int): Number of top documents to return
            
        Returns:
            list: List of tuples (document, rerank_score)
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return [(doc, 0.0) for doc in documents]
        
        try:
            texts = []
            for doc in documents:
                if hasattr(doc, 'payload') and isinstance(doc.payload, dict):
                    texts.append(doc.payload.get('text', ''))
                elif isinstance(doc, dict):
                    texts.append(doc.get('text', ''))
                else:
                    texts.append(str(doc))
            
            pairs = [[query, text] for text in texts]
            scores = self.model.predict(pairs)
            
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking with scores: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]
