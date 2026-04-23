# python/retrieval/rrf_fusion.py
"""
Weighted Reciprocal Rank Fusion (RRF) for merging multiple ranked retrieval lists.

Implements the formula:
    Score(d) = Σ w_i · 1/(k + rank_i(d))

Where:
    w_i = weight for retrieval method i
    k   = smoothing constant (default 60, from Cormack et al. 2009)
    rank_i(d) = position of document d in ranked list i

Adapted for multi-method fusion: vector similarity, BM25, and graph traversal.
"""

import logging

logger = logging.getLogger(__name__)


def rrf_fuse(ranked_lists: list[list[dict]],
             weights: list[float] = None,
             k: int = 60,
             top_k: int = 10) -> list[dict]:
    """
    Fuse multiple ranked lists using Weighted Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists. Each result dict must have
                      at least {"text": str} for identity matching.
                      Additional keys: page, source, score, chunk_index
        weights: Weights for each ranked list (must sum to ~1.0).
                 If None, equal weights are used.
        k: RRF smoothing constant (default 60)
        top_k: Number of results to return

    Returns:
        Fused ranked list with combined scores and retrieval provenance.
    """
    if not ranked_lists:
        return []

    n_lists = len(ranked_lists)

    if weights is None:
        weights = [1.0 / n_lists] * n_lists
    elif len(weights) != n_lists:
        logger.warning(f"[RRF] Weight count ({len(weights)}) != list count ({n_lists}), using equal weights")
        weights = [1.0 / n_lists] * n_lists

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]

    # Build fused score map
    # Key: (text_hash, page, source) to identify unique chunks
    fused = {}  # key -> {"score": float, "data": dict, "methods": list}

    for list_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[list_idx]

        for rank, result in enumerate(ranked_list):
            # Create a dedup key based on content identity
            text = result.get("text", "")
            page = result.get("page", "N/A")
            source = result.get("source", "Unknown")

            # Use a combination for identity (text prefix + page + source)
            chunk_key = (text[:200], str(page), source)

            # RRF score contribution
            rrf_score = weight * (1.0 / (k + rank + 1))  # rank is 0-indexed

            method = result.get("retrieval_method", f"method_{list_idx}")

            if chunk_key in fused:
                fused[chunk_key]["score"] += rrf_score
                if method not in fused[chunk_key]["methods"]:
                    fused[chunk_key]["methods"].append(method)
                # Keep the original score from each method
                fused[chunk_key]["method_scores"][method] = result.get("score", 0.0)
            else:
                fused[chunk_key] = {
                    "score": rrf_score,
                    "data": {
                        "text": text,
                        "page": page,
                        "source": source,
                        "chunk_index": result.get("chunk_index", 0),
                    },
                    "methods": [method],
                    "method_scores": {method: result.get("score", 0.0)},
                }

    # Convert to sorted list
    results = []
    for chunk_key, entry in fused.items():
        result = entry["data"].copy()
        result["score"] = entry["score"]
        result["retrieval_methods"] = entry["methods"]
        result["method_scores"] = entry["method_scores"]
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    # Log fusion summary
    method_counts = {}
    for r in results[:top_k]:
        for m in r.get("retrieval_methods", []):
            method_counts[m] = method_counts.get(m, 0) + 1
    logger.info(f"[RRF] Fused {sum(len(l) for l in ranked_lists)} results from "
                f"{n_lists} lists → {len(results)} unique, returning top {top_k}. "
                f"Method distribution: {method_counts}")

    return results[:top_k]


def hybrid_retrieve(query: str,
                    pdf_id: str,
                    vector_results: list[dict],
                    bm25_index=None,
                    knowledge_graph=None,
                    graph_retriever=None,
                    vector_weight: float = 0.4,
                    bm25_weight: float = 0.3,
                    graph_weight: float = 0.3,
                    community_results=None,
                    community_weight: float = 0.0,
                    rrf_k: int = 60,
                    top_k: int = 10) -> list[dict]:
    """
    High-level hybrid retrieval combining vector, BM25, and graph results.

    Gracefully degrades: if BM25 or graph components are unavailable,
    adjusts weights to redistribute among available methods.

    Args:
        query: User query string
        pdf_id: Document ID to filter by
        vector_results: Pre-computed vector search results from Qdrant
        bm25_index: Optional BM25Index instance
        knowledge_graph: Optional KnowledgeGraph instance
        graph_retriever: Optional GraphRetriever instance
        vector_weight: Weight for vector results (α)
        bm25_weight: Weight for BM25 results (β)
        graph_weight: Weight for graph results (γ)
        rrf_k: RRF smoothing constant
        top_k: Max results to return

    Returns:
        Fused and ranked list of chunk results.
    """
    ranked_lists = []
    weights = []

    # Path A: Vector results (always available)
    if vector_results:
        # Convert Qdrant results to standard dict format
        formatted_vector = []
        for hit in vector_results:
            payload = hit.payload if hasattr(hit, 'payload') and isinstance(hit.payload, dict) else {}
            formatted_vector.append({
                "text": payload.get("text", ""),
                "page": payload.get("page", "N/A"),
                "source": payload.get("source", "Unknown"),
                "chunk_index": payload.get("chunk_index", 0),
                "score": hit.score if hasattr(hit, 'score') else 0.0,
                "retrieval_method": "vector",
            })
        ranked_lists.append(formatted_vector)
        weights.append(vector_weight)
        logger.info(f"[Hybrid] Vector: {len(formatted_vector)} results")
    else:
        logger.warning("[Hybrid] No vector results available")

    # Path B: BM25 keyword search
    if bm25_index is not None:
        try:
            bm25_results = bm25_index.search(query, pdf_id=pdf_id, top_k=top_k * 2)
            if bm25_results:
                ranked_lists.append(bm25_results)
                weights.append(bm25_weight)
                logger.info(f"[Hybrid] BM25: {len(bm25_results)} results")
            else:
                logger.info("[Hybrid] BM25 returned no results")
        except Exception as e:
            logger.warning(f"[Hybrid] BM25 search failed: {e}")
    else:
        logger.info("[Hybrid] BM25 index not available, skipping BM25 path")

    # Path C: Graph traversal
    if graph_retriever is not None and knowledge_graph is not None and knowledge_graph.num_nodes > 0:
        try:
            graph_results = graph_retriever.score_chunks_by_graph(query, top_k=top_k * 2)
            if graph_results:
                ranked_lists.append(graph_results)
                weights.append(graph_weight)
                logger.info(f"[Hybrid] Graph: {len(graph_results)} results")
            else:
                logger.info("[Hybrid] Graph retrieval returned no results")
        except Exception as e:
            logger.warning(f"[Hybrid] Graph retrieval failed: {e}")
    else:
        logger.info("[Hybrid] Knowledge graph not available, skipping graph path")

    if community_results:
        ranked_lists.append(community_results)
        weights.append(community_weight)
        logger.info(f"[Hybrid] Community summaries: {len(community_results)} results")

    # If only vector results, return them directly (no fusion needed)
    if len(ranked_lists) <= 1:
        if ranked_lists:
            logger.info("[Hybrid] Single retrieval path, skipping RRF fusion")
            return ranked_lists[0][:top_k]
        return []

    # Fuse all available results
    from config.models import FUSION_METHOD
    if FUSION_METHOD == "dbsf":
        from .dbsf_fusion import dbsf_fuse
        return dbsf_fuse(ranked_lists, weights=weights, top_k=top_k)
    return rrf_fuse(ranked_lists, weights=weights, k=rrf_k, top_k=top_k)
