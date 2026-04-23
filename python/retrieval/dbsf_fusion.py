# python/retrieval/dbsf_fusion.py
"""
Distribution-Based Score Fusion (DBSF).

Normalizes raw scores from heterogeneous retrievers using μ ± 3σ,
then weighted-sums. Lists with fewer than 2 scores use neutral 0.5
per item (insufficient distribution); σ ≈ 0 yields the same.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize scores to [0, 1] using μ ± 3σ range.

    - < 2 items: return 0.5 for all (no distribution)
    - σ ≈ 0 (all same score): return 0.5 for all
    - Otherwise: clamp((s - (μ - 3σ)) / (6σ), 0, 1)
    """
    n = len(scores)
    if n < 2:
        return [0.5] * n

    mu = sum(scores) / n
    variance = sum((s - mu) ** 2 for s in scores) / n
    sigma = variance ** 0.5

    if sigma < 1e-9:
        return [0.5] * n

    lower = mu - 3 * sigma
    span = 6 * sigma
    return [max(0.0, min(1.0, (s - lower) / span)) for s in scores]


def dbsf_fuse(
    ranked_lists: list[list[dict]],
    weights: Optional[list[float]] = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Fuse ranked lists using DBSF.

    Each result dict must have:
      - 'score': raw retrieval score
      - 'text': for identity/dedup matching
      - 'page', 'source', 'chunk_index': metadata

    Returns dicts aligned with rrf_fuse (score, retrieval_methods, method_scores, ...).
    """
    if not ranked_lists:
        return []

    n_lists = len(ranked_lists)
    if weights is None:
        weights = [1.0 / n_lists] * n_lists
    elif len(weights) != n_lists:
        logger.warning(
            f"[DBSF] Weight count ({len(weights)}) != list count ({n_lists}), using equal weights"
        )
        weights = [1.0 / n_lists] * n_lists

    total_w = sum(weights)
    if total_w > 0:
        weights = [w / total_w for w in weights]

    normalized_lists = []
    for ranked_list in ranked_lists:
        raw_scores = [float(r.get("score", 0.0)) for r in ranked_list]
        norm_scores = _normalize_scores(raw_scores)
        normalized_lists.append(list(zip(ranked_list, norm_scores)))

    fused: dict = {}

    for list_idx, normed in enumerate(normalized_lists):
        w = weights[list_idx]

        for result, norm_score in normed:
            text = result.get("text", "")
            page = result.get("page", "N/A")
            source = result.get("source", "Unknown")
            chunk_key = (text[:200], str(page), source)

            contribution = w * norm_score
            method = result.get("retrieval_method", f"method_{list_idx}")

            if chunk_key in fused:
                fused[chunk_key]["score"] += contribution
                if method not in fused[chunk_key]["methods"]:
                    fused[chunk_key]["methods"].append(method)
                fused[chunk_key]["method_scores"][method] = result.get("score", 0.0)
            else:
                fused[chunk_key] = {
                    "score": contribution,
                    "data": {
                        "text": text,
                        "page": page,
                        "source": source,
                        "chunk_index": result.get("chunk_index", 0),
                    },
                    "methods": [method],
                    "method_scores": {method: result.get("score", 0.0)},
                }

    results = []
    for entry in fused.values():
        r = entry["data"].copy()
        r["score"] = entry["score"]
        r["retrieval_methods"] = entry["methods"]
        r["method_scores"] = entry["method_scores"]
        results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)

    method_counts = {}
    for r in results[:top_k]:
        for m in r.get("retrieval_methods", []):
            method_counts[m] = method_counts.get(m, 0) + 1
    logger.info(
        f"[DBSF] Fused {sum(len(l) for l in ranked_lists)} results from "
        f"{n_lists} lists → {len(results)} unique, returning top {top_k}. "
        f"Method distribution: {method_counts}"
    )

    return results[:top_k]
