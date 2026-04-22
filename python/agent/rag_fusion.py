"""
RAG-Fusion: retrieve once per sub-query, then merge ranked lists with RRF.

Each call to ``retrieve_fn`` already runs hybrid RRF (vector / BM25 / graph).
Merging those outputs again across sub-queries is a second RRF stage on ranks;
see Shi et al. (2024). Documented here so the double-fusion is intentional.
"""

import logging
from typing import Any, Callable, List, Tuple

from retrieval.rrf_fusion import rrf_fuse

logger = logging.getLogger(__name__)


class RAGFusion:
    def __init__(
        self,
        retrieve_fn: Callable[[str, Tuple[float, float, float]], Tuple[str, List[Any]]],
        rrf_k: int = 60,
        top_k: int = 5,
    ):
        self.retrieve_fn = retrieve_fn
        self.rrf_k = int(rrf_k)
        self.top_k = int(top_k)

    def retrieve(
        self,
        sub_queries: List[str],
        weights: Tuple[float, float, float],
    ) -> Tuple[str, List[Any]]:
        if not sub_queries:
            return "", []

        ranked_lists: List[List[dict]] = []
        for i, sq in enumerate(sub_queries):
            _ctx, sources = self.retrieve_fn(sq, weights)
            ranked: List[dict] = []
            method_tag = f"sq{i}"
            for rank, s in enumerate(sources):
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                page = s.get("page", "N/A")
                doc = s.get("document", s.get("source", "Unknown"))
                ranked.append({
                    "text": text,
                    "page": page,
                    "source": doc,
                    "score": float(s.get("score", 0.0)),
                    "retrieval_method": method_tag,
                    "chunk_index": s.get("chunk_index", rank),
                })
            ranked_lists.append(ranked)

        if not ranked_lists:
            return "", []

        if len(ranked_lists) == 1:
            fused = ranked_lists[0][: self.top_k]
        else:
            n = len(ranked_lists)
            fuse_weights = [1.0 / n] * n
            fused = rrf_fuse(
                ranked_lists,
                weights=fuse_weights,
                k=self.rrf_k,
                top_k=self.top_k,
            )

        context_str = ""
        sources_out: List[Any] = []
        for i, r in enumerate(fused):
            text = r.get("text", "")
            page = r.get("page", "N/A")
            source = r.get("source", "Unknown")
            score = r.get("score", 0.0)
            methods = r.get("retrieval_methods", ["unknown"])
            if text.strip():
                context_str += (
                    f"CONTENT FROM SOURCE {i + 1} "
                    f"(Document: {source}, Page: {page}, "
                    f"Score: {score:.4f}, Methods: {', '.join(methods)}):\n"
                    f"{text}\n\n"
                )
                sources_out.append({
                    "id": i + 1,
                    "page": page,
                    "document": source,
                    "score": score,
                    "methods": methods,
                    "text": text,
                })

        logger.info(f"[RAG-Fusion] Merged {len(sub_queries)} sub-queries → {len(sources_out)} chunks")
        return context_str.strip(), sources_out
