# python/reranker/colbert_reranker.py
"""
ColBERT-v2 late-interaction reranker via ragatouille.

Used when RERANKER_TYPE=colbert. Requires ``pip install ragatouille``.
"""

import logging

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """
    Reranker using ColBERT-v2 token-level MaxSim scoring.

    Same interface as SimpleReranker: rerank(query, documents, top_k) -> list
    """

    def __init__(self, model_name=None):
        from config.models import COLBERT_MODEL
        from ragatouille import RAGPretrainedModel

        self.model_name = model_name or COLBERT_MODEL
        logger.info(f"Loading ColBERT reranker: {self.model_name}")
        self.model = RAGPretrainedModel.from_pretrained(self.model_name)
        logger.info("ColBERT reranker loaded successfully.")

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
        if len(documents) <= top_k:
            return documents

        try:
            texts = []
            for doc in documents:
                if hasattr(doc, "payload") and isinstance(doc.payload, dict):
                    texts.append(doc.payload.get("text", ""))
                elif isinstance(doc, dict):
                    texts.append(doc.get("text", ""))
                else:
                    texts.append(str(doc))

            results = self.model.rerank(
                query=query,
                documents=texts,
                k=min(top_k, len(texts)),
            )

            used = [False] * len(documents)
            reranked = []
            for r in results:
                content = r.get("content", "")
                idx = None
                for i, t in enumerate(texts):
                    if used[i]:
                        continue
                    if t == content:
                        idx = i
                        break
                if idx is None:
                    pref = content[:200] if content else ""
                    for i, t in enumerate(texts):
                        if used[i]:
                            continue
                        if t[:200] == pref:
                            idx = i
                            break
                if idx is None and content:
                    for i, t in enumerate(texts):
                        if used[i]:
                            continue
                        if t.startswith(content[: min(len(t), len(content))]) or content.startswith(
                            t[: min(200, len(t))]
                        ):
                            idx = i
                            break
                if idx is not None:
                    reranked.append(documents[idx])
                    used[idx] = True

            if len(reranked) < min(top_k, len(documents)):
                for i, doc in enumerate(documents):
                    if not used[i] and len(reranked) < top_k:
                        reranked.append(doc)
                        used[i] = True

            if not reranked:
                logger.warning("[ColBERT] Could not map results back, returning original order")
                return documents[:top_k]

            top_score = results[0]["score"] if results else 0
            bottom_score = results[-1]["score"] if results else 0
            logger.info(
                f"[ColBERT] Reranked {len(documents)} → {len(reranked)}. "
                f"Top: {top_score:.3f}, Bottom: {bottom_score:.3f}"
            )
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"[ColBERT] Reranking failed: {e}", exc_info=True)
            return documents[:top_k]
