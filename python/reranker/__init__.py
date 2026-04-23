"""Reranker module — cross-encoder (default) or ColBERT late-interaction."""

import logging

from .simple_reranker import SimpleReranker

logger = logging.getLogger(__name__)

__all__ = ["SimpleReranker", "get_reranker"]


def get_reranker():
    """
    Return the configured reranker.

    RERANKER_TYPE (config): ``cross_encoder`` (default) or ``colbert``.
    ColBERT falls back to the cross-encoder if ragatouille is missing or init fails.
    """
    from config.models import RERANKER_TYPE

    rtype = (RERANKER_TYPE or "cross_encoder").lower().strip()
    if rtype == "colbert":
        try:
            from .colbert_reranker import ColBERTReranker

            return ColBERTReranker()
        except ImportError:
            logger.warning(
                "[Reranker] ragatouille not installed, falling back to cross-encoder. "
                "Install with: pip install ragatouille"
            )
        except Exception as e:
            logger.warning(
                "[Reranker] ColBERT init failed (%s), falling back to cross-encoder",
                e,
            )

    return SimpleReranker()
