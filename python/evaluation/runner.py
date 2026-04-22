"""Run a single question through naive (vector-only) or neuro-symbolic (agent) RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from agent import (
    AgenticRAG,
    AnswerGrader,
    QueryDecomposer,
    QueryRouter,
    RAGFusion,
)
from config import (
    AGENT_CONFIDENCE_THRESHOLD,
    AGENT_DECOMP_MAX_SUBQUERIES,
    AGENT_DECOMP_MIN_WORDS,
    AGENT_FUSION_RRF_K,
    AGENT_MAX_RETRIES,
    AGENT_ROUTER_WEIGHT_BOOST,
    CONTEXT_RETRIEVAL_LIMIT,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    GRAPH_WEIGHT,
)

import local_llm
from local_llm import (
    format_context_for_llm,
    generate_rag_response,
    get_qdrant_client,
    make_retrieve_pipeline,
    retrieve_context,
)
from .judge import sources_to_context_string

ConfigName = Literal["naive", "neurosymbolic"]


@dataclass
class RunResult:
    config: str
    answer: str
    sources: List[Any]
    context_for_judge: str
    trace: Dict[str, Any] = field(default_factory=dict)


def _base_weights() -> tuple:
    return (VECTOR_WEIGHT, BM25_WEIGHT, GRAPH_WEIGHT)


def run_one(
    pdf_id: str,
    collection_name: str,
    query: str,
    chat_history: List[dict],
    config: ConfigName,
) -> RunResult:
    local_llm.init_runtime()
    if config == "naive":
        return _run_naive(pdf_id, collection_name, query, chat_history)
    if config == "neurosymbolic":
        return _run_neurosymbolic(pdf_id, collection_name, query, chat_history)
    raise ValueError(f"Unknown config: {config}")


def _run_naive(
    pdf_id: str,
    collection_name: str,
    query: str,
    chat_history: List[dict],
) -> RunResult:
    client = get_qdrant_client()
    retrieved = retrieve_context(
        client,
        collection_name,
        query,
        pdf_id,
        limit=CONTEXT_RETRIEVAL_LIMIT,
    )
    context_str, sources = format_context_for_llm(retrieved)
    answer = generate_rag_response(query, context_str, chat_history)
    return RunResult(
        config="naive",
        answer=answer,
        sources=sources,
        context_for_judge=context_str or sources_to_context_string(sources),
    )


def _run_neurosymbolic(
    pdf_id: str,
    collection_name: str,
    query: str,
    chat_history: List[dict],
) -> RunResult:
    llm = local_llm.llm
    if not llm:
        raise RuntimeError("local_llm.llm is not initialised; call init_runtime() first")

    retrieve_fn, _ = make_retrieve_pipeline(pdf_id, collection_name)
    decomposer = QueryDecomposer(
        llm,
        max_subqueries=AGENT_DECOMP_MAX_SUBQUERIES,
        min_words=AGENT_DECOMP_MIN_WORDS,
    )
    fusion = RAGFusion(
        retrieve_fn,
        rrf_k=AGENT_FUSION_RRF_K,
        top_k=CONTEXT_RETRIEVAL_LIMIT,
    )
    base_weights = _base_weights()
    agent = AgenticRAG(
        llm=llm,
        router=QueryRouter(llm, weight_boost=AGENT_ROUTER_WEIGHT_BOOST),
        grader=AnswerGrader(llm, threshold=AGENT_CONFIDENCE_THRESHOLD),
        retrieve_fn=retrieve_fn,
        generate_fn=lambda q, ctx, hist: generate_rag_response(q, ctx, hist),
        max_retries=AGENT_MAX_RETRIES,
        base_weights=base_weights,
        decomposer=decomposer,
        fusion=fusion,
    )
    ar = agent.run(query, chat_history)
    context_str = sources_to_context_string(ar.sources)
    return RunResult(
        config="neurosymbolic",
        answer=ar.answer,
        sources=ar.sources,
        context_for_judge=context_str,
        trace=ar.trace,
    )
