# python/query_server.py
"""
Persistent Python query server for the RAG pipeline.

Replaces the subprocess-per-query model with a long-running Flask
server (port 5001).  Models load once on startup; per-PDF retrieval
pipelines are LRU-cached.

Not used for ingestion — compute_embeddings.py stays as subprocess.
"""

import json
import logging
import os
import sys
from functools import lru_cache

from flask import Flask, Response, jsonify, request, stream_with_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_initialized = False


def _ensure_init():
    """One-time initialization of embedder, reranker, and LLM."""
    global _initialized
    if _initialized:
        return
    from local_llm import init_runtime, check_dependencies

    if not check_dependencies():
        logger.error("Required services not available")
        sys.exit(1)
    init_runtime()
    _initialized = True


@lru_cache(maxsize=8)
def _get_pipeline(pdf_id: str, collection_name: str):
    """
    Build and cache the retrieval pipeline for a given pdf_id.

    LRU(8) keeps the 8 most recently queried PDFs' BM25 index
    and KG graph in memory — subsequent queries skip disk I/O.
    """
    from local_llm import make_retrieve_pipeline

    logger.info(f"[QueryServer] Building pipeline for pdf_id={pdf_id}")
    return make_retrieve_pipeline(pdf_id, collection_name)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/query", methods=["POST"])
def handle_query():
    _ensure_init()

    data = request.get_json(force=True)
    query_text = data.get("query", "").strip()
    pdf_id = data.get("pdf_id", "")
    collection_name = data.get("collection_name", "documents")
    history = data.get("history", [])

    if not query_text or not pdf_id:
        return jsonify({"answer": "Missing query or pdf_id", "sources": []}), 400

    try:
        from local_llm import generate_rag_response, llm
        from config import (
            VECTOR_WEIGHT,
            BM25_WEIGHT,
            GRAPH_WEIGHT,
            CONTEXT_RETRIEVAL_LIMIT,
            ENABLE_AGENT,
            ENABLE_DECOMPOSITION,
            AGENT_MAX_RETRIES,
            AGENT_CONFIDENCE_THRESHOLD,
            AGENT_ROUTER_WEIGHT_BOOST,
            AGENT_DECOMP_MAX_SUBQUERIES,
            AGENT_DECOMP_MIN_WORDS,
            AGENT_FUSION_RRF_K,
        )

        retrieve_fn, _qclient = _get_pipeline(pdf_id, collection_name)
        base_weights = (VECTOR_WEIGHT, BM25_WEIGHT, GRAPH_WEIGHT)

        if ENABLE_AGENT:
            from agent import (
                QueryRouter,
                AnswerGrader,
                AgenticRAG,
                QueryDecomposer,
                RAGFusion,
            )

            logger.info("[QueryServer] Running agentic control loop")
            decomposer = None
            fusion = None
            if ENABLE_DECOMPOSITION:
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
            ar = agent.run(query_text, history)
            logger.info(f"[QueryServer] Agent trace: {ar.trace}")
            result = {"answer": ar.answer, "sources": ar.sources}
        else:
            context_str, sources = retrieve_fn(query_text, base_weights)
            answer = generate_rag_response(query_text, context_str, history)
            result = {"answer": answer, "sources": sources}

        return jsonify(result)

    except Exception as e:
        logger.error(f"[QueryServer] Error: {e}", exc_info=True)
        return (
            jsonify(
                {
                    "answer": "An error occurred while processing your question. Please try again.",
                    "sources": [],
                }
            ),
            500,
        )


def _format_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _phase_event(phase: str, detail: str | None = None) -> dict:
    event = {"phase": phase}
    if detail:
        event["detail"] = detail
    return event


@app.route("/query/stream", methods=["POST"])
def handle_query_stream():
    _ensure_init()

    data = request.get_json(force=True)
    query_text = (data or {}).get("query", "").strip()
    pdf_id = (data or {}).get("pdf_id", "")
    collection_name = (data or {}).get("collection_name", "documents")
    history = (data or {}).get("history", [])

    if not query_text or not pdf_id:
        return jsonify({"message": "Missing query or pdf_id"}), 400

    @stream_with_context
    def event_stream():
        try:
            from local_llm import (
                _postprocess_rag_answer,
                generate_rag_response,
                generate_rag_response_stream,
                llm,
            )
            from config import (
                VECTOR_WEIGHT,
                BM25_WEIGHT,
                GRAPH_WEIGHT,
                CONTEXT_RETRIEVAL_LIMIT,
                ENABLE_AGENT,
                ENABLE_DECOMPOSITION,
                AGENT_MAX_RETRIES,
                AGENT_CONFIDENCE_THRESHOLD,
                AGENT_ROUTER_WEIGHT_BOOST,
                AGENT_DECOMP_MAX_SUBQUERIES,
                AGENT_DECOMP_MIN_WORDS,
                AGENT_FUSION_RRF_K,
            )

            retrieve_fn, _qclient = _get_pipeline(pdf_id, collection_name)
            base_weights = (VECTOR_WEIGHT, BM25_WEIGHT, GRAPH_WEIGHT)

            phase_queue: list[dict] = []

            def enqueue_phase(phase: str, detail: str | None = None):
                phase_queue.append(_phase_event(phase, detail))

            def drain_phases():
                while phase_queue:
                    yield _format_sse(phase_queue.pop(0))

            if ENABLE_AGENT:
                from agent import (
                    QueryRouter,
                    AnswerGrader,
                    AgenticRAG,
                    QueryDecomposer,
                    RAGFusion,
                )

                decomposer = None
                fusion = None
                if ENABLE_DECOMPOSITION:
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
                agent = AgenticRAG(
                    llm=llm,
                    router=QueryRouter(llm, weight_boost=AGENT_ROUTER_WEIGHT_BOOST),
                    grader=AnswerGrader(
                        llm, threshold=AGENT_CONFIDENCE_THRESHOLD
                    ),
                    retrieve_fn=retrieve_fn,
                    generate_fn=lambda q, ctx, hist: generate_rag_response(
                        q, ctx, hist
                    ),
                    max_retries=AGENT_MAX_RETRIES,
                    base_weights=base_weights,
                    decomposer=decomposer,
                    fusion=fusion,
                )
                for ev in agent.run_stream(
                    query_text,
                    history,
                    lambda q, c, h: generate_rag_response_stream(q, c, h),
                    on_phase=enqueue_phase,
                ):
                    yield from drain_phases()
                    if ev.get("done"):
                        ev = dict(ev)
                        ev["answer"] = _postprocess_rag_answer(
                            ev.pop("raw_answer", "")
                        )
                        logger.info(
                            "[QueryServer] Agent trace: %s", ev.get("trace")
                        )
                    yield _format_sse(ev)
            else:
                yield _format_sse({"phase": "retrieving"})
                context_str, sources = retrieve_fn(
                    query_text, base_weights, None, enqueue_phase
                )
                yield from drain_phases()
                yield _format_sse({"phase": "generating"})
                acc = []
                for token in generate_rag_response_stream(
                    query_text, context_str, history
                ):
                    acc.append(token)
                    yield _format_sse({"token": token})
                full = "".join(acc)
                yield _format_sse(
                    {
                        "done": True,
                        "sources": sources,
                        "trace": {},
                        "answer": _postprocess_rag_answer(full),
                    }
                )
        except Exception as e:
            logger.error(
                f"[QueryServer] stream error: {e}", exc_info=True
            )
            yield _format_sse(
                {
                    "error": "An error occurred while processing your question. Please try again.",
                }
            )

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear the pipeline cache (e.g., after re-ingesting a PDF)."""
    _get_pipeline.cache_clear()
    logger.info("[QueryServer] Pipeline cache cleared")
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PYTHON_QUERY_PORT", "5001"))
    logger.info(f"[QueryServer] Starting on port {port}")
    _ensure_init()
    app.run(host="0.0.0.0", port=port, threaded=True)
