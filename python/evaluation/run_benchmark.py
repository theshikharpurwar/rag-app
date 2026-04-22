"""
CLI: run naive vs neuro-symbolic benchmark, LLM-as-Judge scores, write reports.

Diagnostics go to stderr; result files are written to disk (not stdout), so this
is safe to run as a script without corrupting JSON consumers.

Usage (from `python/`):
  python -m evaluation.run_benchmark --dataset evaluation/datasets/sample.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime, timezone

# Ensure `python/` is on sys.path when run as a script
_PARDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARDIR not in sys.path:
    sys.path.insert(0, _PARDIR)

import local_llm
from config import (
    DEFAULT_COLLECTION,
    EVAL_MAX_CONCURRENCY,
    EVAL_OUTPUT_DIR,
)

from .dataset import load_dataset, resolve_fixture_path
from .judge import EvaluationJudge
from .report import PerQuestionResult, write_reports
from .runner import run_one


def _ingest_fixture(pdf_path: str, pdf_id: str, collection_name: str) -> None:
    script = os.path.join(_PARDIR, "compute_embeddings.py")
    if not os.path.isfile(script):
        print(f"compute_embeddings.py not found at {script}", file=sys.stderr)
        sys.exit(1)
    env = os.environ.copy()
    cmd = [
        sys.executable,
        script,
        pdf_path,
        "--pdf_id",
        pdf_id,
        "--collection_name",
        collection_name,
    ]
    print(f"[Eval] Ingest: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(
        cmd,
        cwd=_PARDIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print("[Eval] Ingest failed.", file=sys.stderr)
        sys.exit(1)
    out = (proc.stdout or "").strip()
    if not out:
        print("[Eval] Ingest returned empty stdout", file=sys.stderr)
        sys.exit(1)
    try:
        meta = json.loads(out)
    except json.JSONDecodeError as e:
        print(f"[Eval] Ingest JSON parse error: {e}\n{out[:500]}", file=sys.stderr)
        sys.exit(1)
    if not meta.get("success"):
        print(f"[Eval] Ingest error: {meta.get('error', meta)}", file=sys.stderr)
        sys.exit(1)
    print(
        f"[Eval] Ingest ok: {meta.get('embeddings_count', '?')} chunks, "
        f"pages {meta.get('page_count', '?')}",
        file=sys.stderr,
    )


def _parse_configs(s: str) -> list:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    valid = {"naive", "neurosymbolic"}
    for p in parts:
        if p not in valid:
            print(f"Unknown config {p!r}; expected subset of {valid}", file=sys.stderr)
            sys.exit(1)
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 evaluation benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/datasets/sample.yaml",
        help="Path to YAML dataset (relative to cwd or absolute)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=EVAL_OUTPUT_DIR,
        help="Output directory for JSON + markdown reports",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="naive,neurosymbolic",
        help="Comma-separated: naive, neurosymbolic",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Do not re-run compute_embeddings (assume pdf_id already in Qdrant)",
    )
    args = parser.parse_args()

    if EVAL_MAX_CONCURRENCY != 1:
        print(
            f"[Eval] EVAL_MAX_CONCURRENCY={EVAL_MAX_CONCURRENCY} (only 1 is used for now)",
            file=sys.stderr,
        )

    ds_path = os.path.abspath(args.dataset)
    if not os.path.isfile(ds_path):
        print(f"Dataset not found: {ds_path}", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(ds_path)
    fixture = resolve_fixture_path(dataset)
    if not os.path.isfile(fixture):
        print(f"Fixture PDF not found: {fixture}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_ingest:
        _ingest_fixture(
            fixture,
            dataset.meta.pdf_id,
            dataset.meta.collection_name or DEFAULT_COLLECTION,
        )

    local_llm.init_runtime()
    if not local_llm.embedder or not local_llm.llm:
        print("[Eval] Runtime failed to init embedder/llm", file=sys.stderr)
        sys.exit(1)

    judge = EvaluationJudge()
    configs = _parse_configs(args.configs)
    chat_history: list = []

    results: list = []
    stem = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}Z"
    empty_runs: list = []

    for q in dataset.questions:
        for cfg in configs:
            print(f"[Eval] {q.id} / {cfg} ...", file=sys.stderr)
            rr = run_one(
                dataset.meta.pdf_id,
                dataset.meta.collection_name or DEFAULT_COLLECTION,
                q.question,
                chat_history,
                cfg,  # type: ignore[arg-type]
            )
            if not rr.sources:
                print(
                    f"[Eval] WARNING: 0 sources retrieved for {q.id} ({cfg}). "
                    "Check that the fixture was ingested for pdf_id="
                    f"{dataset.meta.pdf_id!r}.",
                    file=sys.stderr,
                )
                empty_runs.append((q.id, cfg))
            m = judge.score_all(
                query=q.question,
                answer=rr.answer,
                context=rr.context_for_judge,
                gold_answer=q.gold_answer,
                gold_pages=q.gold_pages or None,
                sources=rr.sources,
            )
            results.append(
                PerQuestionResult(
                    question_id=q.id,
                    question_type=q.type,
                    config=cfg,
                    question=q.question,
                    answer=rr.answer,
                    gold_answer=q.gold_answer,
                    sources=rr.sources,
                    scores=m.as_dict(),
                    trace=rr.trace,
                )
            )

    out_dir = os.path.abspath(args.output)
    jpath, mpath = write_reports(out_dir, stem, dataset.meta, results)
    print(f"[Eval] Wrote {jpath}", file=sys.stderr)
    print(f"[Eval] Wrote {mpath}", file=sys.stderr)

    if empty_runs:
        total = len(dataset.questions) * len(configs)
        print(
            f"[Eval] WARNING: {len(empty_runs)}/{total} runs returned 0 sources. "
            "Verify Qdrant + ingest: "
            f"curl 'http://localhost:6333/collections/{dataset.meta.collection_name or DEFAULT_COLLECTION}/points/scroll' "
            "-H 'Content-Type: application/json' "
            f"-d '{{\"filter\":{{\"must\":[{{\"key\":\"pdf_id\",\"match\":{{\"value\":\"{dataset.meta.pdf_id}\"}}}}]}},\"limit\":1}}'",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
