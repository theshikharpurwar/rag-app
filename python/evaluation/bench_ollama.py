"""
Benchmark Ollama generation and embedding throughput (standalone CLI; stdout OK).

Usage:
  cd python && python -m evaluation.bench_ollama gen --model gemma3:4b --prompts 20 --concurrency 2
  cd python && python -m evaluation.bench_ollama embed --model nomic-embed-text-v2-moe --items 200 --batch 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests


def _percentile_ns(sorted_ns: list[float], p: float) -> float | None:
    if not sorted_ns:
        return None
    xs = sorted(sorted_ns)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (k - f) * (xs[c] - xs[f])


def _api_base(host: str) -> str:
    base = host.rstrip("/")
    if base.endswith("/api"):
        return base
    return f"{base}/api"


def _one_chat_stream(
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    num_batch: int,
    num_ctx: int | None,
    keep_alive: str,
    timeout: int,
) -> tuple[float, float | None]:
    """Returns (wall_seconds, tok_per_s from eval_count/eval_duration or None)."""
    options: dict[str, Any] = {
        "num_predict": max_tokens,
        "temperature": temperature,
        "num_batch": num_batch,
    }
    if num_ctx is not None and num_ctx > 0:
        options["num_ctx"] = num_ctx
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": options,
        "keep_alive": keep_alive,
    }
    t0 = time.perf_counter()
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    r = requests.post(f"{api_base}/chat", json=payload, timeout=timeout, stream=True)
    try:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done"):
                eval_count = chunk.get("eval_count")
                eval_duration_ns = chunk.get("eval_duration")
                break
    finally:
        r.close()
    wall = time.perf_counter() - t0
    tok_per_s: float | None = None
    if eval_count and eval_duration_ns and eval_duration_ns > 0:
        tok_per_s = float(eval_count) / (float(eval_duration_ns) / 1e9)
    return wall, tok_per_s


def cmd_gen(args: argparse.Namespace) -> int:
    api = _api_base(args.host)
    num_ctx = args.num_ctx if args.num_ctx and args.num_ctx > 0 else None
    keep_alive = args.keep_alive
    num_batch = args.num_batch
    prompt = args.prompt

    def run_one(_i: int) -> tuple[float, float | None]:
        return _one_chat_stream(
            api,
            args.model,
            prompt,
            args.max_tokens,
            args.temperature,
            num_batch,
            num_ctx,
            keep_alive,
            args.timeout,
        )

    # Sequential
    seq_lat: list[float] = []
    seq_tok: list[float] = []
    wall_seq_start = time.perf_counter()
    for i in range(args.prompts):
        lat, tps = run_one(i)
        seq_lat.append(lat)
        if tps is not None:
            seq_tok.append(tps)
    wall_seq = time.perf_counter() - wall_seq_start

    # Concurrent (same total count)
    conc_lat: list[float] = []
    conc_tok: list[float] = []
    wall_conc_start = time.perf_counter()
    k = max(1, args.concurrency)
    with ThreadPoolExecutor(max_workers=k) as ex:
        futs = [ex.submit(run_one, i) for i in range(args.prompts)]
        for fut in as_completed(futs):
            lat, tps = fut.result()
            conc_lat.append(lat)
            if tps is not None:
                conc_tok.append(tps)
    wall_conc = time.perf_counter() - wall_conc_start

    def block(name: str, latencies: list[float], tok_rates: list[float], wall: float) -> dict[str, Any]:
        p50 = (_percentile_ns(latencies, 50.0) or 0.0) if latencies else None
        p95 = _percentile_ns(latencies, 95.0) if latencies else None
        med_tok = None
        if tok_rates:
            med_tok = _percentile_ns(tok_rates, 50.0)
        return {
            "phase": name,
            "requests": len(latencies),
            "wall_s": wall,
            "latency_p50_s": p50,
            "latency_p95_s": p95,
            "median_tok_per_s": med_tok,
        }

    out = {
        "command": "gen",
        "model": args.model,
        "host": args.host,
        "num_batch": num_batch,
        "num_ctx": num_ctx,
        "keep_alive": keep_alive,
        "sequential": block("sequential", seq_lat, seq_tok, wall_seq),
        "concurrent": block("concurrent", conc_lat, conc_tok, wall_conc),
        "concurrency": k,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"bench_ollama gen  model={args.model}  requests={args.prompts}  concurrency={k}")
    for phase_key in ("sequential", "concurrent"):
        b = out[phase_key]
        tps = b["median_tok_per_s"]
        tps_s = f"{tps:.1f}" if tps is not None else "n/a"
        print(
            f"  {b['phase']}: wall={b['wall_s']:.2f}s  "
            f"p50={b['latency_p50_s']:.3f}s  p95={b['latency_p95_s']:.3f}s  "
            f"median tok/s={tps_s}"
        )
    return 0


def _one_embed_batch(
    api_base: str,
    model: str,
    batch: list[str],
    keep_alive: str,
    timeout: int,
) -> float:
    t0 = time.perf_counter()
    r = requests.post(
        f"{api_base}/embed",
        json={
            "model": model,
            "input": batch,
            "keep_alive": keep_alive,
            "options": {"task": "search_document"},
        },
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    return time.perf_counter() - t0


def cmd_embed(args: argparse.Namespace) -> int:
    api = _api_base(args.host)
    keep_alive = args.keep_alive
    texts = [args.text_template % i for i in range(args.items)]
    latencies: list[float] = []
    wall_start = time.perf_counter()
    for i in range(0, len(texts), args.batch):
        batch = texts[i : i + args.batch]
        timeout = max(30, 5 * len(batch))
        try:
            lat = _one_embed_batch(api, args.model, batch, keep_alive, timeout)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print("embed: /api/embed returned 404; use a newer Ollama or embed via legacy path.", file=sys.stderr)
                return 2
            raise
        latencies.append(lat)
    wall = time.perf_counter() - wall_start
    p50 = _percentile_ns(latencies, 50.0)
    p95 = _percentile_ns(latencies, 95.0)
    items_per_s = float(args.items) / wall if wall > 0 else None
    out = {
        "command": "embed",
        "model": args.model,
        "host": args.host,
        "items": args.items,
        "batch": args.batch,
        "keep_alive": keep_alive,
        "requests": len(latencies),
        "wall_s": wall,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "items_per_s": items_per_s,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(
        f"bench_ollama embed  model={args.model}  items={args.items}  batch={args.batch}  "
        f"requests={len(latencies)}"
    )
    print(
        f"  wall={wall:.2f}s  items/s={items_per_s:.1f}  "
        f"req p50={p50:.3f}s  req p95={p95:.3f}s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark Ollama chat generation and embeddings.")
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    p.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"),
        help="Ollama base URL (default OLLAMA_HOST_URL or localhost:11434)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="Benchmark /api/chat streaming generation")
    g.add_argument("--model", default=os.environ.get("LLM_MODEL", "gemma3:4b"))
    g.add_argument("--prompts", type=int, default=10)
    g.add_argument("--concurrency", type=int, default=1)
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.2)
    g.add_argument("--prompt", default="Reply with one short sentence confirming you are running.")
    g.add_argument("--num-batch", type=int, default=int(os.environ.get("OLLAMA_NUM_BATCH", "512")))
    g.add_argument("--num-ctx", type=int, default=int(os.environ.get("OLLAMA_NUM_CTX", "0")))
    g.add_argument("--keep-alive", default=os.environ.get("OLLAMA_KEEP_ALIVE", "5m"))
    g.add_argument("--timeout", type=int, default=600)
    g.set_defaults(func=cmd_gen)

    e = sub.add_parser("embed", help="Benchmark /api/embed batched requests")
    e.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", "nomic-embed-text-v2-moe"))
    e.add_argument("--items", type=int, default=100)
    e.add_argument("--batch", type=int, default=int(os.environ.get("EMBED_BATCH_SIZE", "32")))
    e.add_argument("--text-template", default="bench embed token %s for throughput measurement.")
    e.add_argument("--keep-alive", default=os.environ.get("OLLAMA_KEEP_ALIVE", "5m"))
    e.set_defaults(func=cmd_embed)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
