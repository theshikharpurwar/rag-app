"""Aggregate scores and write markdown + JSON reports."""

from __future__ import annotations

import json
import os
import statistics
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from .dataset import BenchmarkMeta


@dataclass
class PerQuestionResult:
    question_id: str
    question_type: str
    config: str
    question: str
    answer: str
    gold_answer: str
    sources: List[Any]
    scores: Dict[str, float]
    trace: Dict[str, Any] = field(default_factory=dict)


def _mean_stdev(values: List[float]) -> tuple:
    if not values:
        return 0.0, 0.0
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, 0.0
    return m, statistics.stdev(values)


def _avg_score(r: PerQuestionResult) -> float:
    s = r.scores
    vals = [
        s.get("faithfulness", 0.0),
        s.get("answer_relevancy", 0.0),
        s.get("context_recall", 0.0),
        s.get("conciseness", 0.0),
    ]
    return sum(vals) / max(1, len(vals))


def _paired_metric_vectors(
    by_q: Dict[str, Dict[str, PerQuestionResult]],
    cfg_a: str,
    cfg_b: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    a_vals: list[float] = []
    b_vals: list[float] = []
    for qid in sorted(by_q.keys()):
        mp = by_q[qid]
        if cfg_a not in mp or cfg_b not in mp:
            continue
        a_vals.append(float(mp[cfg_a].scores.get(metric, 0.0)))
        b_vals.append(float(mp[cfg_b].scores.get(metric, 0.0)))
    return a_vals, b_vals


def _wilcoxon_pvalue_and_statistic(
    a_vals: list[float], b_vals: list[float]
) -> tuple[float, float]:
    if len(a_vals) < 2:
        return 1.0, 0.0
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return 1.0, 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = wilcoxon(
                np.array(b_vals, dtype=float),
                np.array(a_vals, dtype=float),
                zero_method="wilcox",
                alternative="two-sided",
            )
    except ValueError:
        return 1.0, 0.0
    p_value = float(res.pvalue)
    statistic = float(res.statistic)
    if not np.isfinite(p_value) or not np.isfinite(statistic):
        return 1.0, 0.0
    return p_value, statistic


def _bootstrap_mean_ci(deltas: list[float]) -> tuple[float, float]:
    if len(deltas) < 2:
        d = float(deltas[0]) if deltas else 0.0
        return d, d
    if all(abs(x - deltas[0]) < 1e-12 for x in deltas):
        d = float(deltas[0])
        return d, d
    try:
        from scipy.stats import bootstrap
    except Exception:
        mean = float(np.mean(np.array(deltas, dtype=float)))
        return mean, mean
    arr = np.array(deltas, dtype=float)
    try:
        ci = bootstrap(
            (arr,),
            np.mean,
            n_resamples=1000,
            confidence_level=0.95,
            method="BCa",
            random_state=42,
        ).confidence_interval
    except Exception:
        mean = float(np.mean(arr))
        return mean, mean
    return float(ci.low), float(ci.high)


def build_summary(
    results: List[PerQuestionResult],
) -> Dict[str, Any]:
    by_cfg: Dict[str, List[PerQuestionResult]] = {}
    for r in results:
        by_cfg.setdefault(r.config, []).append(r)

    metrics = ("faithfulness", "answer_relevancy", "context_recall", "conciseness")
    out: Dict[str, Any] = {"configs": {}}

    for cfg, rows in by_cfg.items():
        per_m: Dict[str, Any] = {}
        for m in metrics:
            vals = [x.scores.get(m, 0.0) for x in rows]
            mean, sd = _mean_stdev(vals)
            per_m[m] = {"mean": mean, "stdev": sd}
        out["configs"][cfg] = {"metrics": per_m, "n": len(rows)}

    wins: Dict[str, Any] = {}
    by_q: Dict[str, Dict[str, PerQuestionResult]] = {}
    cfgs = sorted(by_cfg.keys())
    for r in results:
        by_q.setdefault(r.question_id, {})[r.config] = r

    if len(cfgs) == 2:
        a, b = cfgs[0], cfgs[1]
        out["win_rate"] = {a: 0, b: 0, "tie": 0}
        for _qid, mp in by_q.items():
            if a not in mp or b not in mp:
                continue
            s1 = _avg_score(mp[a])
            s2 = _avg_score(mp[b])
            eps = 1e-9
            if abs(s1 - s2) < eps:
                out["win_rate"]["tie"] += 1
                w = "tie"
            elif s1 > s2:
                out["win_rate"][a] += 1
                w = a
            else:
                out["win_rate"][b] += 1
                w = b
            wins[_qid] = {"winner": w, a: s1, b: s2}

        small_n_threshold = 20
        significance: Dict[str, Any] = {}
        paired_counts: list[int] = []
        for metric in metrics:
            a_vals, b_vals = _paired_metric_vectors(by_q, a, b, metric)
            n_paired = len(a_vals)
            paired_counts.append(n_paired)
            deltas = [bv - av for av, bv in zip(a_vals, b_vals)]
            delta = float(np.mean(np.array(deltas, dtype=float))) if deltas else 0.0
            p_value, statistic = _wilcoxon_pvalue_and_statistic(a_vals, b_vals)
            ci_low, ci_high = _bootstrap_mean_ci(deltas)

            effect = "equal"
            if delta > 1e-9:
                effect = f"{b}_higher"
            elif delta < -1e-9:
                effect = f"{a}_higher"

            if abs(delta) < 1e-9:
                verdict = "tie"
            elif p_value < 0.05 and n_paired >= small_n_threshold:
                verdict = "significant"
            else:
                verdict = "directional"

            significance[metric] = {
                "n_paired": n_paired,
                "delta": delta,
                "p_value": p_value,
                "statistic": statistic,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "verdict": verdict,
                "effect": effect,
            }
        out["significance"] = significance
        out["significance_meta"] = {
            "n_paired_min": min(paired_counts) if paired_counts else 0,
            "small_n_threshold": small_n_threshold,
            "pair": {"a": a, "b": b},
        }
    out["wins_by_question"] = wins
    return out


def render_markdown(
    meta: BenchmarkMeta,
    results: List[PerQuestionResult],
    summary: Dict[str, Any],
) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Dataset:** {meta.name} (`pdf_id`={meta.pdf_id})",
        f"**Description:** {meta.description}",
        "",
        "## Per-metric means (± stdev)",
        "",
    ]
    for cfg, block in summary.get("configs", {}).items():
        lines.append(f"### {cfg}")
        lines.append("")
        lines.append("| Metric | Mean | Stdev |")
        lines.append("|--------|------|-------|")
        for m, stat in block.get("metrics", {}).items():
            lines.append(
                f"| {m} | {stat.get('mean', 0):.3f} | {stat.get('stdev', 0):.3f} |"
            )
        lines.append("")

    wr = summary.get("win_rate")
    if wr:
        lines.append("## Win rate (higher mean of four metrics per question)")
        lines.append("")
        for k, v in wr.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    significance = summary.get("significance")
    if significance:
        meta_sig = summary.get("significance_meta", {})
        pair = meta_sig.get("pair", {})
        a = pair.get("a", "config_a")
        b = pair.get("b", "config_b")
        lines.append("## Statistical significance (paired Wilcoxon, bootstrap 95% CI)")
        lines.append("")
        if int(meta_sig.get("n_paired_min", 0)) < int(
            meta_sig.get("small_n_threshold", 20)
        ):
            lines.append(
                "_Sample size is below 20 paired questions; treat p-values as "
                "**directional** evidence._"
            )
            lines.append("")
        lines.append(f"| Metric | Δ ({b} - {a}) | 95% CI (BCa) | p (Wilcoxon) | Verdict |")
        lines.append("|--------|----------------|--------------|--------------|---------|")
        for metric, stat in significance.items():
            lines.append(
                f"| {metric} | "
                f"{stat.get('delta', 0.0):+.3f} | "
                f"[{stat.get('ci_low', 0.0):+.3f}, {stat.get('ci_high', 0.0):+.3f}] | "
                f"{stat.get('p_value', 1.0):.3f} | "
                f"{stat.get('verdict', 'directional')} |"
            )
        lines.append("")

    lines.append("## Sample Q&A (up to 3 per config)")
    lines.append("")

    by_c: Dict[str, List[PerQuestionResult]] = {}
    for r in results:
        by_c.setdefault(r.config, []).append(r)
    for cfg, rows in by_c.items():
        lines.append(f"### {cfg}")
        for r in rows[:3]:
            qshort = r.question if len(r.question) <= 200 else r.question[:200] + "..."
            ashort = r.answer if len(r.answer) <= 500 else r.answer[:500] + "..."
            lines.append(f"- **Q [{r.question_id}]:** {qshort}")
            lines.append(f"  - **A:** {ashort}")
        lines.append("")

    return "\n".join(lines)


def results_to_json_serializable(
    meta: BenchmarkMeta,
    results: List[PerQuestionResult],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "meta": {
            "name": meta.name,
            "pdf_id": meta.pdf_id,
            "description": meta.description,
        },
        "summary": summary,
        "results": [asdict(r) for r in results],
    }


def write_reports(
    out_dir: str,
    stem: str,
    meta: BenchmarkMeta,
    results: List[PerQuestionResult],
) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    summary = build_summary(results)
    jpath = os.path.join(out_dir, f"{stem}_results.json")
    mpath = os.path.join(out_dir, f"{stem}_report.md")
    payload = results_to_json_serializable(meta, results, summary)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(render_markdown(meta, results, summary))
    return jpath, mpath
