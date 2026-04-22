"""Aggregate scores and write markdown + JSON reports."""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

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
    cfgs = sorted(by_cfg.keys())
    if len(cfgs) == 2:
        a, b = cfgs[0], cfgs[1]
        out["win_rate"] = {a: 0, b: 0, "tie": 0}
        by_q: Dict[str, Dict[str, PerQuestionResult]] = {}
        for r in results:
            by_q.setdefault(r.question_id, {})[r.config] = r
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
