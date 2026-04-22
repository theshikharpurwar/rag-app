"""Load hand-authored benchmark datasets (YAML)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import yaml

VALID_TYPES = frozenset({"keyword", "multi_hop", "numerical", "broad"})


@dataclass
class BenchmarkMeta:
    name: str
    pdf_id: str
    fixture_pdf: str
    collection_name: str
    description: str = ""


@dataclass
class BenchmarkQuestion:
    id: str
    type: str
    question: str
    gold_answer: str
    gold_pages: List[int] = field(default_factory=list)


@dataclass
class BenchmarkDataset:
    meta: BenchmarkMeta
    questions: List[BenchmarkQuestion]
    dataset_dir: str


def load_dataset(path: str) -> BenchmarkDataset:
    path = os.path.abspath(path)
    dataset_dir = os.path.dirname(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "meta" not in raw or "questions" not in raw:
        raise ValueError("Dataset must contain 'meta' and 'questions'")

    m = raw["meta"]
    for key in ("name", "pdf_id", "fixture_pdf"):
        if key not in m or not str(m.get(key, "")).strip():
            raise ValueError(f"meta.{key} is required")

    meta = BenchmarkMeta(
        name=str(m["name"]).strip(),
        pdf_id=str(m["pdf_id"]).strip(),
        fixture_pdf=str(m["fixture_pdf"]).strip(),
        collection_name=str(m.get("collection_name") or "documents").strip(),
        description=str(m.get("description") or "").strip(),
    )

    questions: List[BenchmarkQuestion] = []
    for i, q in enumerate(raw["questions"]):
        if not isinstance(q, dict):
            raise ValueError(f"questions[{i}] must be a mapping")
        if "id" not in q or "question" not in q or "gold_answer" not in q:
            raise ValueError(f"questions[{i}] needs id, question, gold_answer")
        qtype = str(q.get("type") or "multi_hop").strip().lower()
        if qtype not in VALID_TYPES:
            raise ValueError(f"questions[{i}].type must be one of {sorted(VALID_TYPES)}")
        gp = q.get("gold_pages")
        pages: List[int] = []
        if gp is not None:
            if not isinstance(gp, list):
                raise ValueError(f"questions[{i}].gold_pages must be a list of integers")
            pages = [int(p) for p in gp]
        questions.append(
            BenchmarkQuestion(
                id=str(q["id"]).strip(),
                type=qtype,
                question=str(q["question"]).strip(),
                gold_answer=str(q["gold_answer"]).strip(),
                gold_pages=pages,
            )
        )

    if not questions:
        raise ValueError("Dataset has no questions")

    return BenchmarkDataset(meta=meta, questions=questions, dataset_dir=dataset_dir)


def resolve_fixture_path(dataset: BenchmarkDataset) -> str:
    return os.path.normpath(
        os.path.join(dataset.dataset_dir, dataset.meta.fixture_pdf)
    )
