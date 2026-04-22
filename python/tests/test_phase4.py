# python/tests/test_phase4.py
"""
Phase 4: evaluation dataset loader, LLM-as-Judge, report aggregation.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --- Fake judge LLM (returns JSON score lines) ---


class FakeJudgeLLM:
    def __init__(self, queue=None):
        self.queue = list(queue or [])
        self.prompts: list = []

    def generate_response(
        self, prompt=None, temperature=0.0, max_tokens=200, messages=None, **kwargs
    ):
        if prompt is not None:
            self.prompts.append(prompt)
        if self.queue:
            return self.queue.pop(0)
        return '{"score": 0.5, "reason": "default"}'


def test_load_dataset(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(
        """
meta:
  name: t
  pdf_id: p1
  fixture_pdf: f.pdf
questions:
  - id: a
    type: keyword
    question: "Q?"
    gold_answer: "A"
  - id: b
    type: multi_hop
    question: "Q2"
    gold_answer: "A2"
    gold_pages: [1, 2]
""",
        encoding="utf-8",
    )
    from evaluation.dataset import load_dataset

    ds = load_dataset(str(p))
    assert ds.meta.name == "t" and ds.meta.pdf_id == "p1"
    assert len(ds.questions) == 2
    assert ds.questions[1].gold_pages == [1, 2]


def test_load_dataset_rejects_bad_type(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
meta:
  name: t
  pdf_id: p1
  fixture_pdf: f.pdf
questions:
  - id: a
    type: invalid_type
    question: "Q"
    gold_answer: "A"
""",
        encoding="utf-8",
    )
    from evaluation.dataset import load_dataset

    with pytest.raises(ValueError, match="type"):
        load_dataset(str(p))


def test_conciseness_is_deterministic():
    from evaluation.judge import EvaluationJudge

    fake = FakeJudgeLLM(
        [
            '{"score": 1, "reason": ""}',
            '{"score": 1, "reason": ""}',
            '{"score": 1, "reason": ""}',
        ]
    )
    j = EvaluationJudge(llm=fake)
    m = j.score_all(
        query="q",
        answer="w " * 9 + "w",
        context="c",
        gold_answer="a b c",
        gold_pages=None,
        sources=[],
    )
    assert m.conciseness == pytest.approx(3.0 / 10.0)


def test_judge_llm_metrics():
    from evaluation.judge import EvaluationJudge

    fake = FakeJudgeLLM(
        [
            '{"score": 0.8, "reason": "f"}',
            '{"score": 0.7, "reason": "r"}',
            '{"score": 0.6, "reason": "c"}',
        ]
    )
    j = EvaluationJudge(llm=fake)
    m = j.score_all(
        query="What is revenue?",
        answer="1.2 million",
        context="revenue 1.2 million",
        gold_answer="2024 revenue 1.2M",
        gold_pages=None,
        sources=[],
    )
    assert abs(m.faithfulness - 0.8) < 1e-6
    assert abs(m.answer_relevancy - 0.7) < 1e-6
    assert abs(m.context_recall - 0.6) < 1e-6


def test_context_recall_page_boost():
    from evaluation.judge import EvaluationJudge

    fake = FakeJudgeLLM(
        [
            '{"score": 0.2, "reason": "poor context"}',
        ]
    )
    j = EvaluationJudge(llm=fake)
    sources = [{"page": 2, "text": "hello"}]
    m = j.score_all(
        query="q",
        answer="a",
        context="c",
        gold_answer="g",
        gold_pages=[2],
        sources=sources,
    )
    assert m.context_recall >= 0.2
    assert m.context_recall >= 0.5


def test_judge_empty_retrieval_short_circuits():
    """When both context and sources are empty, faithfulness & context_recall must be 0."""
    from evaluation.judge import EvaluationJudge

    fake = FakeJudgeLLM(['{"score": 0.9, "reason": "whatever"}'])
    j = EvaluationJudge(llm=fake)
    m = j.score_all(
        query="Who is the CEO?",
        answer="I couldn't find relevant information in the document.",
        context="",
        gold_answer="Jane Doe",
        gold_pages=[1],
        sources=[],
    )
    assert m.faithfulness == 0.0
    assert m.context_recall == 0.0
    # Relevancy still evaluated by the LLM (here: the single queued response).
    assert 0.0 <= m.answer_relevancy <= 1.0
    # Only one LLM call was made (for relevancy), not three.
    assert len(fake.prompts) == 1


def test_fixture_pdf_is_extractable():
    """Guards against silently shipping a PDF that has no extractable text."""
    import fitz

    fixture = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "evaluation", "datasets", "fixture.pdf")
    )
    if not os.path.isfile(fixture):
        pytest.skip("fixture.pdf not present")
    doc = fitz.open(fixture)
    try:
        total = sum(len((p.get_text() or "").strip()) for p in doc)
    finally:
        doc.close()
    assert total > 200, f"fixture.pdf has too little extractable text: {total} chars"


def test_report_summary_wins():
    from evaluation.report import PerQuestionResult, build_summary

    results = [
        PerQuestionResult(
            question_id="q1",
            question_type="keyword",
            config="naive",
            question="Q",
            answer="A1",
            gold_answer="G",
            sources=[],
            scores={
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_recall": 0.5,
                "conciseness": 0.5,
            },
        ),
        PerQuestionResult(
            question_id="q1",
            question_type="keyword",
            config="neurosymbolic",
            question="Q",
            answer="A2",
            gold_answer="G",
            sources=[],
            scores={
                "faithfulness": 0.9,
                "answer_relevancy": 0.9,
                "context_recall": 0.9,
                "conciseness": 0.9,
            },
        ),
    ]
    s = build_summary(results)
    assert s["win_rate"]["neurosymbolic"] == 1
    assert s["win_rate"]["naive"] == 0
    assert "q1" in s["wins_by_question"]


def test_make_retrieve_pipeline_is_importable():
    pytest.importorskip("qdrant_client")
    from local_llm import make_retrieve_pipeline

    assert callable(make_retrieve_pipeline)
