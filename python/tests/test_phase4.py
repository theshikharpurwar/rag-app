# python/tests/test_phase4.py
"""
Phase 4: evaluation dataset loader, LLM-as-Judge, report aggregation.
"""
import os
import sys
import time

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


class PromptKeyedJudgeLLM:
    """Returns scores keyed by judge prompt type (order-safe under parallel calls)."""

    def __init__(self):
        self.prompts: list = []

    def generate_response(
        self, prompt=None, temperature=0.0, max_tokens=200, messages=None, **kwargs
    ):
        if prompt is not None:
            self.prompts.append(prompt)
        p = prompt or ""
        if "grounded in the CONTEXT (faithfulness)" in p:
            return '{"score": 0.8, "reason": "f"}'
        if "addresses the QUESTION (relevancy)" in p:
            return '{"score": 0.7, "reason": "r"}'
        if "support the GOLD reference answer (recall)" in p:
            return '{"score": 0.6, "reason": "cr"}'
        return '{"score": 0.5, "reason": "default"}'


class SlowJudgeLLM:
    def __init__(self, delay_s: float = 0.2):
        self.delay_s = delay_s

    def generate_response(
        self, prompt=None, temperature=0.0, max_tokens=200, messages=None, **kwargs
    ):
        time.sleep(self.delay_s)
        return '{"score": 0.5, "reason": "x"}'


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
    assert ds.meta.distractor_pdfs == []


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


def test_load_dataset_with_distractors(tmp_path):
    p = tmp_path / "d_with_distractors.yaml"
    p.write_text(
        """
meta:
  name: t
  pdf_id: p1
  fixture_pdf: f.pdf
  distractor_pdfs:
    - d1.pdf
    - nested/d2.pdf
questions:
  - id: a
    type: keyword
    question: "Q?"
    gold_answer: "A"
""",
        encoding="utf-8",
    )
    from evaluation.dataset import load_dataset, resolve_distractor_paths

    ds = load_dataset(str(p))
    assert ds.meta.distractor_pdfs == ["d1.pdf", "nested/d2.pdf"]
    resolved = resolve_distractor_paths(ds)
    assert resolved == [
        os.path.normpath(os.path.join(str(tmp_path), "d1.pdf")),
        os.path.normpath(os.path.join(str(tmp_path), "nested/d2.pdf")),
    ]


def test_load_dataset_rejects_non_list_distractors(tmp_path):
    p = tmp_path / "bad_distractors.yaml"
    p.write_text(
        """
meta:
  name: t
  pdf_id: p1
  fixture_pdf: f.pdf
  distractor_pdfs: d1.pdf
questions:
  - id: a
    type: keyword
    question: "Q?"
    gold_answer: "A"
""",
        encoding="utf-8",
    )
    from evaluation.dataset import load_dataset

    with pytest.raises(ValueError, match="distractor_pdfs"):
        load_dataset(str(p))


def test_ingest_targets_include_fixture_and_distractors(tmp_path):
    from evaluation.dataset import BenchmarkDataset, BenchmarkMeta
    from evaluation.run_benchmark import _iter_ingest_targets

    ds = BenchmarkDataset(
        meta=BenchmarkMeta(
            name="harder",
            pdf_id="eval_primary",
            fixture_pdf="fixture.pdf",
            distractor_pdfs=["d1.pdf", "nested/d2.pdf"],
            collection_name="documents",
            description="",
        ),
        questions=[],
        dataset_dir=str(tmp_path),
    )

    targets = _iter_ingest_targets(ds)
    assert targets[0] == (os.path.normpath(os.path.join(str(tmp_path), "fixture.pdf")), "eval_primary")
    assert targets[1][0] == os.path.normpath(os.path.join(str(tmp_path), "d1.pdf"))
    assert targets[2][0] == os.path.normpath(os.path.join(str(tmp_path), "nested/d2.pdf"))
    assert targets[1][1].startswith("eval_primary__distractor_1_")
    assert targets[2][1].startswith("eval_primary__distractor_2_")


def test_ingest_fixture_without_reset_omits_flag(monkeypatch, tmp_path):
    import evaluation.run_benchmark as rb

    payload = '{"success": true, "embeddings_count": 1, "page_count": 1}'
    captured = {}

    class FakeCompleted:
        returncode = 0
        stdout = payload
        stderr = ""

    def fake_run(cmd, cwd, env, capture_output, text):
        captured["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr(
        rb.os.path, "isfile", lambda p: str(p).endswith("compute_embeddings.py")
    )
    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    rb._ingest_fixture("fixture.pdf", "pdf_1", "documents", reset=False)
    assert "--reset" not in captured["cmd"]


def test_ingest_fixture_with_reset_adds_flag(monkeypatch, tmp_path):
    import evaluation.run_benchmark as rb

    payload = '{"success": true, "embeddings_count": 1, "page_count": 1}'
    captured = {}

    class FakeCompleted:
        returncode = 0
        stdout = payload
        stderr = ""

    def fake_run(cmd, cwd, env, capture_output, text):
        captured["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr(
        rb.os.path, "isfile", lambda p: str(p).endswith("compute_embeddings.py")
    )
    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    rb._ingest_fixture("fixture.pdf", "pdf_1", "documents", reset=True)
    assert "--reset" in captured["cmd"]


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

    fake = PromptKeyedJudgeLLM()
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


def test_judge_score_all_deterministic_under_concurrency(monkeypatch):
    import evaluation.judge as judge_mod
    from evaluation.judge import EvaluationJudge

    fake = PromptKeyedJudgeLLM()
    j = EvaluationJudge(llm=fake)

    monkeypatch.setattr(judge_mod, "EVAL_MAX_CONCURRENCY", 1)
    a = j.score_all(
        query="What is revenue?",
        answer="1.2 million",
        context="revenue 1.2 million",
        gold_answer="2024 revenue 1.2M",
        gold_pages=None,
        sources=[{"page": 1, "text": "x"}],
    ).as_dict()

    monkeypatch.setattr(judge_mod, "EVAL_MAX_CONCURRENCY", 3)
    b = j.score_all(
        query="What is revenue?",
        answer="1.2 million",
        context="revenue 1.2 million",
        gold_answer="2024 revenue 1.2M",
        gold_pages=None,
        sources=[{"page": 1, "text": "x"}],
    ).as_dict()

    assert a == b


def test_judge_score_all_concurrent_is_faster(monkeypatch):
    """Sanity check: three judge LLMs in parallel should beat serial wall-clock."""
    import evaluation.judge as judge_mod
    from evaluation.judge import EvaluationJudge

    j = EvaluationJudge(llm=SlowJudgeLLM(delay_s=0.2))

    monkeypatch.setattr(judge_mod, "EVAL_MAX_CONCURRENCY", 1)
    t0 = time.perf_counter()
    j.score_all(
        query="q",
        answer="a",
        context="c",
        gold_answer="g",
        gold_pages=None,
        sources=[{"page": 1, "text": "t"}],
    )
    serial_s = time.perf_counter() - t0
    assert serial_s > 0.55

    monkeypatch.setattr(judge_mod, "EVAL_MAX_CONCURRENCY", 3)
    t1 = time.perf_counter()
    j.score_all(
        query="q",
        answer="a",
        context="c",
        gold_answer="g",
        gold_pages=None,
        sources=[{"page": 1, "text": "t"}],
    )
    parallel_s = time.perf_counter() - t1
    assert parallel_s < 0.5


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


def _mk_result(
    question_id: str,
    config: str,
    score: float,
    metric: str = "faithfulness",
    overrides=None,
):
    scores = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_recall": 0.0,
        "conciseness": 0.0,
    }
    scores[metric] = score
    if overrides:
        scores.update(overrides)
    from evaluation.report import PerQuestionResult

    return PerQuestionResult(
        question_id=question_id,
        question_type="keyword",
        config=config,
        question="Q",
        answer="A",
        gold_answer="G",
        sources=[],
        scores=scores,
    )


def test_build_summary_significance_absent_when_single_config():
    from evaluation.report import build_summary

    results = [_mk_result("q1", "naive", 0.1), _mk_result("q2", "naive", 0.2)]
    s = build_summary(results)
    assert "significance" not in s


def test_build_summary_significance_present_with_two_configs():
    pytest.importorskip("scipy")
    from evaluation.report import build_summary

    results = []
    for i in range(1, 6):
        qid = f"q{i}"
        results.append(
            _mk_result(
                qid,
                "naive",
                0.3,
                overrides={
                    "faithfulness": 0.3,
                    "answer_relevancy": 0.3,
                    "context_recall": 0.3,
                    "conciseness": 0.3,
                },
            )
        )
        results.append(
            _mk_result(
                qid,
                "neurosymbolic",
                0.5,
                overrides={
                    "faithfulness": 0.5,
                    "answer_relevancy": 0.5,
                    "context_recall": 0.5,
                    "conciseness": 0.5,
                },
            )
        )
    s = build_summary(results)
    assert "significance" in s
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "conciseness",
    ):
        assert metric in s["significance"]
        row = s["significance"][metric]
        for key in ("delta", "p_value", "ci_low", "ci_high", "verdict", "n_paired"):
            assert key in row


def test_significance_all_ties_returns_tie_verdict():
    pytest.importorskip("scipy")
    from evaluation.report import build_summary

    results = []
    for i in range(1, 6):
        qid = f"q{i}"
        scores = {
            "faithfulness": 0.4,
            "answer_relevancy": 0.4,
            "context_recall": 0.4,
            "conciseness": 0.4,
        }
        results.append(_mk_result(qid, "naive", 0.4, overrides=scores))
        results.append(_mk_result(qid, "neurosymbolic", 0.4, overrides=scores))
    s = build_summary(results)
    for metric in s["significance"].values():
        assert metric["verdict"] == "tie"
        assert abs(metric["delta"]) < 1e-9
        assert metric["p_value"] == pytest.approx(1.0)


def test_significance_small_n_is_directional():
    pytest.importorskip("scipy")
    from evaluation.report import build_summary

    results = []
    for i in range(1, 6):
        qid = f"q{i}"
        results.append(_mk_result(qid, "naive", 0.1, metric="faithfulness"))
        results.append(_mk_result(qid, "neurosymbolic", 0.3, metric="faithfulness"))
    s = build_summary(results)
    stat = s["significance"]["faithfulness"]
    assert stat["delta"] > 0
    assert stat["verdict"] == "directional"


def test_significance_large_n_significant():
    pytest.importorskip("scipy")
    from evaluation.report import build_summary

    results = []
    for i in range(1, 31):
        qid = f"q{i}"
        results.append(_mk_result(qid, "naive", 0.2, metric="answer_relevancy"))
        results.append(_mk_result(qid, "neurosymbolic", 0.5, metric="answer_relevancy"))
    s = build_summary(results)
    stat = s["significance"]["answer_relevancy"]
    assert stat["verdict"] == "significant"
    assert stat["p_value"] < 0.05
    assert stat["ci_low"] > 0


def test_render_markdown_includes_significance_section():
    pytest.importorskip("scipy")
    from evaluation.dataset import BenchmarkMeta
    from evaluation.report import build_summary, render_markdown

    results = []
    for i in range(1, 6):
        qid = f"q{i}"
        results.append(_mk_result(qid, "naive", 0.2, metric="faithfulness"))
        results.append(_mk_result(qid, "neurosymbolic", 0.4, metric="faithfulness"))
    summary = build_summary(results)
    meta = BenchmarkMeta(
        name="t",
        pdf_id="p1",
        fixture_pdf="fixture.pdf",
        distractor_pdfs=[],
        collection_name="documents",
        description="d",
    )
    md = render_markdown(meta, results, summary)
    assert "## Statistical significance (paired Wilcoxon, bootstrap 95% CI)" in md
    assert "Sample size is below 20 paired questions" in md


def test_make_retrieve_pipeline_is_importable():
    pytest.importorskip("qdrant_client")
    from local_llm import make_retrieve_pipeline

    assert callable(make_retrieve_pipeline)
