"""Smoke test for eval module."""

from pathlib import Path

from ekb.pipeline import Pipeline
from eval import EvalCase, evaluate


def test_eval_runs():
    pipeline = Pipeline(db_path=":memory:", faiss_path="test_eval.faiss")
    for f in Path("data/sample").glob("*.json"):
        pipeline.ingest(f)

    cases = [
        EvalCase(
            question="What caused the duplicate charges?",
            relevant_email_ids=["<bug-001@example.com>", "<bug-002@example.com>"],
        ),
    ]
    report = evaluate(pipeline, cases)
    assert len(report.results) == 1
    assert report.mean_recall >= 0.0
    assert report.mean_precision >= 0.0
    assert report.mean_mrr >= 0.0
    assert report.results[0].retrieval_ms > 0

    Path("test_eval.faiss").unlink(missing_ok=True)
    Path("test_eval.meta.npy").unlink(missing_ok=True)
