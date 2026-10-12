"""Ablation runner — compare retrieval configs in one table.

Usage: python -m eval.ablation
"""

from __future__ import annotations

import os
from pathlib import Path

from ekb.pipeline import Pipeline
from eval import EvalReport, evaluate, load_cases

SAMPLE_DIR = Path("data/sample")
CASES_PATH = Path("eval/cases.json")

CONFIGS = [
    {"name": "hybrid+rerank", "EKB_RERANK": "1"},
    {"name": "hybrid-only",   "EKB_RERANK": "0"},
    {"name": "semantic-only", "EKB_RERANK": "0", "_semantic_only": True},
    {"name": "bm25-only",     "EKB_RERANK": "0", "_bm25_only": True},
    {"name": "hybrid+rerank (k=3)", "EKB_RERANK": "1", "_top_k": "3"},
]


def _make_pipeline() -> Pipeline:
    p = Pipeline(db_path=":memory:", faiss_path="ablation_tmp.faiss")
    for f in SAMPLE_DIR.glob("*.json"):
        p.ingest(f)
    return p


def run() -> list[dict]:
    cases = load_cases(CASES_PATH)
    pipeline = _make_pipeline()
    results = []

    for cfg in CONFIGS:
        os.environ["EKB_RERANK"] = cfg.get("EKB_RERANK", "1")

        if cfg.get("_semantic_only"):
            _patch = _patch_semantic_only(pipeline)
        elif cfg.get("_bm25_only"):
            _patch = _patch_bm25_only(pipeline)
        else:
            _patch = _noop()

        _patch.__enter__()
        try:
            report = evaluate(pipeline, cases)
        finally:
            _patch.__exit__(None, None, None)

        results.append({
            "config": cfg["name"],
            "precision": report.mean_precision,
            "recall": report.mean_recall,
            "mrr": report.mean_mrr,
            "latency_ms": report.mean_retrieval_ms,
        })

    Path("ablation_tmp.faiss").unlink(missing_ok=True)
    Path("ablation_tmp.meta.npy").unlink(missing_ok=True)
    return results


class _noop:
    def __enter__(self): return self
    def __exit__(self, *a): pass


class _patch_semantic_only:
    """Temporarily disable BM25 in retrieval."""
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        import ekb.retrieve as r
        self._orig_bm25 = r._bm25
        r._bm25 = lambda *a, **kw: []
        return self

    def __exit__(self, *a):
        import ekb.retrieve as r
        r._bm25 = self._orig_bm25


class _patch_bm25_only:
    """Temporarily disable semantic search in retrieval."""
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        import ekb.retrieve as r
        self._orig_semantic = r._semantic
        r._semantic = lambda *a, **kw: []
        return self

    def __exit__(self, *a):
        import ekb.retrieve as r
        r._semantic = self._orig_semantic


def main():
    rows = run()
    print(f"{'Config':<25} {'P@k':>6} {'R@k':>6} {'MRR':>6} {'ms':>8}")
    print("-" * 55)
    for r in rows:
        print(f"{r['config']:<25} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['mrr']:>6.3f} {r['latency_ms']:>8.0f}")


if __name__ == "__main__":
    main()
