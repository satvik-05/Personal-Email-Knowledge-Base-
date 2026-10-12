"""Run eval: python -m eval"""

from pathlib import Path

from ekb.pipeline import Pipeline
from eval import evaluate, load_cases

SAMPLE_DIR = Path("data/sample")
CASES_PATH = Path("eval/cases.json")


def main():
    pipeline = Pipeline(db_path=":memory:", faiss_path="eval_tmp.faiss")
    for f in SAMPLE_DIR.glob("*.json"):
        pipeline.ingest(f)

    cases = load_cases(CASES_PATH)
    report = evaluate(pipeline, cases)

    print(report.summary())
    print()
    for r in report.results:
        status = "PASS" if r.recall > 0 else "MISS"
        print(f"[{status}] {r.question}")
        print(f"       P={r.precision:.2f} R={r.recall:.2f} MRR={r.mrr:.2f} ({r.retrieval_ms:.0f}ms)")

    # cleanup
    Path("eval_tmp.faiss").unlink(missing_ok=True)
    Path("eval_tmp.meta.npy").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
