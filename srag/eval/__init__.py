"""Retrieval evaluation — measures precision, recall, MRR against labeled queries."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ekb.pipeline import Pipeline


@dataclass
class EvalCase:
    question: str
    relevant_email_ids: list[str]
    top_k: int = 6


@dataclass
class EvalResult:
    question: str
    precision: float
    recall: float
    mrr: float
    retrieval_ms: float
    retrieved_ids: list[str]
    relevant_ids: list[str]


@dataclass
class EvalReport:
    results: list[EvalResult]

    @property
    def mean_precision(self) -> float:
        return sum(r.precision for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_recall(self) -> float:
        return sum(r.recall for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_mrr(self) -> float:
        return sum(r.mrr for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_retrieval_ms(self) -> float:
        return sum(r.retrieval_ms for r in self.results) / len(self.results) if self.results else 0.0

    def summary(self) -> str:
        lines = [
            f"Cases: {len(self.results)}",
            f"Precision@k: {self.mean_precision:.3f}",
            f"Recall@k:    {self.mean_recall:.3f}",
            f"MRR:         {self.mean_mrr:.3f}",
            f"Latency:     {self.mean_retrieval_ms:.0f}ms avg",
        ]
        return "\n".join(lines)


def load_cases(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text())
    return [EvalCase(**c) for c in data]


def evaluate(pipeline: Pipeline, cases: list[EvalCase]) -> EvalReport:
    results = []
    for case in cases:
        t0 = time.perf_counter()
        candidates = pipeline.query(case.question, top_k=case.top_k)
        elapsed = (time.perf_counter() - t0) * 1000

        retrieved_ids = []
        seen = set()
        for c in candidates.citations:
            if c.email_id not in seen:
                retrieved_ids.append(c.email_id)
                seen.add(c.email_id)

        relevant = set(case.relevant_email_ids)
        hits = [eid for eid in retrieved_ids if eid in relevant]

        precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(hits) / len(relevant) if relevant else 0.0

        # MRR: reciprocal rank of first relevant result
        mrr = 0.0
        for rank, eid in enumerate(retrieved_ids, 1):
            if eid in relevant:
                mrr = 1.0 / rank
                break

        results.append(EvalResult(
            question=case.question, precision=precision, recall=recall,
            mrr=mrr, retrieval_ms=elapsed,
            retrieved_ids=retrieved_ids, relevant_ids=case.relevant_email_ids,
        ))
    return EvalReport(results=results)
