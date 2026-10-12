"""Cross-encoder reranker — scores (query, passage) pairs for precision."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from ekb.models import Candidate

# ponytail: small cross-encoder, swap for larger if precision matters more than latency
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[Candidate], top_k: int = 6) -> list[Candidate]:
    """Rerank candidates with a cross-encoder, return top_k."""
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c.chunk.text) for c in candidates]
    scores = model.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [
        Candidate(chunk=c.chunk, score=float(s), source="reranked")
        for c, s in scored[:top_k]
    ]
