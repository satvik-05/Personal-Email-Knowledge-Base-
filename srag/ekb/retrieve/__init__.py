"""Hybrid retrieval: semantic (FAISS) + BM25 (FTS5), with reciprocal rank fusion."""

from __future__ import annotations

from ekb.index.store import EmailStore
from ekb.index.vectors import VectorIndex
from ekb.models import Candidate, Chunk


def retrieve(
    query: str,
    store: EmailStore,
    vectors: VectorIndex,
    top_k: int = 10,
    semantic_k: int = 20,
    bm25_k: int = 20,
) -> list[Candidate]:
    """Hybrid retrieve: fuse semantic + BM25 results via RRF, return top_k."""
    sem_hits = _semantic(query, vectors, semantic_k)
    bm25_hits = _bm25(query, store, bm25_k)
    fused = _rrf(sem_hits, bm25_hits)

    candidates = []
    for chunk_id, score in fused[:top_k]:
        chunk = store.get_chunk(chunk_id)
        if chunk is None:
            continue
        source = "hybrid"
        if chunk_id in {cid for cid, _ in sem_hits} and chunk_id not in {cid for cid, _ in bm25_hits}:
            source = "semantic"
        elif chunk_id in {cid for cid, _ in bm25_hits} and chunk_id not in {cid for cid, _ in sem_hits}:
            source = "bm25"
        candidates.append(Candidate(chunk=chunk, score=score, source=source))
    return candidates


def _semantic(query: str, vectors: VectorIndex, k: int) -> list[tuple[str, float]]:
    return vectors.search(query, top_k=k)


def _bm25(query: str, store: EmailStore, k: int) -> list[tuple[str, float]]:
    """Use FTS5 to find matching emails, then return their chunk IDs with BM25 rank scores."""
    import re
    # ponytail: strip non-alphanumeric FTS5 meta chars, keep words only
    fts_query = " ".join(re.findall(r"\w+", query))
    if not fts_query:
        return []
    rows = store._conn.execute(
        """SELECT e.message_id, bm25(emails_fts) AS score
           FROM emails_fts
           JOIN emails e ON e.rowid = emails_fts.rowid
           WHERE emails_fts MATCH ?
           ORDER BY score
           LIMIT ?""",
        (fts_query, k * 2),
    ).fetchall()

    results: list[tuple[str, float]] = []
    for row in rows:
        chunks = store.get_chunks_by_email(row["message_id"])
        for c in chunks:
            results.append((c.chunk_id, -float(row["score"])))  # bm25() returns negative
    return results[:k]


# ponytail: RRF with k=60, good enough baseline — swap for learned weights if needed
_RRF_K = 60


def _rrf(
    *result_lists: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (chunk_id, _) in enumerate(results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
