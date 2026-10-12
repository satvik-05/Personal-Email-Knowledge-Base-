"""Tests for chunking, vector index, and store chunk operations."""

import tempfile
from pathlib import Path

from ekb.index.chunk import chunk_email, CHUNK_SIZE
from ekb.index.store import EmailStore
from ekb.index.vectors import VectorIndex
from ekb.ingest.clean import clean
from ekb.ingest.parse import parse_file
from ekb.ingest.threads import build_threads
from ekb.pipeline import Pipeline


def test_chunk_short_email():
    emails = parse_file(Path("data/sample/thread_bug_report.json"))
    build_threads(emails)
    e = emails[0]
    chunks = chunk_email(e, clean(e.body))
    assert len(chunks) == 1
    assert chunks[0].email_id == e.message_id
    assert chunks[0].position == 0


def test_chunk_long_body():
    emails = parse_file(Path("data/sample/thread_bug_report.json"))
    build_threads(emails)
    e = emails[0]
    fake_long = "word " * 200
    chunks = chunk_email(e, fake_long)
    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_SIZE for c in chunks)
    assert all(c.chunk_id != chunks[0].chunk_id for c in chunks[1:])


def test_chunk_ids_deterministic():
    emails = parse_file(Path("data/sample/thread_bug_report.json"))
    build_threads(emails)
    e = emails[0]
    body = clean(e.body)
    c1 = chunk_email(e, body)
    c2 = chunk_email(e, body)
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]


def test_store_chunks():
    db = Path(tempfile.mktemp(suffix=".db"))
    store = EmailStore(db)
    emails = parse_file(Path("data/sample/thread_bug_report.json"))
    build_threads(emails)
    for e in emails:
        body_clean = clean(e.body)
        store.insert(e, body_clean)
        for c in chunk_email(e, body_clean):
            store.insert_chunk(c)
    store.commit()
    assert store.count()["chunks"] > 0

    chunks = store.get_chunks_by_email(emails[0].message_id)
    assert len(chunks) >= 1
    assert chunks[0].email_id == emails[0].message_id

    got = store.get_chunk(chunks[0].chunk_id)
    assert got is not None
    assert got.text == chunks[0].text

    store.close()
    db.unlink()


def test_vector_index_search():
    faiss_path = Path(tempfile.mktemp(suffix=".faiss"))
    vi = VectorIndex(faiss_path)
    emails = parse_file(Path("data/sample/thread_project_alpha.json"))
    build_threads(emails)
    all_chunks = []
    for e in emails:
        all_chunks.extend(chunk_email(e, clean(e.body)))
    vi.add(all_chunks)
    assert vi.count == len(all_chunks)

    results = vi.search("Stripe webhook migration", top_k=3)
    assert len(results) > 0
    assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    vi.save()
    vi2 = VectorIndex(faiss_path)
    assert vi2.count == vi.count
    results2 = vi2.search("Stripe webhook migration", top_k=3)
    assert results2[0][0] == results[0][0]

    faiss_path.unlink()
    faiss_path.with_suffix(".meta.npy").unlink()


def test_pipeline_ingest_with_index():
    db = Path(tempfile.mktemp(suffix=".db"))
    faiss_path = Path(tempfile.mktemp(suffix=".faiss"))
    p = Pipeline(db_path=str(db), faiss_path=str(faiss_path))
    count = p.ingest("data/sample")
    assert count == 9
    stats = p.stats
    assert stats["emails"] == 9
    assert stats["chunks"] > 0
    assert p.vectors.count == stats["chunks"]

    results = p.vectors.search("duplicate charges idempotency", top_k=3)
    assert len(results) > 0

    p.store.close()
    db.unlink()
    faiss_path.unlink()
    faiss_path.with_suffix(".meta.npy").unlink()


if __name__ == "__main__":
    test_chunk_short_email()
    test_chunk_long_body()
    test_chunk_ids_deterministic()
    test_store_chunks()
    print("Non-vector tests passed.")
    test_vector_index_search()
    test_pipeline_ingest_with_index()
    print("All index tests passed.")
