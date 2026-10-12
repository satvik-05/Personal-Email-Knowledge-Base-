"""Smoke test for hybrid retrieval + generation pipeline."""

from datetime import datetime, timezone

from ekb.index.store import EmailStore
from ekb.index.vectors import VectorIndex
from ekb.index.chunk import chunk_email
from ekb.ingest.clean import clean
from ekb.models import Email
from ekb.retrieve import retrieve
from ekb.generate import generate


def _make_email(mid: str, subject: str, body: str) -> Email:
    return Email(
        message_id=mid, subject=subject, sender="alice@test.com",
        recipients=["bob@test.com"], date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        body=body, thread_id="t1",
    )


def test_retrieve_and_generate(tmp_path):
    store = EmailStore(tmp_path / "test.db")
    vectors = VectorIndex(tmp_path / "test.faiss")

    e1 = _make_email("m1", "Deploy plan", "We will deploy the new API on Friday at 3pm.")
    e2 = _make_email("m2", "Lunch", "Let's grab lunch at noon tomorrow.")

    for e in [e1, e2]:
        body_clean = clean(e.body)
        store.insert(e, body_clean)
        chunks = chunk_email(e, body_clean)
        for c in chunks:
            store.insert_chunk(c)
        vectors.add(chunks)
    store.commit()
    vectors.save()

    candidates = retrieve("when is the deploy?", store, vectors, top_k=5)
    assert len(candidates) > 0
    assert any("deploy" in c.chunk.text.lower() for c in candidates)

    answer = generate("when is the deploy?", candidates, store)
    assert answer.text
    assert len(answer.citations) > 0
    assert answer.citations[0].subject == "Deploy plan"
