"""Smoke test: models import and construct without error."""

from datetime import datetime, timezone
from ekb.models import Email, Thread, Chunk, Candidate, Citation, Answer


def test_email_round_trip():
    e = Email(
        message_id="<test-1@example.com>",
        subject="Test",
        sender="a@example.com",
        recipients=["b@example.com"],
        date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        body="Hello",
    )
    assert e.message_id == "<test-1@example.com>"
    assert e.thread_id is None
    assert e.references == []


def test_thread_participants():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t = Thread(
        thread_id="t1",
        subject="Test",
        emails=[
            Email("m1", "Test", "a@x.com", ["b@x.com"], now, "hi"),
            Email("m2", "Re: Test", "b@x.com", ["a@x.com"], now, "hey"),
        ],
    )
    assert t.root.message_id == "m1"
    assert t.participants == {"a@x.com", "b@x.com"}


def test_answer_defaults():
    a = Answer(text="No relevant information found.")
    assert a.citations == []
    assert a.candidates_used == 0


if __name__ == "__main__":
    test_email_round_trip()
    test_thread_participants()
    test_answer_defaults()
    print("All model tests passed.")
