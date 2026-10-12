"""Tests for parse, clean, threads, and the full ingest pipeline."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ekb.ingest.parse import parse_file
from ekb.ingest.clean import clean, strip_quotes, strip_signature
from ekb.ingest.threads import build_threads, normalize_subject
from ekb.index.store import EmailStore
from ekb.pipeline import Pipeline


# --- parse ---

def test_parse_json():
    emails = parse_file(Path("data/sample/thread_project_alpha.json"))
    assert len(emails) == 6
    assert emails[0].sender == "alice@example.com"
    assert emails[1].in_reply_to == "<alpha-001@example.com>"


def test_parse_eml():
    eml = b"""From: alice@example.com
To: bob@example.com
Subject: Test
Message-ID: <test-eml@example.com>
Date: Mon, 10 Jan 2025 09:00:00 +0000

Hello from .eml
"""
    tmp = Path(tempfile.mktemp(suffix=".eml"))
    tmp.write_bytes(eml)
    emails = parse_file(tmp)
    tmp.unlink()
    assert len(emails) == 1
    assert emails[0].sender == "alice@example.com"
    assert "Hello from .eml" in emails[0].body


# --- clean ---

def test_strip_quotes_gmail():
    body = "My new reply.\n\nOn Jan 10, 2025, Bob <bob@x.com> wrote:\n> old stuff\n> more old"
    result = strip_quotes(body)
    assert "My new reply" in result
    assert "old stuff" not in result


def test_strip_quotes_outlook():
    body = "New content here.\n\n-----Original Message-----\nFrom: someone\nblah"
    result = strip_quotes(body)
    assert "New content" in result
    assert "From: someone" not in result


def test_strip_signature_rfc():
    body = "Real content here.\n\n-- \nJohn Doe\nCEO, Example Inc."
    result = strip_signature(body)
    assert "Real content" in result
    assert "CEO" not in result


def test_clean_combined():
    body = (
        "My actual reply.\n\n"
        "On Jan 10, 2025, Alice wrote:\n"
        "> previous message\n\n"
        "-- \nBob\nSenior Dev"
    )
    result = clean(body)
    assert "My actual reply" in result
    assert "previous message" not in result
    assert "Senior Dev" not in result


# --- threads ---

def test_normalize_subject():
    assert normalize_subject("Re: Fw: Re: Hello") == "hello"
    assert normalize_subject("FWD: Test") == "test"


def test_build_threads_with_headers():
    emails = parse_file(Path("data/sample/thread_project_alpha.json"))
    threads = build_threads(emails)
    # alpha-005 starts a new subject ("audit complete") with no in_reply_to to the kickoff thread
    assert len(threads) == 2
    sizes = sorted(len(v) for v in threads.values())
    assert sizes == [2, 4]


def test_build_threads_separate():
    alpha = parse_file(Path("data/sample/thread_project_alpha.json"))
    bug = parse_file(Path("data/sample/thread_bug_report.json"))
    threads = build_threads(alpha + bug)
    assert len(threads) == 3


def test_thread_parent_ids():
    emails = parse_file(Path("data/sample/thread_project_alpha.json"))
    build_threads(emails)
    root = [e for e in emails if e.message_id == "<alpha-001@example.com>"][0]
    reply = [e for e in emails if e.message_id == "<alpha-002@example.com>"][0]
    assert root.parent_id is None
    assert reply.parent_id == "<alpha-001@example.com>"


# --- store ---

def test_store_round_trip():
    db = Path(tempfile.mktemp(suffix=".db"))
    store = EmailStore(db)
    emails = parse_file(Path("data/sample/thread_bug_report.json"))
    build_threads(emails)
    for e in emails:
        store.insert(e, clean(e.body))
    store.commit()
    assert store.count()["emails"] == 3
    assert store.count()["threads"] == 1

    got = store.get("<bug-001@example.com>")
    assert got is not None
    assert got.sender == "dave@example.com"

    thread = store.get_thread(emails[0].thread_id)
    assert len(thread.emails) == 3
    assert thread.participants == {"dave@example.com", "bob@example.com", "alice@example.com"}

    store.close()
    db.unlink()


# --- pipeline integration ---

def test_pipeline_ingest():
    db = Path(tempfile.mktemp(suffix=".db"))
    p = Pipeline(db_path=str(db))
    count = p.ingest("data/sample")
    assert count == 9
    assert p.stats["emails"] == 9
    assert p.stats["threads"] == 3
    p.store.close()
    db.unlink()


if __name__ == "__main__":
    test_parse_json()
    test_parse_eml()
    test_strip_quotes_gmail()
    test_strip_quotes_outlook()
    test_strip_signature_rfc()
    test_clean_combined()
    test_normalize_subject()
    test_build_threads_with_headers()
    test_build_threads_separate()
    test_thread_parent_ids()
    test_store_round_trip()
    test_pipeline_ingest()
    print("All ingest tests passed.")
