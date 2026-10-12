"""SQLite storage for emails, metadata, and FTS5 full-text index."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ekb.models import Chunk, Email, Thread

_SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    message_id TEXT PRIMARY KEY,
    subject    TEXT NOT NULL,
    sender     TEXT NOT NULL,
    recipients TEXT NOT NULL,  -- JSON array
    date       TEXT NOT NULL,  -- ISO 8601
    body       TEXT NOT NULL,
    body_clean TEXT NOT NULL,
    in_reply_to TEXT,
    refs       TEXT NOT NULL,  -- JSON array
    thread_id  TEXT,
    parent_id  TEXT,
    headers    TEXT NOT NULL   -- JSON object
);

CREATE VIRTUAL TABLE IF NOT EXISTS emails_fts USING fts5(
    message_id UNINDEXED,
    subject,
    body_clean,
    content='emails',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS emails_ai AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts(rowid, message_id, subject, body_clean)
    VALUES (new.rowid, new.message_id, new.subject, new.body_clean);
END;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    email_id   TEXT NOT NULL REFERENCES emails(message_id),
    thread_id  TEXT,
    text       TEXT NOT NULL,
    position   INTEGER NOT NULL
);
"""


class EmailStore:
    def __init__(self, db_path: str | Path = "ekb.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def insert(self, email: Email, body_clean: str) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO emails
               (message_id, subject, sender, recipients, date, body, body_clean,
                in_reply_to, refs, thread_id, parent_id, headers)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                email.message_id,
                email.subject,
                email.sender,
                json.dumps(email.recipients),
                email.date.isoformat(),
                email.body,
                body_clean,
                email.in_reply_to,
                json.dumps(email.references),
                email.thread_id,
                email.parent_id,
                json.dumps(email.headers),
            ),
        )

    def insert_chunk(self, chunk: Chunk) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO chunks (chunk_id, email_id, thread_id, text, position) VALUES (?, ?, ?, ?, ?)",
            (chunk.chunk_id, chunk.email_id, chunk.thread_id, chunk.text, chunk.position),
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if not row:
            return None
        return Chunk(
            chunk_id=row["chunk_id"], email_id=row["email_id"],
            thread_id=row["thread_id"], text=row["text"], position=row["position"],
        )

    def get_chunks_by_email(self, email_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE email_id = ? ORDER BY position", (email_id,)
        ).fetchall()
        return [Chunk(r["chunk_id"], r["email_id"], r["thread_id"], r["text"], r["position"]) for r in rows]

    def commit(self) -> None:
        self._conn.commit()

    def get(self, message_id: str) -> Email | None:
        row = self._conn.execute(
            "SELECT * FROM emails WHERE message_id = ?", (message_id,)
        ).fetchone()
        return _row_to_email(row) if row else None

    def get_thread(self, thread_id: str) -> Thread:
        rows = self._conn.execute(
            "SELECT * FROM emails WHERE thread_id = ? ORDER BY date",
            (thread_id,),
        ).fetchall()
        emails = [_row_to_email(r) for r in rows]
        subject = emails[0].subject if emails else ""
        return Thread(thread_id=thread_id, subject=subject, emails=emails)

    def all_emails(self, query: str | None = None) -> list[Email]:
        """All emails chronologically, optionally filtered by FTS5 query."""
        if query:
            import re
            fts_query = " ".join(re.findall(r"\w+", query))
            if not fts_query:
                return []
            rows = self._conn.execute(
                """SELECT e.* FROM emails_fts
                   JOIN emails e ON e.rowid = emails_fts.rowid
                   WHERE emails_fts MATCH ?
                   ORDER BY e.date""",
                (fts_query,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM emails ORDER BY date").fetchall()
        return [_row_to_email(r) for r in rows]

    def all_threads(self) -> list[Thread]:
        rows = self._conn.execute(
            "SELECT DISTINCT thread_id FROM emails WHERE thread_id IS NOT NULL"
        ).fetchall()
        return [self.get_thread(r["thread_id"]) for r in rows]

    def count(self) -> dict[str, int]:
        emails = self._conn.execute("SELECT COUNT(*) c FROM emails").fetchone()["c"]
        threads = self._conn.execute(
            "SELECT COUNT(DISTINCT thread_id) c FROM emails WHERE thread_id IS NOT NULL"
        ).fetchone()["c"]
        chunks = self._conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        return {"emails": emails, "threads": threads, "chunks": chunks}

    def close(self) -> None:
        self._conn.close()


def _row_to_email(row: sqlite3.Row) -> Email:
    return Email(
        message_id=row["message_id"],
        subject=row["subject"],
        sender=row["sender"],
        recipients=json.loads(row["recipients"]),
        date=datetime.fromisoformat(row["date"]),
        body=row["body"],
        in_reply_to=row["in_reply_to"],
        references=json.loads(row["refs"]),
        thread_id=row["thread_id"],
        parent_id=row["parent_id"],
        headers=json.loads(row["headers"]),
    )
