"""Split email bodies into chunks for embedding."""

from __future__ import annotations

import hashlib

from ekb.models import Chunk, Email

# ponytail: fixed-size with overlap, no fancy recursive splitting — upgrade to sentence-aware if retrieval recall suffers
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_email(email: Email, body_clean: str) -> list[Chunk]:
    """Split a cleaned email body into overlapping chunks."""
    text = body_clean.strip()
    if not text:
        return []

    if len(text) <= CHUNK_SIZE:
        return [_make_chunk(email, text, 0)]

    chunks = []
    start = 0
    pos = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end]
        chunks.append(_make_chunk(email, chunk_text, pos))
        pos += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
        if end == len(text):
            break

    return chunks


def _make_chunk(email: Email, text: str, position: int) -> Chunk:
    cid = hashlib.sha256(f"{email.message_id}:{position}".encode()).hexdigest()[:16]
    return Chunk(
        chunk_id=cid,
        email_id=email.message_id,
        thread_id=email.thread_id,
        text=text,
        position=position,
    )
