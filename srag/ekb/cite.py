"""Map [n] citation markers in generated text to real email metadata."""

from __future__ import annotations

import re

from ekb.index.store import EmailStore
from ekb.models import Candidate, Citation, Email


def build_citations(candidates: list[Candidate], store: EmailStore) -> list[Citation]:
    """Build ordered citation list from reranked candidates, deduped by email."""
    citations: list[Citation] = []
    seen: set[str] = set()
    for cand in candidates:
        eid = cand.chunk.email_id
        if eid in seen:
            continue
        seen.add(eid)
        email = store.get(eid)
        if email is None:
            continue
        idx = len(citations) + 1
        citations.append(Citation(
            index=idx, email_id=email.message_id, sender=email.sender,
            date=email.date, subject=email.subject,
            snippet=cand.chunk.text[:300],
        ))
    return citations


def strip_invalid_refs(text: str, max_index: int) -> str:
    """Remove [n] markers from text where n > max_index (hallucinated citations)."""
    def _check(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= max_index else ""
    return re.sub(r"\[(\d+)\]", _check, text)
