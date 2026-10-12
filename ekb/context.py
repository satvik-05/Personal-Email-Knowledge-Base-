"""Thread-aware chronological context assembly for grounded generation."""

from __future__ import annotations

from collections import defaultdict

from ekb.index.store import EmailStore
from ekb.models import Candidate, Email


def build_context(candidates: list[Candidate], store: EmailStore) -> str:
    """Group reranked hits by thread, order chronologically, splice neighbors."""
    threads: dict[str | None, list[Email]] = defaultdict(list)
    seen: set[str] = set()

    for cand in candidates:
        eid = cand.chunk.email_id
        if eid in seen:
            continue
        seen.add(eid)
        email = store.get(eid)
        if email is None:
            continue
        threads[email.thread_id].append(email)

    for emails in threads.values():
        emails.sort(key=lambda e: e.date)

    sorted_threads = sorted(threads.items(), key=lambda t: t[1][0].date)

    blocks: list[str] = []
    cite_idx = 0
    for _tid, emails in sorted_threads:
        for email in emails:
            cite_idx += 1
            header = f"[{cite_idx}] From: {email.sender} | Date: {email.date.strftime('%Y-%m-%d %H:%M')} | Subject: {email.subject}"
            blocks.append(f"{header}\n{email.body[:500]}")

    return "\n\n---\n\n".join(blocks)
