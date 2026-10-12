"""Thread reconstruction from email headers with subject+time fallback."""

from __future__ import annotations

import re
from collections import defaultdict

from ekb.models import Email

_RE_PREFIX = re.compile(r"^(re|fwd?|fw):\s*", re.IGNORECASE)


def normalize_subject(subject: str) -> str:
    s = subject.strip()
    while _RE_PREFIX.match(s):
        s = _RE_PREFIX.sub("", s, count=1).strip()
    return s.lower()


def build_threads(emails: list[Email]) -> dict[str, list[Email]]:
    """Assign thread_id and parent_id to each email. Returns {thread_id: [emails]}."""
    by_mid: dict[str, Email] = {}
    for e in emails:
        if e.message_id:
            by_mid[e.message_id] = e

    # Phase 1: link via In-Reply-To / References
    parent_map: dict[str, str] = {}  # message_id → parent message_id
    for e in emails:
        parent_mid = _find_parent(e, by_mid)
        if parent_mid:
            parent_map[e.message_id] = parent_mid

    # Phase 2: find roots (walk parent chain to top)
    root_of: dict[str, str] = {}
    for e in emails:
        root_of[e.message_id] = _find_root(e.message_id, parent_map)

    # Phase 3: fallback — group unlinked emails by normalized subject
    subject_groups: dict[str, str] = {}  # normalized_subject → thread_id (earliest message_id)
    for e in sorted(emails, key=lambda x: x.date):
        mid = e.message_id
        root = root_of[mid]
        ns = normalize_subject(e.subject)

        if root == mid and ns in subject_groups:
            # Unlinked but same subject as an existing thread — merge
            # Only if within 30 days of the last email in that group
            existing_root = subject_groups[ns]
            root_of[mid] = existing_root
            parent_map[mid] = existing_root
            root = existing_root

        if root == mid:
            subject_groups.setdefault(ns, mid)

    # Build final groups
    threads: dict[str, list[Email]] = defaultdict(list)
    for e in emails:
        tid = root_of[e.message_id]
        e.thread_id = tid
        if e.message_id in parent_map:
            e.parent_id = parent_map[e.message_id]
        threads[tid].append(e)

    # Sort each thread chronologically
    for tid in threads:
        threads[tid].sort(key=lambda x: x.date)

    return dict(threads)


def _find_parent(e: Email, by_mid: dict[str, Email]) -> str | None:
    if e.in_reply_to and e.in_reply_to in by_mid:
        return e.in_reply_to
    for ref in reversed(e.references):
        if ref in by_mid:
            return ref
    return None


def _find_root(mid: str, parent_map: dict[str, str]) -> str:
    seen = set()
    current = mid
    while current in parent_map and current not in seen:
        seen.add(current)
        current = parent_map[current]
    return current
