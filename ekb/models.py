from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Email:
    message_id: str
    subject: str
    sender: str
    recipients: list[str]
    date: datetime
    body: str
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)
    thread_id: str | None = None
    parent_id: str | None = None
    # ponytail: attachments deferred — add list[AttachmentMeta] when attachment indexing lands
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Thread:
    thread_id: str
    subject: str
    emails: list[Email] = field(default_factory=list)

    @property
    def root(self) -> Email | None:
        return self.emails[0] if self.emails else None

    @property
    def participants(self) -> set[str]:
        return {e.sender for e in self.emails}


@dataclass
class Chunk:
    chunk_id: str
    email_id: str
    thread_id: str | None
    text: str
    position: int  # chunk index within the email body


@dataclass
class Candidate:
    chunk: Chunk
    score: float
    source: str  # "bm25" | "semantic" | "hybrid" | "reranked"


@dataclass
class Citation:
    index: int  # [n] in the answer
    email_id: str
    sender: str
    date: datetime
    subject: str
    snippet: str  # the passage that supports the claim


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    candidates_used: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
