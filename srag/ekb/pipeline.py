"""RAG orchestrator — wires ingest, retrieval, reranking, context, generation, citation."""

from __future__ import annotations

import os
import time
from pathlib import Path

from ekb.index.chunk import chunk_email
from ekb.index.store import EmailStore
from ekb.index.vectors import VectorIndex
from ekb.ingest.clean import clean
from ekb.ingest.parse import parse_file
from ekb.ingest.threads import build_threads
from ekb.models import Answer
from ekb.retrieve import retrieve
from ekb.generate import generate


class Pipeline:
    def __init__(self, db_path: str = "ekb.db", faiss_path: str = "ekb.faiss"):
        self.db_path = Path(db_path)
        self.faiss_path = Path(faiss_path)
        self.store = EmailStore(self.db_path)
        self.vectors = VectorIndex(self.faiss_path)

    def ingest(self, path: str | Path) -> int:
        """Import emails from file/directory. Returns count ingested."""
        path = Path(path)
        files = list(path.iterdir()) if path.is_dir() else [path]
        all_emails = []
        for f in files:
            if f.suffix.lower() in (".eml", ".json", ".mbox"):
                all_emails.extend(parse_file(f))

        threads = build_threads(all_emails)
        count = 0
        for tid, emails in threads.items():
            for e in emails:
                body_clean = clean(e.body)
                self.store.insert(e, body_clean)
                chunks = chunk_email(e, body_clean)
                for c in chunks:
                    self.store.insert_chunk(c)
                self.vectors.add(chunks)
                count += 1
        self.store.commit()
        self.vectors.save()
        return count

    def query(self, question: str, top_k: int = 6) -> Answer:
        """Full RAG pipeline: retrieve → rerank → generate → cite."""
        t0 = time.perf_counter()
        candidates = retrieve(question, self.store, self.vectors, top_k=top_k * 3)
        t_retrieve = time.perf_counter()

        if os.environ.get("EKB_RERANK", "1") == "1":
            from ekb.retrieve.rerank import rerank
            candidates = rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        t_rerank = time.perf_counter()

        answer = generate(question, candidates, self.store)
        t_generate = time.perf_counter()
        answer.retrieval_ms = (t_retrieve - t0) * 1000
        answer.generation_ms = (t_generate - t_rerank) * 1000
        return answer

    def timeline(self, question: str | None = None) -> list[dict]:
        """Chronological timeline of emails, optionally filtered by topic."""
        emails = self.store.all_emails(query=question)
        return [
            {"date": e.date, "sender": e.sender, "subject": e.subject,
             "snippet": e.body[:200], "message_id": e.message_id}
            for e in emails
        ]

    def decisions(self, question: str | None = None) -> list[dict]:
        """Emails containing decision language, optionally filtered by topic."""
        # ponytail: keyword scan — swap for LLM classification when available
        import re
        _DECISION_RE = re.compile(
            r"(decided|agreed|approved|let'?s go with|we'?ll use|final decision|"
            r"conclusion|resolved|we chose|going with|sign[- ]?off)",
            re.IGNORECASE,
        )
        emails = self.store.all_emails(query=question)
        results = []
        for e in emails:
            matches = _DECISION_RE.findall(e.body)
            if matches:
                results.append({
                    "date": e.date, "sender": e.sender, "subject": e.subject,
                    "snippet": e.body[:200], "message_id": e.message_id,
                    "decision_signals": list(set(matches)),
                })
        return results

    @property
    def stats(self) -> dict:
        """Index statistics: email count, thread count, chunk count."""
        counts = self.store.count()
        return {"emails": counts["emails"], "threads": counts["threads"], "chunks": counts["chunks"]}
