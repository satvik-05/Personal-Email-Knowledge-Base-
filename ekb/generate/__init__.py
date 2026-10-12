"""Answer generation — grounded LLM with extractive fallback."""

from __future__ import annotations

import os

from ekb.cite import build_citations, strip_invalid_refs
from ekb.context import build_context
from ekb.generate.backend import generate_text
from ekb.index.store import EmailStore
from ekb.models import Answer, Candidate

_PROMPT_TEMPLATE = """Answer the question using ONLY the email evidence below. Cite sources as [n].
If the evidence is insufficient, say "I don't have enough information to answer that."

EVIDENCE:
{context}

QUESTION: {question}

ANSWER:"""


def generate(
    question: str,
    candidates: list[Candidate],
    store: EmailStore,
) -> Answer:
    if not candidates:
        return Answer(text="No relevant emails found.", candidates_used=0)

    citations = build_citations(candidates, store)
    backend = os.environ.get("EKB_LLM_BACKEND", "extractive")

    if backend in ("anthropic", "local"):
        context = build_context(candidates, store)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
        raw = generate_text(prompt)
        text = strip_invalid_refs(raw, len(citations))
    else:
        # ponytail: extractive fallback — no LLM needed
        snippets = [f"[{c.index}] {c.snippet}" for c in citations[:5]]
        text = "\n\n".join(snippets) if snippets else "No relevant emails found."

    return Answer(text=text, citations=citations, candidates_used=len(candidates))
