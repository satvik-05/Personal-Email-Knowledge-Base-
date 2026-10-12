"""Pluggable LLM backend — API (Anthropic) or local HF. Env-configured."""

from __future__ import annotations

import os


def generate_text(prompt: str) -> str:
    """Route to API or local backend based on env config."""
    backend = os.environ.get("EKB_LLM_BACKEND", "extractive")

    if backend == "anthropic":
        return _anthropic(prompt)
    elif backend == "local":
        return _local_hf(prompt)
    else:
        return ""


def _anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=os.environ.get("EKB_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ponytail: local HF pipeline, heavy on CPU — use API backend for production
def _local_hf(prompt: str) -> str:
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512)
    return pipe(prompt)[0]["generated_text"]
