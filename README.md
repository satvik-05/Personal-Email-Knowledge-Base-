# Email Knowledge Base (EKB)

A RAG pipeline that turns email archives into a searchable, queryable knowledge base. Upload `.eml`, `.mbox`, or `.json` email files, and ask natural-language questions with cited answers.

## What It Does

- **Ingest** emails from `.eml`, `.mbox`, or JSON files
- **Thread reconstruction** via `In-Reply-To`/`References` headers with subject-line fallback
- **Hybrid retrieval** combining semantic search (FAISS + `all-MiniLM-L6-v2`) and BM25 (SQLite FTS5), fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) for precision
- **Grounded generation** with citation markers `[n]` — uses Claude API, local HuggingFace, or extractive fallback
- **Streamlit UI** with thread browser, timeline view, and decision tracker

## Architecture

```
emails (.eml/.mbox/.json)
  -> parse -> clean (strip quotes/sigs) -> thread linking
  -> chunk (512 chars, 64 overlap) -> embed (MiniLM) -> FAISS index
  -> store in SQLite (metadata + FTS5)

query
  -> hybrid retrieve (semantic + BM25 via RRF)
  -> cross-encoder rerank
  -> context assembly (thread-aware, chronological)
  -> LLM generation with [n] citations
  -> citation validation (strip hallucinated refs)
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

Requirements: `streamlit`, `sentence-transformers`, `faiss-cpu`, `torch`, `anthropic`

### 2. Configure

Copy `.env` and set your preferences:

```bash
# LLM backend: "extractive" (no API needed), "anthropic", or "local"
EKB_LLM_BACKEND=extractive

# Cross-encoder reranking: "1" enabled, "0" disabled
EKB_RERANK=1

# Only needed if using anthropic backend
ANTHROPIC_API_KEY=your-key-here

# Optional: override model (default: claude-sonnet-4-20250514)
EKB_ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### 3. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload emails via the sidebar, then ask questions.

### Docker

```bash
docker build -t ekb .
docker run -p 8501:8501 --env-file .env ekb
```

## LLM Backends

| Backend | Set `EKB_LLM_BACKEND` to | Needs API key? | Notes |
|---|---|---|---|
| **Extractive** | `extractive` | No | Default. Returns top snippets directly, no LLM call. |
| **Claude API** | `anthropic` | Yes (`ANTHROPIC_API_KEY`) | Grounded answers with `[n]` citations via Claude. |
| **Local HF** | `local` | No | Runs `flan-t5-base` locally. Slow on CPU. |

## Eval Suite

Test retrieval quality against labeled cases in `eval/cases.json`:

```bash
# Standard eval — precision, recall, MRR
python -m eval

# Ablation — compare hybrid vs semantic-only vs BM25-only vs reranked
python -m eval.ablation
```

Sample output:
```
Config                    P@k    R@k    MRR       ms
-------------------------------------------------------
hybrid+rerank            0.667  1.000  1.000       42
hybrid-only              0.500  1.000  1.000       28
semantic-only            0.333  0.667  1.000       15
bm25-only                0.333  0.667  0.500       12
```

## Project Structure

```
ekb/
  models.py          # Email, Thread, Chunk, Candidate, Citation, Answer
  pipeline.py        # Orchestrator: ingest, query, timeline, decisions
  context.py         # Thread-aware context assembly for generation
  cite.py            # Citation builder + hallucinated-ref stripper
  ingest/
    parse.py         # .eml / .mbox / .json parser
    clean.py         # Quote and signature stripping
    threads.py       # Thread reconstruction from headers
  index/
    chunk.py         # Fixed-size overlapping chunker (512/64)
    vectors.py       # FAISS index + MiniLM embeddings
    store.py         # SQLite storage + FTS5 full-text index
  retrieve/
    __init__.py      # Hybrid retrieval (semantic + BM25 + RRF)
    rerank.py        # Cross-encoder reranker
  generate/
    __init__.py      # Generation orchestrator with extractive fallback
    backend.py       # Pluggable LLM backend (Anthropic / local HF)
eval/
  __init__.py        # Eval framework (precision, recall, MRR)
  __main__.py        # Run eval
  ablation.py        # Compare retrieval configs
  cases.json         # Labeled test cases
data/sample/         # Sample email threads (JSON)
app.py               # Streamlit UI
```

## Email JSON Format

To import emails as JSON, use this structure (single object or array):

```json
{
  "message_id": "<unique-id@example.com>",
  "subject": "Re: Project kickoff",
  "from": "alice@example.com",
  "to": ["bob@example.com"],
  "date": "2025-01-10T09:00:00Z",
  "body": "Email body text...",
  "in_reply_to": "<parent-id@example.com>",
  "references": ["<root-id@example.com>", "<parent-id@example.com>"]
}
```

`in_reply_to` and `references` are optional but enable thread reconstruction.

## Tests

```bash
pytest
```
