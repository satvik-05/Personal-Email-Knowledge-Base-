"""Streamlit UI for the Email Knowledge Base."""

import tempfile
from pathlib import Path

import streamlit as st

from ekb.pipeline import Pipeline

st.set_page_config(page_title="Email Knowledge Base", layout="wide")
st.title("Email Knowledge Base")


@st.cache_resource
def get_pipeline():
    return Pipeline()


pipeline = get_pipeline()
stats = pipeline.stats

st.sidebar.header("Index")
st.sidebar.metric("Emails", stats["emails"])
st.sidebar.metric("Threads", stats["threads"])
st.sidebar.metric("Chunks", stats["chunks"])

uploaded = st.sidebar.file_uploader(
    "Import emails", type=["eml", "json", "mbox"], accept_multiple_files=True
)
if uploaded and st.sidebar.button("Ingest"):
    total = 0
    for f in uploaded:
        tmp = Path(tempfile.mktemp(suffix=Path(f.name).suffix))
        tmp.write_bytes(f.getvalue())
        total += pipeline.ingest(tmp)
        tmp.unlink()
    st.sidebar.success(f"Ingested {total} emails.")
    st.rerun()

# Thread browser
threads = pipeline.store.all_threads()
if threads:
    st.header("Threads")
    for t in threads:
        with st.expander(f"{t.subject} ({len(t.emails)} emails)"):
            for e in t.emails:
                st.markdown(f"**{e.sender}** — {e.date.strftime('%Y-%m-%d %H:%M')}")
                st.text(e.body[:500])
                st.divider()

st.header("Ask")
question = st.text_input("Question about your emails:")
if question:
    answer = pipeline.query(question)
    st.markdown(answer.text)
    if answer.citations:
        st.subheader("Sources")
        for c in answer.citations:
            st.caption(f"[{c.index}] {c.sender} — {c.subject} ({c.date.strftime('%Y-%m-%d')})")
    st.caption(f"Retrieval: {answer.retrieval_ms:.0f}ms | Candidates: {answer.candidates_used}")

st.header("Views")
view_filter = st.text_input("Filter by topic (optional):", key="view_filter") or None
col1, col2 = st.columns(2)
with col1:
    if st.button("Timeline"):
        events = pipeline.timeline(view_filter)
        if not events:
            st.info("No emails found.")
        for ev in events:
            st.markdown(f"**{ev['date'].strftime('%Y-%m-%d %H:%M')}** — {ev['sender']}")
            st.caption(ev["subject"])
            st.text(ev["snippet"])
            st.divider()
with col2:
    if st.button("Decisions"):
        decisions = pipeline.decisions(view_filter)
        if not decisions:
            st.info("No decisions found.")
        for d in decisions:
            st.markdown(f"**{d['date'].strftime('%Y-%m-%d %H:%M')}** — {d['sender']}")
            st.caption(f"{d['subject']} · Signals: {', '.join(d['decision_signals'])}")
            st.text(d["snippet"])
            st.divider()

