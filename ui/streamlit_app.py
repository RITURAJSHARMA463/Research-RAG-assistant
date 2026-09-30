"""Minimal Streamlit chat UI that streams the graph's steps and renders citations.

Run:  streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the project importable when run via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import get_app  # noqa: E402
from app.config import settings  # noqa: E402

st.set_page_config(page_title="Research RAG Assistant", page_icon="📚")
st.title("📚 Research RAG Assistant")
st.caption("Agentic, self-correcting RAG over arXiv papers · answers with citations")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask about the indexed papers...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        graph = get_app()
        config = {"configurable": {"thread_id": "ui"}, "recursion_limit": settings.recursion_limit}

        labels = {
            "route": "Routing the question",
            "retrieve": "Hybrid retrieval + rerank",
            "grade_documents": "Grading retrieved passages",
            "rewrite": "Rewriting the query",
            "web_search": "Falling back to web search",
            "generate": "Generating the answer",
            "grade_answer": "Checking faithfulness",
        }
        final_state = {}
        for update in graph.stream(
            {"question": question, "original_question": question},
            config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                final_state.update(node_output or {})
                status.write(f"• {labels.get(node_name, node_name)}")
        status.update(label="Done", state="complete", expanded=False)

        answer = final_state.get("generation", "_No answer produced._")
        st.markdown(answer)

        sources = final_state.get("sources", [])
        if sources:
            st.markdown("**Sources**")
            for s in sources:
                cite = s.get("citation") or s.get("title")
                url = s.get("url")
                line = f"[{s['n']}] {cite}"
                st.markdown(f"{line} — [{url}]({url})" if url else line)

        st.session_state.history.append({"role": "assistant", "content": answer})
