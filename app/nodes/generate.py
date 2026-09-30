"""Generation node — answers strictly from the numbered context and emits inline
[n] citations. Builds the sources list that the API/UI render."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.llm import get_llm, safe_invoke
from app.prompts import GENERATE_SYSTEM, GENERATE_USER
from app.state import GraphState


def _format_context(docs: List[Document]) -> tuple[str, List[dict]]:
    blocks, sources = [], []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        header = m.get("title", "source")
        blocks.append(f"[{i}] {header}\n{doc.page_content}")
        sources.append({
            "n": i,
            "title": m.get("title", "Untitled"),
            "citation": m.get("citation", header),
            "authors": m.get("authors", ""),
            "year": m.get("year"),
            "url": m.get("url", ""),
        })
    return "\n\n".join(blocks), sources


def generate(state: GraphState) -> dict:
    docs = state.get("documents", [])
    context, sources = _format_context(docs)

    llm = get_llm()  # generator may use the configured (warmer) temperature
    answer = safe_invoke(llm, [
        ("system", GENERATE_SYSTEM),
        ("human", GENERATE_USER.format(question=state["original_question"], context=context)),
    ]).content.strip()

    return {
        "generation": answer,
        "sources": sources,
        "generation_count": state.get("generation_count", 0) + 1,
    }
