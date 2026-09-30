"""Query rewriting node — reformulates the question for better retrieval after a
weak retrieval or an off-topic answer. Increments the rewrite guard."""
from __future__ import annotations

from app.llm import get_llm, safe_invoke
from app.prompts import REWRITE_SYSTEM
from app.state import GraphState


def rewrite_query(state: GraphState) -> dict:
    llm = get_llm(temperature=0)
    improved = safe_invoke(llm, [
        ("system", REWRITE_SYSTEM),
        ("human", state["question"]),
    ]).content.strip()

    return {
        "question": improved,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }
