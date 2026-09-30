"""Retrieval node — hybrid dense+sparse search with cross-encoder reranking."""
from __future__ import annotations

from app.retrieval.hybrid import hybrid_search
from app.state import GraphState


def retrieve(state: GraphState) -> dict:
    docs = hybrid_search(state["question"])
    return {"documents": docs, "used_web": False}
