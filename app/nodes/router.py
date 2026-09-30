"""Adaptive routing node — decides whether a question needs the vectorstore,
a live web search, or a direct answer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.llm import get_llm, safe_invoke
from app.prompts import ROUTER_SYSTEM
from app.state import GraphState


class _Route(BaseModel):
    destination: Literal["vectorstore", "web_search", "direct"] = Field(
        ..., description="Where to send the question."
    )


def route_query(state: GraphState) -> dict:
    question = state["question"]
    router = get_llm(temperature=0).with_structured_output(_Route)
    decision = safe_invoke(router, [("system", ROUTER_SYSTEM), ("human", question)])
    return {
        "route": decision.destination,
        "original_question": state.get("original_question", question),
        "rewrite_count": state.get("rewrite_count", 0),
        "generation_count": state.get("generation_count", 0),
    }
