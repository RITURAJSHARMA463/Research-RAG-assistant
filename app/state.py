"""The single typed state object that flows through every node in the graph.

Every node receives this dict and returns a partial dict; LangGraph merges
partial returns by key (last write wins for these non-accumulating fields).
"""
from __future__ import annotations

from typing import List, Literal, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # Inputs / working query
    question: str               # current question (may be rewritten mid-flight)
    original_question: str      # the user's first question, never mutated

    # Routing decision from the router node
    route: Literal["vectorstore", "web_search", "direct"]

    # Retrieval + grading
    documents: List[Document]   # candidate context after retrieval/rerank
    docs_relevant: bool         # set by the document grader
    used_web: bool              # whether the web fallback supplied the context

    # Generation + grading
    generation: str             # the drafted answer
    sources: List[dict]         # citation metadata aligned to [n] markers
    grounded: bool              # faithfulness grade (no hallucination)
    answers_question: bool      # answer-relevance grade

    # Self-correction counters (guard against infinite loops)
    rewrite_count: int
    generation_count: int
