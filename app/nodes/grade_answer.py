"""Answer grading node (the Self-RAG check) — grades the draft on two axes:
faithfulness (is it grounded in the context?) and answer-relevance (does it
actually address the question?).

Both checks run in a SINGLE LLM call (one structured-output response with two
boolean fields) instead of two separate calls — halves this node's request cost,
which matters a lot on tight free-tier per-minute limits.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.llm import get_llm, safe_invoke
from app.prompts import GRADE_ANSWER_SYSTEM, GRADE_FAITHFULNESS_SYSTEM
from app.state import GraphState

_COMBINED_SYSTEM = f"""{GRADE_FAITHFULNESS_SYSTEM}

Separately, also judge answer relevance:
{GRADE_ANSWER_SYSTEM}

Return both judgments."""


class _CombinedGrade(BaseModel):
    grounded: bool = Field(..., description="True if every claim is supported by the context.")
    answers_question: bool = Field(..., description="True if the answer resolves the user's question.")


def _context(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def grade_answer(state: GraphState) -> dict:
    generation = state.get("generation", "")
    grader = get_llm(temperature=0).with_structured_output(_CombinedGrade)

    result = safe_invoke(grader, [
        ("system", _COMBINED_SYSTEM),
        ("human",
         f"Question:\n{state['original_question']}\n\n"
         f"Context:\n{_context(state.get('documents', []))}\n\n"
         f"Answer:\n{generation}"),
    ])

    return {"grounded": result.grounded, "answers_question": result.answers_question}
