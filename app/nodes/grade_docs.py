"""Document grading node (the CRAG check) — keeps only relevant passages and
decides whether retrieval was good enough to generate from.

Grades ALL retrieved passages in a single LLM call (not one call per passage).
This is the single biggest lever on request volume: with the default final_k=5,
per-passage grading meant 5 LLM calls for this node alone — easily blowing through
a free-tier per-minute cap on its own. One structured-output call handles the same
job for the price of one request.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from app.llm import get_llm, safe_invoke
from app.prompts import GRADE_DOCS_SYSTEM
from app.state import GraphState


class _PassageVerdict(BaseModel):
    index: int = Field(..., description="1-based index of the passage, matching the prompt.")
    relevant: bool = Field(..., description="True if this passage helps answer the question.")


class _BatchGrade(BaseModel):
    verdicts: List[_PassageVerdict] = Field(
        ..., description="One verdict per numbered passage, in any order."
    )


def grade_documents(state: GraphState) -> dict:
    question = state["question"]
    docs = state.get("documents", [])
    if not docs:
        return {"documents": [], "docs_relevant": False}

    numbered = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1))
    grader = get_llm(temperature=0).with_structured_output(_BatchGrade)

    result = safe_invoke(grader, [
        ("system", GRADE_DOCS_SYSTEM),
        ("human", f"Question:\n{question}\n\nNumbered passages:\n{numbered}\n\n"
                  f"Return a verdict for every passage from 1 to {len(docs)}."),
    ])

    relevant_idx = {v.index for v in result.verdicts if v.relevant}
    kept = [d for i, d in enumerate(docs, start=1) if i in relevant_idx]

    # "Relevant enough" = at least one passage survived grading.
    return {"documents": kept, "docs_relevant": len(kept) > 0}
