"""Assembles the agentic, self-correcting RAG graph.

Topology (matches the architecture diagram):

    route ─┬─ vectorstore ─> retrieve ─> grade_documents ─┐
           ├─ web_search ──────────────────────┐          │ relevant
           └─ direct ─────────────────────────┐│          ▼
                                              ││        generate ─> grade_answer ─┐
                              weak docs        ││           ▲   hallucinated      │ passes
                              & rewrites left  ││           └────────────────────┘ ▼
                                  rewrite <─────┘ (loop)                          END
                                     │
                                     └──> retrieve (loop)

The conditional edges below implement: adaptive routing, the CRAG document loop
(rewrite & retry, then fall back to web search), and the Self-RAG answer loop
(regenerate on hallucination, rewrite on off-topic). Counters in state cap every
loop so the graph always terminates.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.nodes.generate import generate
from app.nodes.grade_answer import grade_answer
from app.nodes.grade_docs import grade_documents
from app.nodes.retrieve import retrieve
from app.nodes.rewrite import rewrite_query
from app.nodes.router import route_query
from app.nodes.web_search import web_search
from app.state import GraphState


# --- Conditional edge routers (pure: read state, return a node name) ---------

def _after_route(state: GraphState) -> str:
    return {
        "vectorstore": "retrieve",
        "web_search": "web_search",
        "direct": "generate",
    }[state["route"]]


def _after_grade_docs(state: GraphState) -> str:
    if state.get("docs_relevant"):
        return "generate"
    if state.get("rewrite_count", 0) < settings.max_rewrites:
        return "rewrite"          # try a better query
    return "web_search"           # rewrites exhausted -> live fallback


def _after_grade_answer(state: GraphState) -> str:
    # Regenerate if the answer hallucinated and we still have budget.
    if not state.get("grounded") and state.get("generation_count", 0) < settings.max_generations:
        return "generate"
    # Re-retrieve via a rewrite if the answer was off-topic and budget remains.
    if not state.get("answers_question") and state.get("rewrite_count", 0) < settings.max_rewrites:
        return "rewrite"
    return END                    # grounded + on-topic, or budget spent


def build_graph(checkpointer: MemorySaver | None = None):
    g = StateGraph(GraphState)

    g.add_node("route", route_query)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite", rewrite_query)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)
    g.add_node("grade_answer", grade_answer)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", _after_route,
                            {"retrieve": "retrieve", "web_search": "web_search", "generate": "generate"})

    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", _after_grade_docs,
                            {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"})

    g.add_edge("rewrite", "retrieve")     # the CRAG re-retrieval loop
    g.add_edge("web_search", "generate")

    g.add_edge("generate", "grade_answer")
    g.add_conditional_edges("grade_answer", _after_grade_answer,
                            {"generate": "generate", "rewrite": "rewrite", END: END})

    return g.compile(checkpointer=checkpointer or MemorySaver())


@lru_cache(maxsize=1)
def get_app():
    """Cached compiled graph for serving."""
    return build_graph()


def answer(question: str, thread_id: str = "default") -> dict:
    """Convenience runner used by the CLI/UI: returns answer + sources + trace flags."""
    app = get_app()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.recursion_limit}
    final = app.invoke({"question": question, "original_question": question}, config)
    return {
        "answer": final.get("generation", ""),
        "sources": final.get("sources", []),
        "route": final.get("route"),
        "used_web": final.get("used_web", False),
        "grounded": final.get("grounded"),
        "answers_question": final.get("answers_question"),
        "rewrites": final.get("rewrite_count", 0),
        "generations": final.get("generation_count", 0),
    }
