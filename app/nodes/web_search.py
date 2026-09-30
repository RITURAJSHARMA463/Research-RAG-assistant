"""Web search fallback node — used when routing says the question needs fresh
info, or when re-retrieval keeps failing. Uses Tavily; degrades gracefully if no
key is configured."""
from __future__ import annotations

from langchain_core.documents import Document

from app.config import settings
from app.state import GraphState


def web_search(state: GraphState) -> dict:
    question = state["question"]
    api_key = settings.keys.get("tavily")
    if not api_key:
        # No key: return an explicit, citable note rather than crashing.
        note = Document(
            page_content="Web search is unavailable (no TAVILY_API_KEY configured).",
            metadata={"title": "Web search unavailable", "url": "", "citation": "system"},
        )
        return {"documents": [note], "docs_relevant": True, "used_web": True}

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=question,
        max_results=settings.tavily_max_results,
        search_depth="advanced",
    )

    docs = [
        Document(
            page_content=r.get("content", ""),
            metadata={
                "title": r.get("title", "Web result"),
                "url": r.get("url", ""),
                "citation": r.get("title", "Web result"),
                "year": None,
            },
        )
        for r in resp.get("results", [])
    ]
    return {"documents": docs, "docs_relevant": bool(docs), "used_web": True}
