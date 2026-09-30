"""FastAPI service. Two endpoints:

  POST /chat         -> runs the graph, returns answer + sources + trace flags
  POST /chat/stream  -> server-sent events, emits each node as it executes so a UI
                        can show "routing -> retrieving -> grading -> generating..."
"""
from __future__ import annotations

import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.graph import answer as run_answer
from app.graph import get_app

app = FastAPI(title="Research RAG Assistant", version="0.1.0")


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    result = run_answer(req.question, thread_id=thread_id)
    result["thread_id"] = thread_id
    return result


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    graph = get_app()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.recursion_limit}

    def event_gen():
        final_state = {}
        for update in graph.stream(
            {"question": req.question, "original_question": req.question},
            config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                final_state.update(node_output or {})
                yield f"event: step\ndata: {json.dumps({'node': node_name})}\n\n"
        payload = {
            "answer": final_state.get("generation", ""),
            "sources": final_state.get("sources", []),
            "used_web": final_state.get("used_web", False),
            "grounded": final_state.get("grounded"),
            "thread_id": thread_id,
        }
        yield f"event: done\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
