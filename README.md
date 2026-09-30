# Research RAG Assistant

An **agentic, self-correcting RAG** system over a corpus of arXiv papers, built with
**LangGraph + LangChain**. It answers research questions with **inline citations**,
and — unlike a linear retrieve-then-generate chain — it *judges its own work*: it
grades retrieved passages for relevance, rewrites the query and re-retrieves when
retrieval is weak, falls back to live web search, and checks its own answer for
hallucination before returning it.

It unifies three published patterns into one graph: **Adaptive RAG** (route per
query), **Corrective RAG / CRAG** (grade documents, recover on failure), and
**Self-RAG** (grade the generation, regenerate or re-retrieve).

---

## Why this is more than a tutorial RAG

| Capability | What it does | Where |
|---|---|---|
| Adaptive routing | Per-query decision: vectorstore vs. web vs. answer directly | `app/nodes/router.py` |
| Hybrid retrieval | Dense (Chroma) + BM25, fused with Reciprocal Rank Fusion | `app/retrieval/hybrid.py` |
| Cross-encoder rerank | Reorders candidates by jointly scoring (query, passage) | `app/retrieval/hybrid.py` |
| Document grading (CRAG) | Drops irrelevant chunks; triggers rewrite / web fallback. Grades all passages in **one** batched call, not one call per passage | `app/nodes/grade_docs.py` |
| Answer grading (Self-RAG) | Faithfulness + answer-relevance, graded together in **one** call | `app/nodes/grade_answer.py` |
| Bounded self-correction | Counters in state cap every loop so it always terminates | `app/graph.py` |
| Real citations | Every chunk carries title/authors/year/URL; answers cite `[n]` | `app/nodes/generate.py` |
| Evaluation | RAGAS: faithfulness, response relevance, context precision/recall | `eval/ragas_eval.py` |
| Serving | FastAPI (`/chat`, `/chat/stream` SSE) + Streamlit UI | `app/api.py`, `ui/` |

## Architecture

```
        ┌─────────┐  current info        ┌────────────┐
        │  route  │─────────────────────▶│ web_search │──┐
        └────┬────┘                       └────────────┘  │
   needs docs│                                            ▼
        ┌────▼─────┐   relevant    ┌──────────┐    ┌────────────┐  passes  ┌─────┐
        │ retrieve │──────────────▶│ generate │───▶│grade_answer│─────────▶│ END │
        │ +rerank  │               └──────────┘    └─────┬──────┘          └─────┘
        └────┬─────┘                    ▲                │ hallucinated
             ▼                          └────────────────┘
     ┌────────────────┐  weak docs   ┌─────────┐
     │ grade_documents│─────────────▶│ rewrite │──▶ (loops back to retrieve)
     └────────────────┘              └─────────┘
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your GOOGLE_API_KEY (get one at aistudio.google.com/app/apikey)
```

Defaults use **Gemini for the LLM** (router, graders, generator) and a **local HF
model for embeddings** (free, offline, no rate limits — the right choice for
bulk-embedding hundreds or thousands of chunks at ingest time; Gemini's free tier
caps embedding requests per minute and a full ingest run will exceed it). The
cross-encoder reranker also runs **locally**. Set `USE_RERANKER=false` for a fully
API-only footprint, or `EMBEDDINGS_PROVIDER=gemini` if you'd rather use Gemini for
embeddings too (rate-limited/retried automatically, but slower for large corpora).
The Tavily web fallback is optional — without a key the graph still runs and just
skips it.

## Usage

**1. Build the index from arXiv:**
```bash
python -m scripts.ingest_arxiv "retrieval augmented generation" --max 20
```

**2a. Ask via the Streamlit UI** (shows each graph step live + citations):
```bash
streamlit run ui/streamlit_app.py
```
> You may see `ModuleNotFoundError: No module named 'torchvision'` spam in the logs
> on startup. That's Streamlit's file-watcher probing `transformers`' optional
> vision submodules (unused here — our models are text-only) and is harmless. To
> silence it: `streamlit run ui/streamlit_app.py --server.fileWatcherType none`
> (trades away auto-reload on code edits).

**2b. Or via the API:**
```bash
uvicorn app.api:app --reload
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question":"How does a reranker improve retrieval precision?"}' | jq
```

**2c. Or from Python:**
```python
from app.graph import answer
print(answer("What problem does attention solve vs. recurrence?"))
```

## Evaluation

This is the part that proves the system *works*, not just runs. Point it at a golden
set and get per-question + aggregate scores:

```bash
cp eval/golden_set.example.json eval/golden_set.json   # then edit / expand it
python -m eval.ragas_eval --golden eval/golden_set.json
```

RAGAS separates the two failure surfaces of any RAG system — **retrieval** (context
precision/recall) and **generation** (faithfulness, response relevance) — so a low
score tells you *which half* to fix. Results are written to `eval/ragas_results.csv`.

## Configuration

Everything is environment-driven (`app/config.py`). Swap the LLM provider, switch
embeddings to OpenAI, turn the reranker off, or change `MAX_REWRITES` /
`MAX_GENERATIONS` without touching code. Set `LANGCHAIN_TRACING_V2=true` with a
LangSmith key to get full per-node traces.

**Staying on free tiers, by design.** Free-tier LLM quotas can be surprisingly
tight (some Gemini models cap at ~5 requests/minute). The graph is built to live
within that comfortably: document grading and the answer check each cost **one**
LLM call regardless of how many passages/checks are involved (`app/nodes/grade_docs.py`,
`app/nodes/grade_answer.py`), so a typical question costs ~4 LLM calls end-to-end.
Every LLM call also goes through `app/llm.py:safe_invoke`, which retries
automatically on a 429 using the provider's own suggested wait time
(`app/rate_limit.py`) — a rate limit becomes a short automatic pause, not a crash.
No paid plan is required for normal use of this project.

## Project layout

```
app/
  config.py          # env-driven settings
  state.py           # typed GraphState
  llm.py             # model + embedding factories
  prompts.py         # all prompts
  graph.py           # StateGraph: nodes, conditional edges, loops
  api.py             # FastAPI service
  retrieval/
    ingest.py        # arXiv -> chunks -> Chroma + BM25
    hybrid.py        # dense + sparse + RRF + cross-encoder rerank
  nodes/             # one file per graph node
eval/ragas_eval.py   # RAGAS harness + golden set
ui/streamlit_app.py  # streaming chat UI
scripts/ingest_arxiv.py
```

## Notes & extensions

- Swap Chroma for **pgvector** by replacing the store in `retrieval/` — a good
  resume signal (RAG in plain Postgres). The hybrid/RRF/rerank layer is store-agnostic.
- The compiled graph uses an in-memory checkpointer; switch to `SqliteSaver` /
  `PostgresSaver` for resumable, multi-request sessions and human-in-the-loop gates
  (`interrupt_before`).
- Add multi-hop decomposition by inserting a planner node before `retrieve`.
