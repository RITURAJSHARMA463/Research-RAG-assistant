# Running & Deploying the Research RAG Assistant

This guide has two parts: **running locally** (the normal dev path) and
**deploying** (Docker + cloud). Read the memory note before you pick a host —
this app is heavier than a typical web service.

---

## Part 1 — Run locally

### Prerequisites
- Python 3.10–3.12
- An LLM API key: `GOOGLE_API_KEY` for Gemini (default — get one free at
  aistudio.google.com/app/apikey), or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` with
  `LLM_PROVIDER` set accordingly
- (Optional) `TAVILY_API_KEY` for the live web-search fallback
- ~3 GB free disk on first run (the local embedding + reranker models download once)

### Steps

1. **Create a virtualenv and install:**
   ```bash
   python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set `GOOGLE_API_KEY` (get one at aistudio.google.com/app/apikey).
   Defaults already use **local** embeddings
   and a **local** reranker, so only the LLM nodes cost money.

3. **Build the index from arXiv** (downloads PDFs → chunks → dense + BM25 index):
   ```bash
   python -m scripts.ingest_arxiv "retrieval augmented generation" --max 20
   ```
   The index is written to `./data` (Chroma + `bm25.pkl`). Re-run with a different
   topic to extend the corpus. First run also downloads the embedding/reranker models.

4. **Run it** — pick one:

   **UI (recommended for a demo):**
   ```bash
   streamlit run ui/streamlit_app.py        # http://localhost:8501
   ```

   **API:**
   ```bash
   uvicorn app.api:app --reload             # http://localhost:8000
   curl -s localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question":"How does a reranker improve retrieval precision?"}' | jq
   ```

   **Python:**
   ```python
   from app.graph import answer
   print(answer("What problem does attention solve vs. recurrence?"))
   ```

5. **(Optional) Evaluate:**
   ```bash
   cp eval/golden_set.example.json eval/golden_set.json
   python -m eval.ragas_eval --golden eval/golden_set.json
   ```

### Run locally with Docker (no Python setup)
```bash
python -m scripts.ingest_arxiv "your topic" --max 20    # build ./data first (needs deps)
docker compose up --build                                # API :8000, UI :8501
```
The compose file mounts `./data` and caches downloaded models in a volume.

---

## Part 2 — Deploy

### ⚠️ Read this first: memory footprint
The **default (full-fidelity) profile** loads a cross-encoder reranker and a local
embedding model — both pull in PyTorch. Resident memory is roughly **2 GB**. That does
**not** fit on small free tiers (e.g. a free Render web service is 512 MB RAM and spins
down after 15 minutes idle). You have two real options:

| Profile | What runs | RAM needed | Where it fits |
|---|---|---|---|
| **A. Full fidelity** | local embeddings + cross-encoder rerank | ~2 GB | Fly.io (1–2GB machine), Railway, a 2GB+ VPS, Render Standard |
| **B. Lightweight cloud** | OpenAI embeddings, reranker off (or Cohere API) | ~0.5–1 GB | Render free/Starter, most small hosts |

Profile B is already wired: `requirements-cloud.txt` (no torch) +
`EMBEDDINGS_PROVIDER=openai` + `USE_RERANKER=false`. You keep dense + BM25 + RRF; you
lose the local cross-encoder. To keep reranking in Profile B, add Cohere (below).

> **Embeddings must match.** The index and the queries must use the *same* embedding
> model. If you deploy Profile B (OpenAI embeddings), build the index with OpenAI
> embeddings too — don't ship an index built locally with the HF model.

### The index: how to get it onto the server
The index in `./data` must exist at runtime. Two patterns:

- **Persistent disk + ingest once (recommended).** Attach a disk mounted at `/app/data`,
  deploy, then run the ingest command once via the host's shell. The index survives restarts.
- **Bake into the image.** Build the index locally, comment out `data` in `.dockerignore`,
  and `COPY` it into the image. Simplest for a read-only demo; rebuild to refresh the corpus.

---

### Option 1 — Render (Profile B, via the included `render.yaml`)
1. Push the repo to GitHub.
2. In Render: **New → Blueprint**, point at the repo. It reads `render.yaml`:
   a Docker web service on a Starter instance with a 1 GB disk at `/app/data`.
3. Set secrets in the dashboard: `OPENAI_API_KEY` (and optionally `TAVILY_API_KEY`).
   Render exposes service env vars as Docker build args, so `REQS=requirements-cloud.txt`
   selects the lightweight image.
4. After the first deploy, open the service **Shell** and build the index once:
   ```bash
   EMBEDDINGS_PROVIDER=openai python -m scripts.ingest_arxiv "your topic" --max 20
   ```
5. Hit `https://<service>.onrender.com/health`, then `/chat`.

For **Profile A** on Render, change `plan: starter` → `plan: standard` (≥2 GB) and set
`REQS=requirements.txt`, `EMBEDDINGS_PROVIDER=hf`, `USE_RERANKER=true`.

### Option 2 — Fly.io (good for Profile A — full fidelity)
```bash
fly launch --no-deploy                     # generates fly.toml; choose a region
fly volumes create index --size 2          # persistent disk for ./data
# In fly.toml: mount the volume at /app/data, set [http_service] internal_port = 8000
fly secrets set OPENAI_API_KEY=sk-... TAVILY_API_KEY=tvly-...
fly deploy
fly ssh console -C "python -m scripts.ingest_arxiv 'your topic' --max 20"
```
Scale memory if the reranker OOMs: `fly scale memory 2048`.

### Option 3 — Any VPS (full control, cheapest for always-on)
A 2 GB VPS runs the whole stack via Docker Compose:
```bash
git clone <repo> && cd research-rag-assistant
cp .env.example .env && nano .env          # add keys
docker compose run --rm api python -m scripts.ingest_arxiv "your topic" --max 20
docker compose up -d                        # API :8000, UI :8501
```
Put Caddy or Nginx in front for TLS.

### The Streamlit UI in the cloud
The included UI runs the graph **in-process**, so hosting it needs the same heavy deps
and the index. Two clean choices:
- **One box:** run API + UI together via Compose on a VPS (Option 3).
- **Split:** deploy the API anywhere, then change `ui/streamlit_app.py` to POST to
  `<api-url>/chat` instead of calling `get_app()` — now the UI is a thin client you can
  host on Streamlit Community Cloud for free.

---

## Optional: keep reranking in Profile B (Cohere API)
Avoids torch while keeping a reranker. Add `langchain-cohere` (already commented in
`requirements-cloud.txt`), set `COHERE_API_KEY`, and in `app/retrieval/hybrid.py`
replace the `_rerank` body with a call to Cohere Rerank over the fused candidates.

## Troubleshooting
- **`429 RESOURCE_EXHAUSTED` on `generate_content` (chat model, e.g. during grading
  or generation)** → the LLM free tier's requests-per-minute cap (can be as low as
  5/min for some Gemini models). The graph batches document grading and the answer
  check into one call each (not one per item) to minimize call volume, and every
  call auto-retries with the provider's suggested wait via `app/rate_limit.py` — it
  will pause and continue rather than failing. If you're asking several questions
  back-to-back and hitting this often, wait ~60s between questions, or set
  `GEMINI_MODEL=gemini-2.5-flash-lite` for a typically higher free-tier ceiling.
- **`429 RESOURCE_EXHAUSTED` during ingestion (Gemini)** → expected on the free tier,
  which caps `embed_content` at ~100 requests/minute per project. Ingestion batches
  and paces requests automatically and retries on 429 (see `EMBED_BATCH_SIZE` /
  `EMBED_BATCH_DELAY` / `EMBED_MAX_RETRIES` in `.env`) — it will slow down and recover
  on its own rather than failing the run. For a large corpus, either be patient, lower
  `--max` in the ingest command, or move to a paid Gemini tier for higher limits.
  Better yet, the default `EMBEDDINGS_PROVIDER=hf` avoids this entirely by embedding
  locally — only switch to `gemini`/`openai` embeddings if you specifically want to.
- **Killed / OOM on deploy** → you're on Profile A with too little RAM. Use Profile B or a ≥2 GB instance.
- **First request hangs ~30–60s** → free hosts cold-start after idle; or the HF model is
  downloading on first call (bake models in the image — see the Dockerfile comment).
- **`Index not found`** → run the ingest command on the host (the disk was empty).
- **Empty/odd citations** → corpus too small or off-topic; ingest more papers on the subject.
- **Port errors** → the container binds `$PORT`; don't hardcode 8000 on hosts that inject it.
