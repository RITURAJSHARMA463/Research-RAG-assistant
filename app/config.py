"""Central configuration. All knobs are environment-driven so the same code runs
locally (mostly offline) or against hosted APIs without edits."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# override=True: .env is authoritative. python-dotenv's default (override=False)
# leaves any variable already exported in your shell untouched — so if
# EMBEDDINGS_PROVIDER=gemini was ever exported in this terminal session (e.g. from
# an earlier one-off command like `EMBEDDINGS_PROVIDER=gemini python -m ...`), the
# .env file's value would be silently ignored for the rest of that session. This is
# a single-user local project, not a multi-environment deployment, so we want the
# .env file to always win rather than debugging invisible shell state.
load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _b(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Models ---------------------------------------------------------------
    # LLM provider drives the router / graders / generator. "gemini", "openai", or "anthropic".
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Embeddings: "hf" (local, free, default — avoids the Gemini free-tier quota
    # entirely since indexing hundreds/thousands of chunks needs many embed calls),
    # "gemini", or "openai".
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "hf")
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Reranker: local cross-encoder, no API key required.
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    use_reranker: bool = _b("USE_RERANKER", "true")

    # Only relevant if EMBEDDINGS_PROVIDER=gemini: that free tier caps embed_content
    # at ~100 requests/minute (project-wide), so a bulk ingest can exceed it. These
    # knobs batch and pace requests client-side and retry 429s with backoff. Unused
    # (and harmless) with the default "hf" provider.
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "90"))
    embed_batch_delay: float = float(os.getenv("EMBED_BATCH_DELAY", "1.5"))
    embed_max_retries: int = int(os.getenv("EMBED_MAX_RETRIES", "6"))

    # Free-tier chat model quotas (e.g. Gemini's 5-15 req/min) are much tighter than
    # embedding quotas. Every LLM call in the graph goes through safe_invoke(), which
    # retries on 429 with the provider's own suggested wait time.
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "6"))

    # --- Retrieval ------------------------------------------------------------
    collection_name: str = os.getenv("COLLECTION_NAME", "arxiv_papers")
    chroma_dir: str = os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma"))
    bm25_path: str = os.getenv("BM25_PATH", str(DATA_DIR / "bm25.pkl"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1100"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    dense_k: int = int(os.getenv("DENSE_K", "12"))
    sparse_k: int = int(os.getenv("SPARSE_K", "12"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))          # RRF damping constant
    final_k: int = int(os.getenv("FINAL_K", "5"))        # docs handed to the LLM

    # --- Web search fallback --------------------------------------------------
    tavily_max_results: int = int(os.getenv("TAVILY_MAX_RESULTS", "4"))

    # --- Self-correction guards ----------------------------------------------
    max_rewrites: int = int(os.getenv("MAX_REWRITES", "2"))
    max_generations: int = int(os.getenv("MAX_GENERATIONS", "2"))
    recursion_limit: int = int(os.getenv("RECURSION_LIMIT", "25"))

    # --- Keys (read lazily by the integrations) ------------------------------
    # Gemini: langchain-google-genai reads GOOGLE_API_KEY first, then GEMINI_API_KEY
    # as a fallback. We mirror that same precedence here for consistency.
    keys: dict = field(default_factory=lambda: {
        "gemini": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "tavily": os.getenv("TAVILY_API_KEY"),
    })


settings = Settings()
