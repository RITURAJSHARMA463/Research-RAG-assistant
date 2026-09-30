"""Factories for the chat model and embeddings, so providers are swappable
from config without touching node code."""
from __future__ import annotations

import time
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.rate_limit import call_with_retry


class RateLimitedEmbeddings(Embeddings):
    """Wraps any Embeddings model to batch requests and pace them under a
    requests-per-minute ceiling, retrying with backoff on 429s.

    Written for Gemini's free tier (embed_content capped at ~100 req/min per
    project), but harmless for any provider — it just adds a small delay between
    batches and turns a hard failure into an automatic wait-and-retry.
    """

    def __init__(self, base: Embeddings, batch_size: int, delay_seconds: float, max_retries: int):
        self._base = base
        self._batch_size = max(1, batch_size)
        self._delay = max(0.0, delay_seconds)
        self._max_retries = max(1, max_retries)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        batches = [texts[i:i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        for i, batch in enumerate(batches):
            vectors.extend(call_with_retry(
                self._base.embed_documents, batch,
                max_retries=self._max_retries, label="embedding",
            ))
            if i < len(batches) - 1 and self._delay:
                time.sleep(self._delay)  # stay under the requests-per-minute ceiling
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return call_with_retry(
            self._base.embed_query, text,
            max_retries=self._max_retries, label="embedding",
        )


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None):
    """Return a chat model. Supports Gemini, OpenAI, and Anthropic.

    All graders use temperature 0; only the generator may run warmer.
    """
    temp = settings.llm_temperature if temperature is None else temperature
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=temp,
            google_api_key=settings.keys.get("gemini"),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.openai_model, temperature=temp)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.anthropic_model, temperature=temp)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def safe_invoke(runnable, messages):
    """Invoke any LangChain runnable (a chat model, or the result of
    `.with_structured_output(...)`) with automatic retry-with-backoff on 429s.

    Every node should call this instead of `runnable.invoke(...)` directly, so a
    free-tier rate limit becomes an automatic wait instead of a crashed graph run.
    """
    return call_with_retry(
        runnable.invoke, messages,
        max_retries=settings.llm_max_retries, label="LLM call",
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Return an embeddings model. Defaults to a local HF model (free, offline, no
    rate limits — the right choice for bulk-embedding hundreds/thousands of chunks
    at ingest time). Gemini and OpenAI are supported as opt-in alternatives.

    The Gemini path is wrapped in RateLimitedEmbeddings since its free tier enforces
    a per-minute request cap that a bulk ingest will exceed; irrelevant for "hf".
    """
    provider = settings.embeddings_provider.lower()
    if provider == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        base = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.keys.get("gemini"),
        )
        return RateLimitedEmbeddings(
            base,
            batch_size=settings.embed_batch_size,
            delay_seconds=settings.embed_batch_delay,
            max_retries=settings.embed_max_retries,
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.openai_embedding_model)
    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {settings.embeddings_provider}")
