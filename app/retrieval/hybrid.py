"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with Reciprocal Rank
Fusion, then reordered by a cross-encoder reranker.

This is the part that actually moves the needle on RAG quality, and the part worth
talking through in an interview: dense recall + lexical precision + a reranker that
reads the (query, passage) pair jointly rather than comparing pre-computed vectors.
"""
from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.config import settings
from app.llm import get_embeddings

_TOKEN = re.compile(r"\w+")


def _tok(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _load():
    """Load and cache the dense store, BM25 model, and the chunk corpus."""
    from langchain_chroma import Chroma
    from rank_bm25 import BM25Okapi

    if not Path(settings.bm25_path).exists():
        raise FileNotFoundError(
            "Index not found. Run:  python -m scripts.ingest_arxiv \"<topic>\""
        )

    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )
    with open(settings.bm25_path, "rb") as f:
        corpus: List[Document] = pickle.load(f)
    bm25 = BM25Okapi([_tok(d.page_content) for d in corpus])
    return store, bm25, corpus


def _key(doc: Document) -> str:
    return f"{doc.metadata.get('paper_id')}::{hash(doc.page_content)}"


def _rrf(rank_lists: List[List[Document]], k: int) -> List[Document]:
    """Reciprocal Rank Fusion across an arbitrary number of ranked lists."""
    scores: dict[str, float] = {}
    keep: dict[str, Document] = {}
    for ranked in rank_lists:
        for rank, doc in enumerate(ranked):
            key = _key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            keep.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [keep[key] for key in ordered]


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.reranker_model)


def _rerank(query: str, docs: List[Document], top_k: int) -> List[Document]:
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = _reranker().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


def hybrid_search(query: str, final_k: int | None = None) -> List[Document]:
    """Run dense + sparse retrieval, fuse, optionally rerank, return top docs."""
    store, bm25, corpus = _load()
    final_k = final_k or settings.final_k

    # Dense
    dense = store.similarity_search(query, k=settings.dense_k)

    # Sparse (BM25)
    bm25_scores = bm25.get_scores(_tok(query))
    top_idx = sorted(range(len(corpus)), key=lambda i: bm25_scores[i], reverse=True)
    sparse = [corpus[i] for i in top_idx[: settings.sparse_k]]

    fused = _rrf([dense, sparse], settings.rrf_k)

    if settings.use_reranker:
        # Rerank a generous candidate pool, then cut to final_k.
        return _rerank(query, fused[: settings.dense_k + settings.sparse_k], final_k)
    return fused[:final_k]
