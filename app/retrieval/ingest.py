"""Ingest arXiv papers into the hybrid index.

Pipeline: search arXiv -> download PDFs -> extract full text -> recursive chunking
-> (a) embed into a persistent Chroma collection for dense retrieval and
   (b) build a BM25 index over the same chunks for sparse retrieval.

Each chunk carries rich metadata (title, authors, year, url) so the generator can
produce real, clickable citations.
"""
from __future__ import annotations

import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import List

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.llm import get_embeddings

# arxiv>=4.0.0 removed Result.download_pdf()/download_source() — Result now only
# exposes `pdf_url`; downloading is the caller's job. We fetch it ourselves with a
# polite retry loop instead of urllib.request.urlretrieve (no retry/timeout control).
_UA = "research-rag-assistant/0.1 (mailto:you@example.com)"


def _download_pdf(pdf_url: str, dest: Path, num_retries: int = 3, timeout: int = 30) -> None:
    last_err: Exception | None = None
    for attempt in range(num_retries):
        try:
            resp = requests.get(pdf_url, headers={"User-Agent": _UA}, timeout=timeout)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return
        except Exception as exc:  # noqa: BLE001 - retry, then propagate
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {pdf_url}: {last_err}")


def _clean(text: str) -> str:
    text = re.sub(r"-\n", "", text)           # join hyphenated line breaks
    text = re.sub(r"\s*\n\s*", " ", text)      # collapse newlines
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _extract_pdf_text(pdf_path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def fetch_papers(query: str, max_papers: int) -> List[dict]:
    """Return paper records: {paper_id, title, authors, year, url, text}."""
    import arxiv

    client = arxiv.Client(page_size=min(max_papers, 50), delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for result in client.results(search):
            paper_id = result.get_short_id()
            try:
                if not result.pdf_url:
                    raise RuntimeError("no pdf_url on this result")
                fname = f"{paper_id.replace('/', '_')}.pdf"
                dest = Path(tmp) / fname
                _download_pdf(result.pdf_url, dest)
                text = _clean(_extract_pdf_text(str(dest)))
            except Exception as exc:  # noqa: BLE001 - skip a bad paper, keep going
                print(f"  ! skipping {paper_id}: {exc}")
                continue
            if len(text) < 500:
                continue
            authors = [a.name for a in result.authors]
            papers.append({
                "paper_id": paper_id,
                "title": result.title.strip(),
                "authors": authors,
                "year": result.published.year if result.published else None,
                "url": result.entry_id,            # canonical arXiv abs URL
                "text": text,
            })
            print(f"  + {paper_id}  {result.title[:70]}")
    return papers


def _chunk(papers: List[dict]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: List[Document] = []
    for p in papers:
        first_author = p["authors"][0] if p["authors"] else "Unknown"
        cite = f"{first_author} et al. ({p['year']})" if p["year"] else first_author
        for chunk in splitter.split_text(p["text"]):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "authors": ", ".join(p["authors"][:6]),
                    "year": p["year"],
                    "url": p["url"],
                    "citation": cite,
                },
            ))
    return docs


def build_index(query: str, max_papers: int = 15) -> int:
    """End-to-end ingest. Returns the number of chunks indexed."""
    from langchain_chroma import Chroma

    # Loud and early: this is the #1 source of confusion when EMBEDDINGS_PROVIDER
    # in .env doesn't match what you think it is (e.g. a stale value from before
    # switching providers). Fail fast on ambiguity rather than mid-embedding.
    print(f"Config: EMBEDDINGS_PROVIDER={settings.embeddings_provider!r}  "
          f"LLM_PROVIDER={settings.llm_provider!r}")
    if settings.embeddings_provider.lower() == "gemini":
        print("  ! Using Gemini for embeddings — its free tier caps embed_content at "
              "~100 req/min, so a bulk ingest like this can hit 429s. If you hit that, "
              "either wait for the automatic retry or set EMBEDDINGS_PROVIDER=hf in .env "
              "for a rate-limit-free local model (recommended for ingestion).")

    print(f"Searching arXiv for: {query!r}")
    papers = fetch_papers(query, max_papers)
    if not papers:
        raise RuntimeError("No papers fetched — try a broader query.")

    docs = _chunk(papers)
    print(f"Chunked {len(papers)} papers into {len(docs)} passages.")

    # Dense index (Chroma, persisted).
    print("Embedding + writing dense index (Chroma)...")
    Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
    )

    # Sparse index (BM25) — persist the raw docs; retriever tokenizes on load.
    print("Writing sparse index (BM25)...")
    with open(settings.bm25_path, "wb") as f:
        pickle.dump(docs, f)

    print(f"Done. {len(docs)} chunks indexed.")
    return len(docs)
