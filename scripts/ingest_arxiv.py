"""CLI to build the index from arXiv.

Examples:
    python -m scripts.ingest_arxiv "retrieval augmented generation" --max 20
    python -m scripts.ingest_arxiv "attention is all you need transformers" --max 15
"""
from __future__ import annotations

import argparse

from app.retrieval.ingest import build_index


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the hybrid index.")
    parser.add_argument("query", help="arXiv search query / topic")
    parser.add_argument("--max", type=int, default=15, help="max papers to fetch")
    args = parser.parse_args()
    build_index(args.query, max_papers=args.max)


if __name__ == "__main__":
    main()
