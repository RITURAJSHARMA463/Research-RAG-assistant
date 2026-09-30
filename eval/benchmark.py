"""Latency benchmark for the RAG graph.

Produces the numbers you can actually put on a resume:
  - end-to-end latency: mean / p50 / p95 over N runs
  - per-node latency breakdown (which node dominates -> what to optimize)
  - tokens/cost are best read from LangSmith; this script focuses on wall-clock.

Run it twice with a config changed (e.g. USE_RERANKER true vs false, or before/after
parallelizing the grader) to get a defensible "reduced X by Y%" delta.

Usage:
    python -m eval.benchmark --runs 20
    USE_RERANKER=false python -m eval.benchmark --runs 20   # A/B against the default
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict

from app.config import settings
from app.graph import get_app

QUESTIONS = [
    "What problem does the attention mechanism solve compared to recurrence?",
    "How does retrieval-augmented generation reduce hallucination?",
    "What is the role of a reranker in a RAG pipeline?",
    "How does reciprocal rank fusion combine retrievers?",
    "What are common failure modes of dense retrieval?",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run(n_runs: int) -> None:
    app = get_app()

    def _config(tag: str) -> dict:
        return {"configurable": {"thread_id": f"bench-{tag}"},
                "recursion_limit": settings.recursion_limit}

    # Warm-up (excludes one-time model load / first-call download from the numbers).
    print("Warming up...")
    app.invoke({"question": QUESTIONS[0], "original_question": QUESTIONS[0]}, _config("warm"))

    e2e: list[float] = []
    node_times: dict[str, list[float]] = defaultdict(list)

    print(f"Running {n_runs} iterations...")
    for i in range(n_runs):
        q = QUESTIONS[i % len(QUESTIONS)]
        last = time.perf_counter()
        start = last
        for update in app.stream(
            {"question": q, "original_question": q},
            _config(f"{i}"),
            stream_mode="updates",
        ):
            now = time.perf_counter()
            for node_name in update:
                node_times[node_name].append(now - last)
            last = now
        e2e.append(time.perf_counter() - start)

    print("\n=== End-to-end latency (seconds) ===")
    print(f"  mean : {statistics.mean(e2e):.2f}")
    print(f"  p50  : {_percentile(e2e, 0.50):.2f}")
    print(f"  p95  : {_percentile(e2e, 0.95):.2f}")
    print(f"  min  : {min(e2e):.2f}   max: {max(e2e):.2f}")

    print("\n=== Per-node mean latency (seconds) — your optimization targets ===")
    rows = sorted(((sum(v) / len(v), k) for k, v in node_times.items()), reverse=True)
    total = sum(mean for mean, _ in rows) or 1.0
    for mean, name in rows:
        print(f"  {name:18s} {mean:6.2f}   ({100 * mean / total:4.1f}% of node time)")

    print(f"\nConfig: reranker={settings.use_reranker}  embeddings={settings.embeddings_provider}"
          f"  final_k={settings.final_k}  llm={settings.llm_provider}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    main_args = parser.parse_args()
    run(main_args.runs)


if __name__ == "__main__":
    main()
