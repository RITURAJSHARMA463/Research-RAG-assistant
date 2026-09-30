"""RAGAS evaluation harness.

Runs the full graph over a golden set of questions, collects (question, answer,
retrieved_contexts, reference), and scores it on:
  - Faithfulness                  : is the answer grounded in the retrieved context?
  - ResponseRelevancy             : does the answer address the question?
  - LLMContextPrecisionWithReference / LLMContextRecall : retrieval quality.

The retriever and generator are the two failure surfaces of any RAG system; these
metrics keep them separate so you know which half to fix.

Usage:
    python -m eval.ragas_eval --golden eval/golden_set.json
"""
from __future__ import annotations

import argparse
import json

from app.config import settings
from app.graph import get_app
from app.llm import get_embeddings, get_llm
from app.retrieval.hybrid import hybrid_search


def _run_pipeline(question: str) -> tuple[str, list[str]]:
    """Return (answer, retrieved_contexts) for one question."""
    app = get_app()
    config = {"configurable": {"thread_id": f"eval-{abs(hash(question))}"},
              "recursion_limit": settings.recursion_limit}
    final = app.invoke({"question": question, "original_question": question}, config)
    # Re-derive contexts from the final (possibly rewritten) query for scoring.
    contexts = [d.page_content for d in final.get("documents", [])]
    if not contexts:
        contexts = [d.page_content for d in hybrid_search(question)]
    return final.get("generation", ""), contexts


def build_dataset(golden: list[dict]):
    from ragas import EvaluationDataset

    samples = []
    for row in golden:
        answer, contexts = _run_pipeline(row["question"])
        samples.append({
            "user_input": row["question"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": row.get("reference", ""),
        })
    return EvaluationDataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden_set.json")
    args = parser.parse_args()

    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    with open(args.golden) as f:
        golden = json.load(f)

    print(f"Running pipeline over {len(golden)} golden questions...")
    dataset = build_dataset(golden)

    evaluator_llm = LangchainLLMWrapper(get_llm(temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )
    print("\n=== RAGAS scores ===")
    print(result)
    df = result.to_pandas()
    df.to_csv("eval/ragas_results.csv", index=False)
    print("\nPer-question results written to eval/ragas_results.csv")


if __name__ == "__main__":
    main()
