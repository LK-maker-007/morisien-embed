"""Evaluate a SentenceTransformer model on the Kreol Morisien retrieval benchmark.

Retrieval prompts are passed explicitly so E5-style models get their prefixes while prefix-free models
(LaBSE, MiniLM) get none. For this bitext task, E5 scores best with no prefix (verified empirically).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from morisien_embed import benchmark

REPORTED = (
    ("accuracy@1", "cosine_accuracy@1"),
    ("accuracy@10", "cosine_accuracy@10"),
    ("mrr@10", "cosine_mrr@10"),
    ("ndcg@10", "cosine_ndcg@10"),
    ("map@100", "cosine_map@100"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/data/eng"))
    parser.add_argument("--query-prompt", default="")
    parser.add_argument("--corpus-prompt", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    bench = benchmark.load(args.data_dir)
    model = SentenceTransformer(args.model)
    results = benchmark.evaluate(
        model,
        bench,
        name=args.data_dir.name,
        query_prompt=args.query_prompt or None,
        corpus_prompt=args.corpus_prompt or None,
        batch_size=args.batch_size,
    )

    print(f"\n{args.model}")
    for label, suffix in REPORTED:
        value = next((v for key, v in results.items() if key.endswith(suffix)), None)
        print(f"  {label:12} {value:.4f}" if value is not None else f"  {label:12} n/a")


if __name__ == "__main__":
    main()
