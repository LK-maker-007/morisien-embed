"""Evaluate a SentenceTransformer model on the Kreol Morisien retrieval benchmark."""

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
    parser.add_argument(
        "--revision", default=None, help="Hub revision of the model, a branch, tag or commit. Local paths ignore it."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/data/eng"))
    parser.add_argument("--query-prompt", default="")
    parser.add_argument("--corpus-prompt", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--truncate-dim", type=int, default=None, help="truncate embeddings (Matryoshka dims)")
    args = parser.parse_args()

    bench = benchmark.load(args.data_dir)
    model = SentenceTransformer(args.model, revision=args.revision, truncate_dim=args.truncate_dim)
    results = benchmark.evaluate(
        model,
        bench,
        name=args.data_dir.name,
        query_prompt=args.query_prompt or None,
        corpus_prompt=args.corpus_prompt or None,
        batch_size=args.batch_size,
    )

    scores = {
        label: next((v for key, v in results.items() if key.endswith(suffix)), None) for label, suffix in REPORTED
    }
    missing = [label for label, value in scores.items() if value is None]
    if missing:
        raise RuntimeError(f"expected cosine {', '.join(missing)} in evaluator output, got: {sorted(results)}")

    print(f"\n{args.model}")
    for label, value in scores.items():
        print(f"  {label:12} {value:.4f}")


if __name__ == "__main__":
    main()
