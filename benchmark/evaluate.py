"""Evaluate a SentenceTransformer model on the Kreol Morisien retrieval benchmark.

Loads the ``(queries, corpus, qrels)`` artifacts written by ``build_benchmark.py`` and runs
``InformationRetrievalEvaluator``. Retrieval prompts are passed explicitly so E5-style models get
their required ``query:`` / ``passage:`` prefixes while prefix-free models (LaBSE, MiniLM) get none.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

REPORTED = (
    ("accuracy@1", "cosine_accuracy@1"),
    ("accuracy@10", "cosine_accuracy@10"),
    ("mrr@10", "cosine_mrr@10"),
    ("ndcg@10", "cosine_ndcg@10"),
    ("map@100", "cosine_map@100"),
)


def load_benchmark(data_dir: Path) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    def load_jsonl(name: str) -> list[dict[str, str]]:
        return [json.loads(line) for line in (data_dir / name).read_text(encoding="utf-8").splitlines()]

    queries = {row["_id"]: row["text"] for row in load_jsonl("queries.jsonl")}
    corpus = {row["_id"]: row["text"] for row in load_jsonl("corpus.jsonl")}
    qrels = json.loads((data_dir / "qrels.json").read_text(encoding="utf-8"))
    relevant_docs = {qid: set(cids) for qid, cids in qrels.items()}
    return queries, corpus, relevant_docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/data"))
    parser.add_argument("--query-prompt", default="")
    parser.add_argument("--corpus-prompt", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    queries, corpus, relevant_docs = load_benchmark(args.data_dir)
    model = SentenceTransformer(args.model)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        query_prompt=args.query_prompt or None,
        corpus_prompt=args.corpus_prompt or None,
        batch_size=args.batch_size,
        show_progress_bar=True,
        name="kreol-morisien-retrieval",
    )
    results = evaluator(model)

    print(f"\n{args.model}")
    for label, suffix in REPORTED:
        value = next((v for k, v in results.items() if k.endswith(suffix)), None)
        print(f"  {label:12} {value:.4f}" if value is not None else f"  {label:12} n/a")


if __name__ == "__main__":
    main()
