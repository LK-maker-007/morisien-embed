"""Fine-tune a multilingual base into a Mauritian Creole embedding model.

Trains with MultipleNegativesRankingLoss on the merged Creole↔{English,French} pairs: each Creole
sentence is the anchor and its translation the positive, with the rest of the batch as in-batch
negatives. The dev split scores retrieval each epoch; the held-out test split is scored once at the
end, next to LaBSE for reference. No E5 prefix is applied — it scores best without one on this task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

from morisien_embed import benchmark, data

LABSE_REFERENCE = "LaBSE test: accuracy@1=0.9090  ndcg@10=0.9393"


def load_training_pairs(path: Path, limit: int | None) -> Dataset:
    """Load (anchor=creole, positive=translation) pairs. Column order is the MNRL contract."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit:
        rows = rows[:limit]
    return Dataset.from_dict(
        {"anchor": [row["creole"] for row in rows], "positive": [row["translation"] for row in rows]}
    )


def dev_evaluator() -> InformationRetrievalEvaluator:
    queries, corpus, qrels = benchmark.build(data.morisienmt("dev"))
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=qrels, name="morisienmt-dev", batch_size=64
    )


def report_test(model: SentenceTransformer) -> None:
    results = benchmark.evaluate(model, benchmark.build(data.morisienmt("test"), target_lang="eng"))
    acc = next(v for k, v in results.items() if k.endswith("cosine_accuracy@1"))
    ndcg = next(v for k, v in results.items() if k.endswith("cosine_ndcg@10"))
    print(f"\nFINAL (Creole->English test): accuracy@1={acc:.4f}  ndcg@10={ndcg:.4f}")
    print(f"  reference: {LABSE_REFERENCE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="intfloat/multilingual-e5-small")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/morisien-embed"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="cap training pairs (for smoke tests)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model = SentenceTransformer(args.base)
    train_dataset = load_training_pairs(args.train_file, args.limit)
    loss = MultipleNegativesRankingLoss(model)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=0.1,
        fp16=args.fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        seed=args.seed,
        run_name="morisien-embed",
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator(),
    )
    trainer.train()

    final_dir = args.output_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"saved model -> {final_dir}")
    report_test(model)


if __name__ == "__main__":
    main()
