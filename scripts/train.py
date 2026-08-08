"""Fine-tune a multilingual base into a Mauritian Creole embedding model.

Trains with in-batch-negatives contrastive learning on the merged Creole↔{English,French} pairs:
each Creole sentence is the anchor and its translation the positive. The dev split scores retrieval
each epoch; the held-out test split is scored once at the end, next to LaBSE for reference. No E5
prefix is applied — it scores best without one on this task.

Passing ``--mine-with <model>`` switches on the stronger recipe used by state-of-the-art embedding
models: hard negatives are mined with that model (skipping candidates too close to the true positive,
so real positives are not mislabelled as negatives), and training uses
``CachedMultipleNegativesRankingLoss`` so a large batch of negatives fits in memory via gradient
caching. ``--matryoshka`` additionally trains truncatable embeddings that stay accurate at smaller
dimensions.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.util import mine_hard_negatives

from morisien_embed import benchmark, data

LABSE_REFERENCE = "LaBSE test: accuracy@1=0.9090  ndcg@10=0.9393"


def load_training_pairs(path: Path, limit: int | None) -> Dataset:
    """Load (anchor=creole, positive=translation) pairs. Column order is the loss contract."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit:
        rows = rows[:limit]
    return Dataset.from_dict(
        {"anchor": [row["creole"] for row in rows], "positive": [row["translation"] for row in rows]}
    )


def mine_negatives(
    pairs: Dataset,
    mining_model: str,
    num_negatives: int,
    range_min: int,
    range_max: int | None,
    relative_margin: float,
) -> Dataset:
    """Return (anchor, positive, neg_1, …, neg_n) tuples with hard negatives from ``mining_model``.

    ``range_min`` skips the closest matches (which may be paraphrases of the positive) and
    ``relative_margin`` drops any candidate whose similarity comes within that fraction of the
    positive's — both guard against mislabelling a true positive as a negative. ``range_max`` widens
    the candidate pool so every anchor can still reach ``num_negatives`` after that filtering.
    """
    model = SentenceTransformer(mining_model)
    mined = mine_hard_negatives(
        pairs,
        model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        num_negatives=num_negatives,
        range_min=range_min,
        range_max=range_max,
        relative_margin=relative_margin,
        sampling_strategy="top",
        output_format="n-tuple",
        batch_size=256,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mined


def matryoshka_dims(full_dim: int) -> list[int]:
    """Truncation sizes for Matryoshka training: the model's full dimension down to 64."""
    return [full_dim, *(dim for dim in (512, 256, 128, 64) if dim < full_dim)]


def embedding_dim(model: SentenceTransformer) -> int:
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return getter()


def dev_evaluator() -> InformationRetrievalEvaluator:
    """Score the dev split on the same Creole→English task the final test uses, for a matched signal."""
    queries, corpus, qrels = benchmark.build(data.morisienmt("dev"), target_lang="eng")
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=qrels, name="morisienmt-dev-eng", batch_size=64
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
    parser.add_argument("--mine-with", default=None, help="model to mine hard negatives with; enables cached loss")
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--range-min", type=int, default=10)
    parser.add_argument("--range-max", type=int, default=None, help="widen candidate pool to avoid negative shortfall")
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--matryoshka", action="store_true", help="train truncatable Matryoshka embeddings")
    parser.add_argument(
        "--checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write per-epoch checkpoints; disable to save only the final model (checkpoints cost ~3x model size)",
    )
    args = parser.parse_args()

    train_dataset = load_training_pairs(args.train_file, args.limit)
    if args.mine_with:
        train_dataset = mine_negatives(
            train_dataset,
            args.mine_with,
            args.num_negatives,
            args.range_min,
            args.range_max,
            args.relative_margin,
        )

    model = SentenceTransformer(args.base)
    loss = (
        CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
        if args.mine_with
        else MultipleNegativesRankingLoss(model)
    )
    if args.matryoshka:
        loss = MatryoshkaLoss(model, loss, matryoshka_dims=matryoshka_dims(embedding_dim(model)))

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=0.1,
        fp16=args.fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch" if args.checkpoints else "no",
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
