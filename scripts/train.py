"""Fine-tune a multilingual base into a Mauritian Creole embedding model."""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter
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


def load_training_pairs(
    path: Path, limit: int | None, min_words: int = 1, sample: int | None = None, sample_seed: int = 0
) -> Dataset:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if min_words > 1:
        kept = [row for row in rows if len(row["creole"].split()) >= min_words]
        print(f"min-words {min_words}: kept {len(kept)} of {len(rows)} pairs")
        rows = kept
    if sample is not None:
        if sample > len(rows):
            raise ValueError(f"--sample {sample} exceeds the {len(rows)} available pairs")
        rows = random.Random(sample_seed).sample(rows, sample)
        print(f"sampled {len(rows)} pairs with seed {sample_seed}")
    if limit is not None:
        rows = rows[:limit]
    return Dataset.from_dict(
        {"anchor": [row["creole"] for row in rows], "positive": [row["translation"] for row in rows]}
    )


def dropped_rows(pairs: Dataset, mined: Dataset) -> list[dict[str, str]]:
    kept = Counter(zip(mined["anchor"], mined["positive"], strict=True))
    dropped = []
    for anchor, positive in zip(pairs["anchor"], pairs["positive"], strict=True):
        if kept[(anchor, positive)]:
            kept[(anchor, positive)] -= 1
        else:
            dropped.append({"anchor": anchor, "positive": positive})
    return dropped


def mine_negatives(
    pairs: Dataset,
    mining_model: str,
    num_negatives: int,
    range_min: int,
    range_max: int | None,
    relative_margin: float,
    dropped_path: Path | None = None,
) -> Dataset:
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

    dropped = dropped_rows(pairs, mined)
    # A disagreement here means the input and output could not be matched up, which would make the
    # written file wrong rather than merely incomplete.
    assert len(dropped) == len(pairs) - len(mined), (
        f"recovered {len(dropped)} dropped rows but the miner returned {len(pairs) - len(mined)} fewer"
    )
    print(f"negatives shortfall: {len(dropped)} of {len(pairs)} pairs ({len(dropped) / len(pairs):.2%})")
    if dropped_path is not None:
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        with dropped_path.open("w", encoding="utf-8") as handle:
            for row in dropped:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {dropped_path}")
    return mined


def matryoshka_dims(full_dim: int) -> list[int]:
    return [full_dim, *(dim for dim in (512, 256, 128, 64) if dim < full_dim)]


def apply_lora(model: SentenceTransformer, rank: int) -> None:
    from peft import LoraConfig

    model.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.05,
            bias="none",
        )
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA r={rank}: {trainable / 1e6:.2f}M of {total / 1e6:.0f}M parameters trainable")


def merge_lora(model: SentenceTransformer) -> None:
    from peft.tuners.lora import LoraLayer

    inner = model[0].auto_model
    for module in inner.modules():
        if isinstance(module, LoraLayer):
            module.merge()

    def strip(parent) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, LoraLayer):
                setattr(parent, name, child.get_base_layer())
            else:
                strip(child)

    strip(inner)
    inner._hf_peft_config_loaded = False
    if hasattr(inner, "peft_config"):
        del inner.peft_config


def dev_evaluator() -> InformationRetrievalEvaluator:
    queries, corpus, qrels = benchmark.build(data.morisienmt("dev"), target_lang="eng")
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=qrels, name="morisienmt-dev-eng", batch_size=64
    )


def report_test(model: SentenceTransformer) -> None:
    results = benchmark.evaluate(model, benchmark.build(data.morisienmt("test"), target_lang="eng"))
    acc = next((v for k, v in results.items() if k.endswith("cosine_accuracy@1")), None)
    ndcg = next((v for k, v in results.items() if k.endswith("cosine_ndcg@10")), None)
    if acc is None or ndcg is None:
        raise RuntimeError(f"expected cosine accuracy@1 and ndcg@10 in evaluator output, got: {sorted(results)}")
    print(f"\nFINAL (Creole->English test): accuracy@1={acc:.4f}  ndcg@10={ndcg:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="intfloat/multilingual-e5-base")
    parser.add_argument(
        "--base-revision",
        default=None,
        help="Hub revision of the model, a branch, tag or commit. Local paths ignore it.",
    )
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/morisien-embed"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="cap training pairs (for smoke tests)")
    parser.add_argument("--min-words", type=int, default=1, help="drop pairs whose Creole side is shorter")
    parser.add_argument("--sample", type=int, default=None, help="random subset size, applied after --min-words")
    parser.add_argument("--sample-seed", type=int, default=0, help="seed for --sample, separate from --seed")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mine-with", default=None, help="model to mine hard negatives with; enables cached loss")
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--range-min", type=int, default=10)
    parser.add_argument("--range-max", type=int, default=None, help="widen candidate pool to avoid negative shortfall")
    parser.add_argument(
        "--dropped-negatives",
        type=Path,
        default=None,
        help="write the pairs mining discarded to this JSONL, so the shortfall is a file not a log line",
    )
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument(
        "--cached-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force gradient caching on or off; defaults to on when --mine-with is given",
    )
    parser.add_argument("--matryoshka", action="store_true", help="train truncatable Matryoshka embeddings")
    parser.add_argument("--lora", action="store_true", help="train a LoRA adapter instead of every weight")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write per-epoch checkpoints; disable to save only the final model (checkpoints cost ~3x model size)",
    )
    args = parser.parse_args()

    train_dataset = load_training_pairs(args.train_file, args.limit, args.min_words, args.sample, args.sample_seed)
    if args.mine_with:
        train_dataset = mine_negatives(
            train_dataset,
            args.mine_with,
            args.num_negatives,
            args.range_min,
            args.range_max,
            args.relative_margin,
            args.dropped_negatives,
        )

    model = SentenceTransformer(args.base, revision=args.base_revision)
    if args.lora:
        apply_lora(model, args.lora_r)
    # Gradient caching is what lets a large batch fit, so it is worth having without mining too:
    # plain MNRL at batch 128 holds every pair in one graph and runs out of memory on a 16 GB card.
    cached = args.cached_loss if args.cached_loss is not None else bool(args.mine_with)
    loss = (
        CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
        if cached
        else MultipleNegativesRankingLoss(model)
    )
    if args.matryoshka:
        loss = MatryoshkaLoss(model, loss, matryoshka_dims=matryoshka_dims(model.get_embedding_dimension()))

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
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

    if args.lora:
        merge_lora(model)

    final_dir = args.output_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"saved model -> {final_dir}")
    report_test(model)


if __name__ == "__main__":
    main()
