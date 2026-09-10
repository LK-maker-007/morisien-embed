"""Build a hard-negative retrieval benchmark from the FLORES+ devtest split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

from morisien_embed import benchmark, data
from morisien_embed.data import loose
from morisien_embed.flores import flores_split


def candidate_pool(target_lang: str, exclude: set[str]) -> list[str]:
    """Translation-side sentences usable as distractors, excluding anything in ``exclude``.

    Drawn from the MorisienMT and Kreyol-MT training splits, so no gold passage and no evaluation
    sentence can enter the pool.
    """
    pairs = data.morisienmt("train") + data.kreyol_mt("train")
    pool: list[str] = []
    seen: set[str] = set()
    for pair in pairs:
        if pair["lang"] != target_lang:
            continue
        text = pair["translation"]
        key = loose(text)
        if not key or key in exclude or key in seen:
            continue
        seen.add(key)
        pool.append(text)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("eng", "fra"), default="eng")
    parser.add_argument("--split", choices=("dev", "devtest"), default="devtest")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--mine-with", default="intfloat/multilingual-e5-base")
    parser.add_argument("--num-negatives", type=int, default=10)
    parser.add_argument("--range-min", type=int, default=1)
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    creole = flores_split("mfe", args.split)
    target = flores_split(args.target, args.split)
    pairs = [
        {"creole": creole[sid], "translation": target[sid], "lang": args.target}
        for sid in sorted(creole.keys() & target.keys())
    ]

    train_rows = [json.loads(line) for line in args.train_file.read_text(encoding="utf-8").splitlines()]
    train_loose = {loose(row["creole"]) for row in train_rows}
    clean = [pair for pair in pairs if loose(pair["creole"]) not in train_loose]
    print(f"flores {args.split} aligned pairs: {len(pairs)}   leaked (dropped): {len(pairs) - len(clean)}")

    gold_keys = {loose(pair["translation"]) for pair in clean}
    pool = candidate_pool(args.target, gold_keys)
    print(f"distractor candidate pool: {len(pool)} unique {args.target} sentences")

    dataset = Dataset.from_dict(
        {"anchor": [pair["creole"] for pair in clean], "positive": [pair["translation"] for pair in clean]}
    )
    mined = mine_hard_negatives(
        dataset,
        SentenceTransformer(args.mine_with),
        corpus=pool,
        num_negatives=args.num_negatives,
        range_min=args.range_min,
        relative_margin=args.relative_margin,
        sampling_strategy="top",
        output_format="n-tuple",
        batch_size=args.batch_size,
    )

    negatives = {
        text for column in mined.column_names if column.startswith("negative") for text in mined[column] if text
    }
    print(f"mined hard negatives: {len(negatives)} unique")

    queries, corpus, qrels = benchmark.build(clean, target_lang=args.target)
    next_id = len(corpus)
    for text in sorted(negatives):
        corpus[f"d{next_id}"] = text
        next_id += 1

    out_dir = args.output_dir or Path("benchmark/data") / f"flores-hard-{args.target}"
    benchmark.write(out_dir, (queries, corpus, qrels))
    print(f"[flores-hard {args.target}] queries: {len(queries)}  corpus: {len(corpus)}  ->  {out_dir}")


if __name__ == "__main__":
    main()
