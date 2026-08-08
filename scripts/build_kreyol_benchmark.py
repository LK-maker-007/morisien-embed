"""Build an independent Kreyòl-MT test benchmark, de-leaked against the training set.

The primary benchmark (MorisienMT test) shares MorisienMT's distribution with the training data, so a
high score there could reflect fitting that style rather than genuine Creole understanding. This
builds a second benchmark from the Kreyòl-MT test split and removes any pair whose Creole sentence
appears in the training set — exactly (case/whitespace) and punctuation-insensitively — so a model's
score here measures generalization, not memorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from morisien_embed import benchmark, data
from morisien_embed.data import loose


def training_creoles(train_file: Path) -> set[str]:
    rows = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines()]
    return {row["creole"].lower() for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--target", choices=("eng", "fra"), default="eng")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    train_exact = training_creoles(args.train_file)
    train_loose = {loose(creole) for creole in train_exact}

    test_pairs = data.kreyol_mt("test")
    clean = [
        pair
        for pair in test_pairs
        if pair["creole"].lower() not in train_exact and loose(pair["creole"]) not in train_loose
    ]
    dropped = len(test_pairs) - len(clean)
    print(f"kreyol-mt test pairs: {len(test_pairs)}   leaked (dropped): {dropped}   clean: {len(clean)}")

    out_dir = args.output_dir or Path("benchmark/data") / f"kreyol-{args.target}"
    bench = benchmark.build(clean, target_lang=args.target)
    benchmark.write(out_dir, bench)

    queries, corpus, _ = bench
    print(f"[kreyol {args.target}] queries: {len(queries)}  corpus: {len(corpus)}  ->  {out_dir}")


if __name__ == "__main__":
    main()
