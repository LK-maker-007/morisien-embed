"""Build an independent Creole retrieval benchmark from the FLORES+ devtest split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from morisien_embed import benchmark
from morisien_embed.data import loose
from morisien_embed.flores import flores_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("eng", "fra"), default="eng")
    parser.add_argument("--split", choices=("dev", "devtest"), default="devtest")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
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

    out_dir = args.output_dir or Path("benchmark/data") / f"flores-{args.target}"
    bench = benchmark.build(clean, target_lang=args.target)
    benchmark.write(out_dir, bench)

    queries, corpus, _ = bench
    print(f"[flores {args.target}] queries: {len(queries)}  corpus: {len(corpus)}  ->  {out_dir}")


if __name__ == "__main__":
    main()
