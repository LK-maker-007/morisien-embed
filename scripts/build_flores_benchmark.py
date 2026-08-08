"""Build an independent Creole retrieval benchmark from the FLORES+ devtest split.

FLORES+ (``openlanguagedata/flores_plus``, CC-BY-SA-4.0, gated) added Mauritian Creole in 2025:
professionally translated wikinews sentences, native-reviewed, aligned across languages by ``id``.
Unlike the MorisienMT test split, this data shares no source with the training corpora, so scores
here measure generalization to an independent domain. Any pair whose Creole sentence nevertheless
appears in the training set is dropped — exactly and punctuation-insensitively — and reported.

Access requires accepting the dataset's terms on the Hugging Face Hub and an ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

from morisien_embed import benchmark, data

FLORES_REPO = "openlanguagedata/flores_plus"
LANG_FILES = {"mfe": "mfe_Latn.jsonl", "eng": "eng_Latn.jsonl", "fra": "fra_Latn.jsonl"}


def flores_split(lang: str, split: str) -> dict[int, str]:
    """Map sentence ``id`` to text for one language in one split."""
    path = hf_hub_download(FLORES_REPO, f"{split}/{LANG_FILES[lang]}", repo_type="dataset")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return {row["id"]: data.normalize(row["text"]) for row in rows}


def loose(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


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
    train_exact = {row["creole"].lower() for row in train_rows}
    train_loose = {loose(creole) for creole in train_exact}
    clean = [
        pair
        for pair in pairs
        if pair["creole"].lower() not in train_exact and loose(pair["creole"]) not in train_loose
    ]
    print(f"flores {args.split} aligned pairs: {len(pairs)}   leaked (dropped): {len(pairs) - len(clean)}")

    out_dir = args.output_dir or Path("benchmark/data") / f"flores-{args.target}"
    bench = benchmark.build(clean, target_lang=args.target)
    benchmark.write(out_dir, bench)

    queries, corpus, _ = bench
    print(f"[flores {args.target}] queries: {len(queries)}  corpus: {len(corpus)}  ->  {out_dir}")


if __name__ == "__main__":
    main()
