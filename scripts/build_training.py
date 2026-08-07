"""Assemble the merged, leak-free Mauritian Creole training set and write it as JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from morisien_embed import data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/train.jsonl"))
    args = parser.parse_args()

    raw = data.morisienmt("train") + data.kreyol_mt("train")
    kept, dropped = data.merge(raw, data.reserved_creole())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in kept:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    langs = Counter(pair["lang"] for pair in kept)
    print(f"raw pairs:       {len(raw):>7}")
    print(f"dropped (leak):  {dropped['leak']:>7}")
    print(f"dropped (dup):   {dropped['duplicate']:>7}")
    print(f"training pairs:  {len(kept):>7}   {dict(langs)}")


if __name__ == "__main__":
    main()
