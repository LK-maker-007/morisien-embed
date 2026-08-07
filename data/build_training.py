"""Assemble the Mauritian Creole training set from every available parallel corpus.

Merges MorisienMT and Kreyòl-MT training pairs, drops any pair whose Creole sentence also appears in
the MorisienMT dev/test benchmark (so no evaluation sentence leaks into training), removes exact
duplicates, and writes the result. The corpora carry mixed licenses, so only the trained model is
released — never this file, which regenerates from the sources.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sources


def blocked_creole() -> set[str]:
    """Lower-cased Creole sentences reserved for evaluation, which training must never contain."""
    reserved: set[str] = set()
    for split in ("dev", "test"):
        reserved.update(pair["creole"].lower() for pair in sources.morisienmt(split))
    return reserved


def merge(pairs: list[dict[str, str]], reserved: set[str]) -> tuple[list[dict[str, str]], Counter]:
    kept: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    dropped: Counter = Counter()
    for pair in pairs:
        if pair["creole"].lower() in reserved:
            dropped["leak"] += 1
            continue
        key = (pair["creole"].lower(), pair["translation"].lower())
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(pair)
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/train.jsonl"))
    args = parser.parse_args()

    raw = sources.morisienmt("train") + sources.kreyol_mt("train")
    kept, dropped = merge(raw, blocked_creole())

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
