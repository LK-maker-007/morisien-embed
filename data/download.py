"""Build the Mauritian Creole training and evaluation sets from the Kreyòl-MT corpus.

Kreyòl-MT (`jhu-clsp/kreyol-mt`) aligns Mauritian Creole (`mfe`) with English and French. This
script extracts `(creole, translation)` pairs from the `mfe-eng` and `mfe-fra` configurations,
normalizes and deduplicates them, and writes the corpus's own train and test splits. The output is
not committed; rerun this script to regenerate it.
"""

import argparse
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

REPO_ID = "jhu-clsp/kreyol-mt"
CONFIGS = ("mfe-eng", "mfe-fra")
CREOLE_LANG = "mfe"


def to_pair(row: dict) -> dict[str, str]:
    entry = row["translation"]
    if entry["src_lang"] == CREOLE_LANG:
        return {
            "creole": entry["src_text"],
            "translation": entry["tgt_text"],
            "lang": entry["tgt_lang"],
        }
    return {
        "creole": entry["tgt_text"],
        "translation": entry["src_text"],
        "lang": entry["src_lang"],
    }


def load_pairs(config: str, split: str) -> Dataset:
    dataset = load_dataset(REPO_ID, config, split=split)
    dataset = dataset.map(to_pair, remove_columns=dataset.column_names)
    return dataset.map(
        lambda row: {"creole": row["creole"].strip(), "translation": row["translation"].strip()}
    )


def is_valid(row: dict[str, str], min_chars: int, max_chars: int) -> bool:
    creole, translation = row["creole"], row["translation"]
    if not creole or not translation:
        return False
    return min_chars <= len(creole) <= max_chars and min_chars <= len(translation) <= max_chars


def deduplicate(dataset: Dataset) -> Dataset:
    frame = dataset.to_pandas().drop_duplicates(subset=["creole", "translation"], ignore_index=True)
    return Dataset.from_pandas(frame)


def build_split(split: str, min_chars: int, max_chars: int) -> Dataset:
    pairs = concatenate_datasets([load_pairs(config, split) for config in CONFIGS])
    pairs = pairs.filter(lambda row: is_valid(row, min_chars, max_chars))
    return deduplicate(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=512)
    args = parser.parse_args()

    train = build_split("train", args.min_chars, args.max_chars)
    test = build_split("test", args.min_chars, args.max_chars)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_json(args.output_dir / "train.jsonl")
    test.to_json(args.output_dir / "test.jsonl")

    print(f"train pairs: {len(train):,}")
    print(f"test pairs:  {len(test):,}")


if __name__ == "__main__":
    main()
