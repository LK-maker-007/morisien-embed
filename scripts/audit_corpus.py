"""Measure what the training corpus and the evaluation split actually contain.

The paper makes claims about corpus composition, leakage filtering and the MTEB task that were not
backed by a recorded measurement. This script produces them all in one pass and writes the result as
JSON, so each number in the paper points at a file rather than at a recollection.

Every figure here is derived from the pinned loaders in :mod:`morisien_embed.data`, so re-running it
reproduces the same values.
"""

from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from morisien_embed import data, flores


def word_count(text: str) -> int:
    return len(text.split())


def tokenize(text: str, *, strip_punctuation: bool) -> list[str]:
    """Split on whitespace, lower-cased.

    With ``strip_punctuation`` the accents are decomposed and non-alphanumeric characters dropped, so
    a word carrying a trailing comma is not counted as a different word from the bare form. The two
    settings answer different questions and the coverage figures below report both.
    """
    tokens = [token.casefold() for token in text.split()]
    if not strip_punctuation:
        return tokens
    stripped = ("".join(c for c in unicodedata.normalize("NFKD", token) if c.isalnum()) for token in tokens)
    return [token for token in stripped if token]


def coverage(train: list[str], test: list[str], *, strip_punctuation: bool) -> dict[str, object]:
    """How much of the evaluation split's wording already appears in training."""
    vocabulary = {token for text in train for token in tokenize(text, strip_punctuation=strip_punctuation)}
    tokens = [token for text in test for token in tokenize(text, strip_punctuation=strip_punctuation)]
    seen = sum(1 for token in tokens if token in vocabulary)
    no_unseen = sum(
        1 for text in test if all(token in vocabulary for token in tokenize(text, strip_punctuation=strip_punctuation))
    )
    return {
        "training_vocabulary": len(vocabulary),
        "test_tokens": len(tokens),
        "tokens_seen_in_training": seen,
        "token_coverage_pct": round(100 * seen / len(tokens), 2),
        "sentences_with_no_unseen_token": no_unseen,
        "sentences_with_no_unseen_token_pct": round(100 * no_unseen / len(test), 2),
    }


def composition(pairs: list[dict[str, str]]) -> dict[str, object]:
    """Length profile of the Creole side, which is what the model is trained to encode."""
    lengths = [word_count(pair["creole"]) for pair in pairs]
    return {
        "pairs": len(pairs),
        "single_word": sum(1 for n in lengths if n == 1),
        "single_word_pct": round(100 * sum(1 for n in lengths if n == 1) / len(lengths), 2),
        "six_words_or_more": sum(1 for n in lengths if n >= 6),
        "median_words": statistics.median(lengths),
        "mean_words": round(statistics.fmean(lengths), 2),
        "by_lang": dict(Counter(pair["lang"] for pair in pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/phase-d-corpus-audit.json"))
    args = parser.parse_args()

    # 1. Corpus accounting. How many rows each filter actually removes.
    raw = data.morisienmt("train") + data.kreyol_mt("train")
    reserved = data.reserved_creole()
    kept, dropped = data.merge(raw, reserved)

    # Re-merging with an empty reserved set isolates what the leak filter itself removes, which the
    # paper describes as a safeguard. If the two row counts agree, it removed nothing.
    unfiltered, dropped_unfiltered = data.merge(raw, set())

    filters = {
        "raw_rows": len(raw),
        "reserved_creole_keys": len(reserved),
        "dropped_leak": dropped["leak"],
        "dropped_duplicate": dropped["duplicate"],
        "kept": len(kept),
        "kept_without_leak_filter": len(unfiltered),
        "rows_the_leak_filter_removes": len(unfiltered) - len(kept),
        "duplicates_without_leak_filter": dropped_unfiltered["duplicate"],
    }

    # 2. Composition of the corpus the released model trains on.
    corpus = composition(kept)

    # 3. The evaluation split, and what the four MTEB subsets share.
    test_pairs = data.morisienmt("test")
    by_lang: dict[str, list[dict[str, str]]] = {}
    for pair in test_pairs:
        by_lang.setdefault(pair["lang"], []).append(pair)
    creole_by_lang = {lang: [p["creole"] for p in rows] for lang, rows in by_lang.items()}
    sets = {lang: set(map(data.loose, texts)) for lang, texts in creole_by_lang.items()}
    langs = sorted(sets)
    shared = set.intersection(*(sets[lang] for lang in langs)) if len(langs) > 1 else set(sets[langs[0]])

    test_lengths = [word_count(text) for text in creole_by_lang[langs[0]]]
    train_lengths = [word_count(pair["creole"]) for pair in kept]

    # 4. Vocabulary overlap. How much of the test split's wording the model has already seen.
    train_creole = [pair["creole"] for pair in kept]
    test_creole = creole_by_lang[langs[0]]

    evaluation = {
        "morisienmt_test_rows": len(test_pairs),
        "rows_per_direction": {lang: len(rows) for lang, rows in by_lang.items()},
        "distinct_creole_sentences_per_direction": {lang: len(s) for lang, s in sets.items()},
        "creole_sentences_shared_by_all_directions": len(shared),
        "test_median_words": statistics.median(test_lengths),
        "test_min_words": min(test_lengths),
        "test_max_words": max(test_lengths),
        "train_median_words": statistics.median(train_lengths),
        "train_rows_10_words_or_more": sum(1 for n in train_lengths if n >= 10),
        "train_rows_10_words_or_more_pct": round(
            100 * sum(1 for n in train_lengths if n >= 10) / len(train_lengths), 2
        ),
        "vocabulary_overlap": {
            "punctuation_stripped": coverage(train_creole, test_creole, strip_punctuation=True),
            "whitespace_only": coverage(train_creole, test_creole, strip_punctuation=False),
            "method": (
                "Creole side only, whitespace tokens, lower-cased. The punctuation_stripped variant "
                "additionally decomposes accents and keeps alphanumeric characters, so a word with "
                "attached punctuation is not counted as unseen. It is the figure the paper quotes."
            ),
        },
    }

    # 5. FLORES+ split sizes. The dev split is the holdout no code here has ever selected on.
    flores_splits = {split: len(flores.flores_split("mfe", split)) for split in ("dev", "devtest")}

    report = {
        "run": "Corpus and evaluation-split audit for the paper revision",
        "produced_by": "scripts/audit_corpus.py",
        "sources": {
            "morisienmt": {"repo": data.MORISIEN_REPO, "revision": data.MORISIEN_REVISION},
            "kreyol_mt": {"repo": data.KREYOL_REPO, "revision": data.KREYOL_REVISION},
        },
        "filters": filters,
        "corpus": corpus,
        "evaluation_split": evaluation,
        "flores_plus": {
            "repo": flores.FLORES_REPO,
            "revision": flores.FLORES_REVISION,
            "mfe_sentences": flores_splits,
            "note": "Only devtest has been used. dev is an untouched holdout.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
