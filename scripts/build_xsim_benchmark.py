"""Build a hard Creole retrieval benchmark from FLORES+ and the released xSIM++ distractors.

The plain FLORES+ benchmark saturates: morisien-embed and LaBSE both sit at the ceiling over its own
1,012-passage corpus, and a char-n-gram TF-IDF baseline with no model reaches 0.82 accuracy@1, so the
pool cannot separate models. xSIM++ (Chen et al., ACL 2023) is the published fix: it perturbs the
English side of FLORES with rule-based edits -- entity replacement, number replacement and causality
alternation -- so each distractor is the gold passage with one semantically critical token changed.
Nothing but meaning separates a distractor from its gold, which is what corpus-mined distractors
cannot offer.

The released augmentation covers the English side only and is shared across source languages, so it
applies to Creole->English unchanged. Its 996 originals were verified to match the FLORES+ devtest
English text exactly, so gold passages align by string despite xSIM++ being built on FLORES200.

Access to FLORES+ is gated: accept the dataset's terms on the Hugging Face Hub and set ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from morisien_embed import benchmark, data
from morisien_embed.data import loose
from morisien_embed.flores import flores_split
from morisien_embed.xsim import fetch_errtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "devtest"), default="devtest")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark/.xsim"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/data/xsim-eng"))
    args = parser.parse_args()

    creole = flores_split("mfe", args.split)
    english = flores_split("eng", args.split)
    errtype = fetch_errtype(args.cache_dir)

    # An augmented sentence is a distractor for the original it was derived from; group them so that
    # only FLORES sentences xSIM++ actually augmented become queries.
    augmented: dict[str, list[str]] = {}
    for sentence, meta in errtype.items():
        augmented.setdefault(data.normalize(meta["src"]), []).append(data.normalize(sentence))

    pairs = [
        {"creole": creole[sid], "translation": english[sid], "lang": "eng"}
        for sid in sorted(creole.keys() & english.keys())
        if english[sid] in augmented
    ]
    print(f"flores {args.split}: {len(pairs)} pairs with xSIM++ augmentations")

    train_loose = {
        loose(json.loads(line)["creole"]) for line in args.train_file.read_text(encoding="utf-8").splitlines()
    }
    clean = [pair for pair in pairs if loose(pair["creole"]) not in train_loose]
    print(f"leaked (dropped): {len(pairs) - len(clean)}")

    queries, corpus, qrels = benchmark.build(clean, target_lang="eng")
    gold = {pair["translation"] for pair in clean}
    seen = set(corpus.values())
    # `benchmark.build` mints d0..d{n-1}, but deriving the next id from the highest in use does not
    # depend on that, so a change there cannot silently produce colliding ids here.
    next_id = max((int(cid[1:]) for cid in corpus), default=-1) + 1
    collisions = 0
    for passages in augmented.values():
        for passage in passages:
            if passage in gold:
                # A perturbation that reproduces another sentence's gold would make that query
                # unanswerable. Counted rather than dropped silently, so a future FLORES+ or xSIM++
                # revision that starts producing them is visible instead of quietly shrinking the pool.
                collisions += 1
                continue
            if passage in seen:
                continue
            seen.add(passage)
            corpus[f"d{next_id}"] = passage
            next_id += 1
    print(f"perturbations reproducing a gold passage (excluded): {collisions}")

    benchmark.write(args.output_dir, (queries, corpus, qrels))
    print(f"[xsim eng] queries: {len(queries)}  corpus: {len(corpus)}  ->  {args.output_dir}")


if __name__ == "__main__":
    main()
