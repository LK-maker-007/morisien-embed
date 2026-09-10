"""Build a Creole retrieval benchmark from google/smol, a second held-out domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from morisien_embed import benchmark, data
from morisien_embed.data import loose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--distractors-from",
        type=Path,
        default=None,
        help="corpus.jsonl whose passages are added to widen the pool",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/data/smol-eng"))
    args = parser.parse_args()

    pairs = data.smol()
    print(f"smol pairs: {len(pairs)}")

    seen_train = {
        loose(json.loads(line)["creole"]) for line in args.train_file.read_text(encoding="utf-8").splitlines()
    }
    kept = [p for p in pairs if loose(p["creole"]) not in seen_train]
    print(f"dropped {len(pairs) - len(kept)} pairs whose Creole side is in {args.train_file}")

    queries, corpus, qrels = benchmark.build(kept)
    print(f"queries {len(queries)}  corpus {len(corpus)}")

    if args.distractors_from:
        present = set(corpus.values())
        # Deriving the next id from the highest in use does not depend on how `build` mints ids, so a
        # change there cannot silently produce a collision here.
        next_id = max((int(cid[1:]) for cid in corpus), default=-1) + 1
        added = 0
        for line in args.distractors_from.read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            if text in present:
                continue
            corpus[f"d{next_id}"] = text
            present.add(text)
            next_id += 1
            added += 1
        print(f"added {added} distractors from {args.distractors_from}; corpus now {len(corpus)}")

    benchmark.write(args.output_dir, (queries, corpus, qrels))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
