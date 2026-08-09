"""Build the Kreol Morisien retrieval benchmark from the held-out MorisienMT test split.

By default queries are Creole and passages are translations; ``--reverse`` swaps the direction
(translation queries retrieving Creole passages) over the same pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from morisien_embed import benchmark, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("eng", "fra"), default="eng")
    parser.add_argument("--reverse", action="store_true", help="translation queries -> Creole passages")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    name = f"{args.target}-creole" if args.reverse else args.target
    fields = {"query_field": "translation", "passage_field": "creole"} if args.reverse else {}
    out_dir = args.output_dir or Path("benchmark/data") / name
    bench = benchmark.build(data.morisienmt("test"), target_lang=args.target, **fields)
    benchmark.write(out_dir, bench)

    queries, corpus, _ = bench
    print(f"[{name}] queries: {len(queries)}  corpus: {len(corpus)}  ->  {out_dir}")


if __name__ == "__main__":
    main()
