"""Build the Kreol Morisien retrieval benchmark from the held-out MorisienMT test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from morisien_embed import benchmark, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("eng", "fra"), default="eng")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or Path("benchmark/data") / args.target
    bench = benchmark.build(data.morisienmt("test"), target_lang=args.target)
    benchmark.write(out_dir, bench)

    queries, corpus, _ = bench
    print(f"[{args.target}] queries: {len(queries)}  corpus: {len(corpus)}  ->  {out_dir}")


if __name__ == "__main__":
    main()
