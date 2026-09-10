"""Score a retrieval benchmark with a character n-gram TF-IDF baseline and no neural model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from morisien_embed import benchmark


def lexical_accuracy(queries: dict[str, str], corpus: dict[str, str], qrels: dict[str, list[str]]) -> float:
    qids, cids = list(queries), list(corpus)
    passage_texts = [corpus[c] for c in cids]
    query_texts = [queries[q] for q in qids]
    # Fitting on queries as well as passages, and damping term frequency, both raise the baseline.
    # The point is to make it as strong as a lexical method can be, so the kill criterion stays
    # conservative: 0.2831 without these, 0.3273 with them, on the same pool.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit(passage_texts + query_texts)
    passages = vec.transform(passage_texts)
    asked = vec.transform(query_texts)
    index = {c: i for i, c in enumerate(cids)}
    hits = 0
    # One block at a time: the full similarity matrix is queries by passages and does not fit for a
    # pool of tens of thousands.
    for start in range(0, len(qids), 256):
        block = (asked[start : start + 256] @ passages.T).toarray()
        for row, top in enumerate(block.argmax(axis=1)):
            gold = {index[g] for g in qrels[qids[start + row]]}
            hits += int(top in gold)
    return hits / len(qids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="+", help="benchmark directories to score")
    parser.add_argument("--threshold", type=float, default=0.50, help="reject a benchmark above this")
    parser.add_argument("--output", type=Path, default=None, help="write the scores here as JSON")
    args = parser.parse_args()

    scores = {}
    for data_dir in args.data_dir:
        queries, corpus, qrels = benchmark.load(data_dir)
        acc = lexical_accuracy(queries, corpus, qrels)
        verdict = "REJECT" if acc > args.threshold else "ok"
        scores[str(data_dir)] = acc
        print(f"{data_dir}  queries {len(queries):6d}  corpus {len(corpus):6d}  acc@1 {acc:.4f}  {verdict}")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "method": "char_wb TF-IDF 3-5grams, no neural model",
                    "kill_criterion": f"reject the benchmark if TF-IDF acc@1 > {args.threshold}",
                    "acc@1": scores,
                    "passed": all(v <= args.threshold for v in scores.values()),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
