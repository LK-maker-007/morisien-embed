"""Assemble the Kreol Morisien retrieval benchmark from the held-out test pairs.

Each unique Creole sentence becomes a query and each unique translation becomes a corpus passage; the
relevance judgements link a query to the passage(s) that translate it. The result is a standard
``(queries, corpus, qrels)`` retrieval task that ``InformationRetrievalEvaluator`` consumes directly.
The output is regenerated from the test split, so it is not committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    """Read the ``(creole, translation, lang)`` pairs written by ``data/download.py``."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build(
    rows: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]], dict[str, str]]:
    """Turn the pairs into ``queries``, ``corpus``, ``qrels`` and a per-passage language map.

    Queries and passages are keyed by first appearance so identical text collapses to one id. A
    Creole sentence with several distinct translations (e.g. one English and one French) yields one
    query pointing at several relevant passages.
    """
    query_id: dict[str, str] = {}
    corpus_id: dict[str, str] = {}
    qrels: dict[str, set[str]] = {}
    corpus_lang: dict[str, str] = {}
    for row in rows:
        creole, translation, lang = row["creole"], row["translation"], row["lang"]
        if creole not in query_id:
            query_id[creole] = f"q{len(query_id)}"
        if translation not in corpus_id:
            corpus_id[translation] = f"d{len(corpus_id)}"
        cid = corpus_id[translation]
        corpus_lang[cid] = lang
        qrels.setdefault(query_id[creole], set()).add(cid)
    return query_id, corpus_id, qrels, corpus_lang


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/data"))
    args = parser.parse_args()

    rows = load_rows(args.test_file)
    query_id, corpus_id, qrels, corpus_lang = build(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "queries.jsonl",
        [{"_id": qid, "text": text} for text, qid in query_id.items()],
    )
    write_jsonl(
        args.output_dir / "corpus.jsonl",
        [{"_id": cid, "text": text, "lang": corpus_lang[cid]} for text, cid in corpus_id.items()],
    )
    (args.output_dir / "qrels.json").write_text(
        json.dumps({qid: sorted(cids) for qid, cids in qrels.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    judged = sum(len(cids) for cids in qrels.values())
    print(f"queries: {len(query_id)}  corpus: {len(corpus_id)}  judged pairs: {judged}")


if __name__ == "__main__":
    main()
