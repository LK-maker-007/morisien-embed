"""Turn Creole↔translation pairs into a retrieval benchmark and evaluate models on it.

A benchmark is the standard ``(queries, corpus, qrels)`` triple that ``InformationRetrievalEvaluator``
consumes: each unique Creole sentence is a query, each unique translation a passage, and the relevance
judgements link a query to the passage(s) that translate it.
"""

from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

from morisien_embed.data import normalize

Benchmark = tuple[dict[str, str], dict[str, str], dict[str, set[str]]]


def build(
    pairs: list[dict[str, str]],
    target_lang: str | None = None,
    *,
    query_field: str = "creole",
    passage_field: str = "translation",
) -> Benchmark:
    """Build queries, corpus and qrels from pairs, optionally restricting to one translation language.

    ``query_field``/``passage_field`` choose the retrieval direction: the defaults score
    Creole→translation; swapping them scores translation→Creole over the same pairs. Texts are
    whitespace-normalized so the written JSONL never contains line-breaking characters.
    """
    query_id: dict[str, str] = {}
    corpus_id: dict[str, str] = {}
    qrels: dict[str, set[str]] = {}
    for pair in pairs:
        if target_lang and pair["lang"] != target_lang:
            continue
        query, passage = normalize(pair[query_field]), normalize(pair[passage_field])
        query_id.setdefault(query, f"q{len(query_id)}")
        corpus_id.setdefault(passage, f"d{len(corpus_id)}")
        qrels.setdefault(query_id[query], set()).add(corpus_id[passage])
    queries = {qid: text for text, qid in query_id.items()}
    corpus = {cid: text for text, cid in corpus_id.items()}
    return queries, corpus, qrels


def write(out_dir: Path, benchmark: Benchmark) -> None:
    queries, corpus, qrels = benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "queries.jsonl", ({"_id": q, "text": t} for q, t in queries.items()))
    _write_jsonl(out_dir / "corpus.jsonl", ({"_id": c, "text": t} for c, t in corpus.items()))
    (out_dir / "qrels.json").write_text(
        json.dumps({q: sorted(c) for q, c in qrels.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(data_dir: Path) -> Benchmark:
    def read(name: str) -> dict[str, str]:
        rows = [json.loads(line) for line in (data_dir / name).read_text(encoding="utf-8").split("\n") if line]
        records = {row["_id"]: row["text"] for row in rows}
        if len(records) != len(rows):
            raise ValueError(f"{data_dir / name} contains duplicate _id entries")
        return records

    queries = read("queries.jsonl")
    corpus = read("corpus.jsonl")
    qrels_raw = json.loads((data_dir / "qrels.json").read_text(encoding="utf-8"))
    return queries, corpus, {qid: set(cids) for qid, cids in qrels_raw.items()}


def evaluate(
    model: SentenceTransformer,
    benchmark: Benchmark,
    *,
    name: str = "kreol-morisien-retrieval",
    query_prompt: str | None = None,
    corpus_prompt: str | None = None,
    batch_size: int = 64,
) -> dict[str, float]:
    queries, corpus, qrels = benchmark
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        query_prompt=query_prompt,
        corpus_prompt=corpus_prompt,
        batch_size=batch_size,
        name=name,
    )
    return evaluator(model)


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
