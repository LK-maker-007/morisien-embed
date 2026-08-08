from __future__ import annotations

from pathlib import Path

from morisien_embed import benchmark

PAIRS = [
    {"creole": "Mo pe ale lakaz", "translation": "I am going home", "lang": "eng"},
    {"creole": "Mo pe ale lakaz", "translation": "Je rentre chez moi", "lang": "fra"},
    {"creole": "Li pe manze", "translation": "He is eating", "lang": "eng"},
    {"creole": "Zot pe zwe", "translation": "He is eating", "lang": "eng"},
]


def test_build_deduplicates_queries_and_corpus() -> None:
    queries, corpus, qrels = benchmark.build(PAIRS)
    assert len(queries) == 3  # three unique creole sentences
    assert len(corpus) == 3  # "He is eating" shared by two queries collapses to one passage
    assert len(qrels) == 3


def test_build_links_one_creole_to_all_its_translations() -> None:
    queries, corpus, qrels = benchmark.build(PAIRS)
    qid = next(q for q, text in queries.items() if text == "Mo pe ale lakaz")
    relevant_texts = {corpus[cid] for cid in qrels[qid]}
    assert relevant_texts == {"I am going home", "Je rentre chez moi"}


def test_build_target_lang_filters_pairs() -> None:
    queries, corpus, qrels = benchmark.build(PAIRS, target_lang="fra")
    assert list(queries.values()) == ["Mo pe ale lakaz"]
    assert list(corpus.values()) == ["Je rentre chez moi"]


def test_write_then_load_roundtrips(tmp_path: Path) -> None:
    built = benchmark.build(PAIRS)
    benchmark.write(tmp_path, built)
    assert benchmark.load(tmp_path) == built
