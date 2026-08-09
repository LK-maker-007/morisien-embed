from __future__ import annotations

from pathlib import Path

import pytest

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


def test_build_reversed_direction_swaps_queries_and_corpus() -> None:
    queries, corpus, qrels = benchmark.build(PAIRS, query_field="translation", passage_field="creole")
    assert len(queries) == 3  # "He is eating" collapses to one query
    assert len(corpus) == 3
    qid = next(q for q, text in queries.items() if text == "He is eating")
    relevant_texts = {corpus[cid] for cid in qrels[qid]}
    assert relevant_texts == {"Li pe manze", "Zot pe zwe"}  # one query, both creole translations relevant


def test_build_normalizes_line_breaking_whitespace() -> None:
    pairs = [{"creole": "Mo pe ale", "translation": "I am\ngoing", "lang": "eng"}]
    queries, corpus, _ = benchmark.build(pairs)
    assert list(queries.values()) == ["Mo pe ale"]
    assert list(corpus.values()) == ["I am going"]


def test_build_empty_pairs() -> None:
    assert benchmark.build([]) == ({}, {}, {})


def test_write_then_load_roundtrips(tmp_path: Path) -> None:
    built = benchmark.build(PAIRS)
    benchmark.write(tmp_path, built)
    assert benchmark.load(tmp_path) == built


def test_write_then_load_roundtrips_non_ascii(tmp_path: Path) -> None:
    pairs = [{"creole": "Zanfan-la pe manz so dipin astèr", "translation": "L'enfant mange, déjà", "lang": "fra"}]
    built = benchmark.build(pairs)
    benchmark.write(tmp_path, built)
    loaded = benchmark.load(tmp_path)
    assert loaded == built
    assert "astèr" in next(iter(loaded[0].values()))


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    benchmark.write(tmp_path, benchmark.build(PAIRS))
    queries_file = tmp_path / "queries.jsonl"
    first_line = queries_file.read_text(encoding="utf-8").split("\n")[0]
    queries_file.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate _id"):
        benchmark.load(tmp_path)
