"""End-to-end contract tests for the stages the pipeline chains together.

Every other test file covers one function. These cover the seams: the corpus loader writes a file
that the training loader reads, and the benchmark builder writes a directory that the benchmark
loader reads and the scorers consume. A change to either side of one of those seams passes its own
unit tests and breaks the pipeline, which is the failure these exist to catch.

They deliberately do not run an encoder. The suite is offline by design, and a model download would
make the seam tests depend on the Hub. What is tested here is the data contract; the scoring itself
is covered by `test_xsim_margin.py` against the LASER reference.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from morisien_embed import benchmark, data

# Loaded by path, as tests/test_train.py does. ``scripts`` is not an installed package, so an
# ``import scripts.train`` resolves only when pytest happens to put the repository root on the path
# and would skip silently in CI rather than fail.
_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_for_pipeline", _TRAIN_PATH)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

PAIRS = [
    {"creole": "Mo pe ale lakaz", "translation": "I am going home", "lang": "eng"},
    {"creole": "Li pe manze enn dipin", "translation": "He is eating a bread", "lang": "eng"},
    {"creole": "Mo pe ale lakaz", "translation": "Je rentre chez moi", "lang": "fra"},
]


def test_merge_output_feeds_the_training_loader(tmp_path: Path) -> None:
    """`build_training.py` writes what `train.load_training_pairs` reads, field names included."""
    kept, _ = data.merge(PAIRS, reserved=set())
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in kept), encoding="utf-8")

    dataset = train.load_training_pairs(path, limit=None)

    assert dataset.column_names == ["anchor", "positive"]
    assert dataset["anchor"] == [p["creole"] for p in kept]
    assert dataset["positive"] == [p["translation"] for p in kept]


def test_benchmark_survives_a_write_and_load_round_trip(tmp_path: Path) -> None:
    """Ids are assigned at build time and resolved at load time, so the round trip must be lossless."""
    built = benchmark.build(PAIRS, target_lang="eng")
    benchmark.write(tmp_path / "eng", built)

    loaded = benchmark.load(tmp_path / "eng")

    assert loaded == built


def test_a_freshly_built_benchmark_passes_the_loader_validation(tmp_path: Path) -> None:
    """The builder must never emit qrels the loader rejects, or no benchmark could be built at all."""
    for lang in ("eng", "fra"):
        benchmark.write(tmp_path / lang, benchmark.build(PAIRS, target_lang=lang))
        queries, corpus, qrels = benchmark.load(tmp_path / lang)
        assert set(qrels) <= set(queries)
        assert {cid for cids in qrels.values() for cid in cids} <= set(corpus)
        assert all(cids for cids in qrels.values())


def test_one_creole_sentence_with_two_translations_keeps_both_judgements(tmp_path: Path) -> None:
    """ "Mo pe ale lakaz" has an English and a French translation. Building without restricting the
    language must give that query both passages rather than silently dropping one."""
    benchmark.write(tmp_path / "both", benchmark.build(PAIRS, target_lang=None))
    queries, corpus, qrels = benchmark.load(tmp_path / "both")

    shared = next(qid for qid, text in queries.items() if text == "Mo pe ale lakaz")
    assert {corpus[cid] for cid in qrels[shared]} == {"I am going home", "Je rentre chez moi"}


def test_reserved_creole_removes_the_evaluation_side_from_training(tmp_path: Path) -> None:
    """The leak guard and the benchmark must agree on which sentences are reserved."""
    reserved = {data.loose(PAIRS[0]["creole"])}
    kept, dropped = data.merge(PAIRS, reserved=reserved)

    assert dropped["leak"] == 2, "both language rows of the reserved sentence go"
    assert all(data.loose(p["creole"]) not in reserved for p in kept)
