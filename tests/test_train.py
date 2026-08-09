from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_TRAIN_PATH = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train", _TRAIN_PATH)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def test_matryoshka_dims_are_the_full_dim_then_smaller_powers() -> None:
    assert train.matryoshka_dims(768) == [768, 512, 256, 128, 64]
    assert train.matryoshka_dims(384) == [384, 256, 128, 64]
    assert train.matryoshka_dims(64) == [64]


def test_load_training_pairs_maps_creole_to_anchor_and_translation_to_positive(tmp_path: Path) -> None:
    rows = [{"creole": "a", "translation": "x", "lang": "eng"}, {"creole": "b", "translation": "y", "lang": "fra"}]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = train.load_training_pairs(path, limit=None)
    assert dataset.column_names == ["anchor", "positive"]
    assert dataset["anchor"] == ["a", "b"]
    assert dataset["positive"] == ["x", "y"]


def test_load_training_pairs_limit_zero_is_empty_not_unlimited(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = (json.dumps({"creole": str(i), "translation": str(i)}) for i in range(10))
    path.write_text("\n".join(rows), encoding="utf-8")
    assert len(train.load_training_pairs(path, limit=0)) == 0
    assert len(train.load_training_pairs(path, limit=3)) == 3
    assert len(train.load_training_pairs(path, limit=None)) == 10


def _stub_evaluate(returned: dict[str, float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train.data, "morisienmt", lambda split: [])
    monkeypatch.setattr(train.benchmark, "build", lambda pairs, target_lang=None: ({}, {}, {}))
    monkeypatch.setattr(train.benchmark, "evaluate", lambda model, bench: returned)


def test_report_test_prints_the_expected_metrics(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _stub_evaluate({"eng_cosine_accuracy@1": 0.9, "eng_cosine_ndcg@10": 0.95}, monkeypatch)
    train.report_test(object())
    out = capsys.readouterr().out
    assert "accuracy@1=0.9000" in out
    assert "ndcg@10=0.9500" in out


def test_report_test_raises_when_metric_keys_are_renamed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_evaluate({"unexpected_metric": 1.0}, monkeypatch)
    with pytest.raises(RuntimeError, match="expected cosine"):
        train.report_test(object())
