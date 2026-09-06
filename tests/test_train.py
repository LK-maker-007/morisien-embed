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


def _write_pairs(tmp_path: Path, creoles: list[str]) -> Path:
    path = tmp_path / "train.jsonl"
    rows = (json.dumps({"creole": c, "translation": f"t{i}"}) for i, c in enumerate(creoles))
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_min_words_keeps_a_row_of_exactly_that_length(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path, ["one", "two words", "three words here", "a b c d e f"])
    kept = train.load_training_pairs(path, limit=None, min_words=3)["anchor"]
    # a row of exactly min_words stays, which is the boundary a > rather than >= would get wrong
    assert kept == ["three words here", "a b c d e f"]


def test_min_words_one_is_a_no_op(tmp_path: Path) -> None:
    creoles = ["one", "two words", "three words here"]
    path = _write_pairs(tmp_path, creoles)
    assert train.load_training_pairs(path, limit=None, min_words=1)["anchor"] == creoles


def test_sample_is_reproducible_and_seed_dependent(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path, [f"row {i}" for i in range(40)])
    first = train.load_training_pairs(path, limit=None, sample=10, sample_seed=0)["anchor"]
    again = train.load_training_pairs(path, limit=None, sample=10, sample_seed=0)["anchor"]
    other = train.load_training_pairs(path, limit=None, sample=10, sample_seed=1)["anchor"]
    assert len(first) == 10
    assert first == again
    assert first != other


def test_sample_draws_from_the_filtered_rows_not_the_whole_file(tmp_path: Path) -> None:
    # 30 single-word rows then 8 long ones. The filter must run first, leaving 8, and the sample
    # must then cut those to 5. If the sample were skipped the result would be 8, and if it drew
    # from the unfiltered file it would contain single-word rows.
    path = _write_pairs(tmp_path, ["x"] * 30 + [f"a long row number {i}" for i in range(8)])
    got = train.load_training_pairs(path, limit=None, min_words=3, sample=5)["anchor"]
    assert len(got) == 5
    assert all(len(a.split()) >= 3 for a in got)


def test_limit_is_a_head_slice_of_the_sample_not_of_the_file(tmp_path: Path) -> None:
    """All three filters compose in one order: min_words, then sample, then limit.

    ``limit`` is a head slice, so applying it before the sample would return the head of the file.
    The file here is grouped like the real one, short rows first, so a head slice of the file would
    be all single-word rows and a head slice of the sample cannot be.
    """
    path = _write_pairs(tmp_path, ["x"] * 30 + [f"a long row number {i}" for i in range(20)])

    got = train.load_training_pairs(path, limit=4, min_words=3, sample=10, sample_seed=0)["anchor"]

    assert len(got) == 4, "limit must cut the sample down to its own size"
    assert all(len(a.split()) >= 3 for a in got), "min_words must run before both"
    sampled = train.load_training_pairs(path, limit=None, min_words=3, sample=10, sample_seed=0)["anchor"]
    assert got == sampled[:4], "limit takes the head of the sample, in the sample's order"


def test_limit_alone_takes_the_head_of_the_file(tmp_path: Path) -> None:
    """Without a sample, limit is the head of the file as written, which is what --limit means."""
    creoles = [f"row {i}" for i in range(10)]
    path = _write_pairs(tmp_path, creoles)

    assert train.load_training_pairs(path, limit=3)["anchor"] == creoles[:3]


def test_sample_larger_than_the_pool_raises(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path, [f"row {i}" for i in range(5)])
    with pytest.raises(ValueError, match="exceeds"):
        train.load_training_pairs(path, limit=None, sample=6)


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


def _tiny_lora_model():
    """A stand-in for a SentenceTransformer with a LoRA adapter injected into module 0.

    Building this by hand rather than loading a checkpoint keeps the test offline and fast, while
    still exercising real peft layers.
    """
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Linear(8, 8, bias=False)
            self.value = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.value(self.query(x))

    class Inner(torch.nn.Module):
        """Nested the way a real encoder is, so a non-recursive strip cannot reach the adapters."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.ModuleList([Attention()])

        def forward(self, x):
            return self.encoder[0](x)

    inner = Inner()
    peft.inject_adapter_in_model(
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["query", "value"], bias="none"), inner
    )
    inner._hf_peft_config_loaded = True

    # lora_B starts at zero, so the adapter is a no-op until it is moved off it
    with torch.no_grad():
        for name, param in inner.named_parameters():
            if "lora_B" in name:
                param.add_(0.1)

    class Module0:
        def __init__(self, auto_model):
            self.auto_model = auto_model

    class Stub:
        def __init__(self, auto_model):
            self._m = [Module0(auto_model)]

        def __getitem__(self, i):
            return self._m[i]

    return torch, peft, inner, Stub(inner)


def test_merge_lora_keeps_the_adapter_effect_in_the_base_weights() -> None:
    torch, _, inner, model = _tiny_lora_model()
    x = torch.ones(1, 8)
    with torch.no_grad():
        before = inner(x).clone()

    train.merge_lora(model)

    with torch.no_grad():
        after = model[0].auto_model(x)
    # If merge() were skipped, stripping the adapter would throw its contribution away and the
    # output would fall back to the untouched base weights.
    assert torch.allclose(before, after, atol=1e-6)


def test_merge_lora_leaves_no_peft_layers_or_flags_behind() -> None:
    _, peft, inner, model = _tiny_lora_model()
    from peft.tuners.lora import LoraLayer

    assert any(isinstance(m, LoraLayer) for m in inner.modules())

    train.merge_lora(model)

    after = model[0].auto_model
    # save_pretrained takes the adapter-only path while either of these survives
    assert not any(isinstance(m, LoraLayer) for m in after.modules())
    assert after._hf_peft_config_loaded is False
    assert not hasattr(after, "peft_config")
