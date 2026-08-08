from __future__ import annotations

import io
import zipfile
from pathlib import Path

from morisien_embed import data


def pair(creole: str, translation: str, lang: str = "eng") -> dict[str, str]:
    return {"creole": creole, "translation": translation, "lang": lang}


def test_normalize_collapses_whitespace() -> None:
    assert data.normalize("  Mo   pe\tale \n lakaz  ") == "Mo pe ale lakaz"


def test_loose_ignores_case_and_punctuation() -> None:
    assert data.loose("Lerla li dir Mari, bien for!") == data.loose("lerla li dir mari bien for")
    assert data.loose("Mo pe ale") != data.loose("Mo pa ale")


def test_loose_ignores_accents_and_invisible_characters() -> None:
    assert data.loose("ast\u00e8r") == data.loose("aster")  # accented e vs plain e
    assert data.loose("ast\u00e9r") == data.loose("aste\u0301r")  # NFC vs NFD composition
    assert data.loose("mo\u200bpe ale") == data.loose("Mo pe ale")  # zero-width space
    assert data.loose("li\u2019n ale") == data.loose("li'n ale")  # curly vs straight apostrophe
    assert data.loose("mo\u00a0pe ale") == data.loose("mo pe ale")  # non-breaking space


def test_merge_drops_exact_duplicates_case_insensitively() -> None:
    pairs = [pair("Mo pe ale", "I am going"), pair("MO PE ALE", "i am going"), pair("Mo pe ale", "I go")]
    kept, dropped = data.merge(pairs, reserved=set())
    assert [p["translation"] for p in kept] == ["I am going", "I go"]
    assert dropped["duplicate"] == 1


def test_merge_drops_reserved_evaluation_sentences_loosely() -> None:
    reserved = {data.loose("Mo pe ale lakaz.")}
    pairs = [
        pair("Mo pe ale lakaz", "punctuation variant leaks"),
        pair("MO PE ALE LAKAZ!!", "case and punctuation variant leaks"),
        pair("Mo pe alé lakaz", "accent variant leaks"),
        pair("Li pe manze", "clean pair survives"),
    ]
    kept, dropped = data.merge(pairs, reserved=reserved)
    assert [p["creole"] for p in kept] == ["Li pe manze"]
    assert dropped["leak"] == 3


def test_merge_keeps_first_occurrence() -> None:
    pairs = [pair("Mo pe ale", "I am going", "eng"), pair("Mo pe ale", "I am going", "fra")]
    kept, _ = data.merge(pairs, reserved=set())
    assert len(kept) == 1
    assert kept[0]["lang"] == "eng"


def test_kreyol_mt_orients_pairs_creole_first_and_skips_empty_rows(monkeypatch) -> None:
    def row(src_lang: str, src_text: str, tgt_lang: str, tgt_text: str) -> dict:
        entry = {"src_lang": src_lang, "src_text": src_text, "tgt_lang": tgt_lang, "tgt_text": tgt_text}
        return {"translation": entry}

    rows = {
        "mfe-eng": [
            row("mfe", "Mo pe ale", "eng", "I am going"),  # creole on the source side
            row("eng", "He is eating", "mfe", "Li pe manze"),  # creole on the target side
            row("eng", "  ", "mfe", "Zot pe zwe"),  # blank translation is skipped
        ],
        "mfe-fra": [row("fra", "Je rentre chez moi", "mfe", "Mo pe ale  lakaz")],
    }
    monkeypatch.setattr(data, "load_dataset", lambda repo, config, split: rows[config])
    assert data.kreyol_mt("train") == [
        {"creole": "Mo pe ale", "translation": "I am going", "lang": "eng"},
        {"creole": "Li pe manze", "translation": "He is eating", "lang": "eng"},
        {"creole": "Mo pe ale lakaz", "translation": "Je rentre chez moi", "lang": "fra"},
    ]


def test_morisienmt_reads_zip_archives_and_skips_empty_rows(monkeypatch, tmp_path: Path) -> None:
    rows = {
        "en-cr": [
            {"input": "I am going home", "target": "Mo pe ale lakaz"},
            {"input": "  ", "target": "Mo pe ale"},  # blank translation is skipped
        ],
        "fr-cr": [{"input": "Je rentre chez moi", "target": "Mo pe ale  lakaz"}],
    }

    def fake_download(repo: str, filename: str, repo_type: str) -> str:
        pair_name = Path(filename).stem
        archive = tmp_path / f"{pair_name}.zip"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            lines = "\n".join(f'{{"input": "{r["input"]}", "target": "{r["target"]}"}}' for r in rows[pair_name])
            bundle.writestr(f"{pair_name}_test.jsonl", lines)
        archive.write_bytes(buffer.getvalue())
        return str(archive)

    monkeypatch.setattr(data, "hf_hub_download", fake_download)
    pairs = data.morisienmt("test")
    assert pairs == [
        {"creole": "Mo pe ale lakaz", "translation": "I am going home", "lang": "eng"},
        {"creole": "Mo pe ale lakaz", "translation": "Je rentre chez moi", "lang": "fra"},
    ]
