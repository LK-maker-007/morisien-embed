from __future__ import annotations

from morisien_embed import data


def pair(creole: str, translation: str, lang: str = "eng") -> dict[str, str]:
    return {"creole": creole, "translation": translation, "lang": lang}


def test_normalize_collapses_whitespace() -> None:
    assert data.normalize("  Mo   pe\tale \n lakaz  ") == "Mo pe ale lakaz"


def test_loose_ignores_case_and_punctuation() -> None:
    assert data.loose("Lerla li dir Mari, bien for!") == data.loose("lerla li dir mari bien for")
    assert data.loose("Mo pe ale") != data.loose("Mo pa ale")


def test_merge_drops_exact_duplicates_case_insensitively() -> None:
    pairs = [pair("Mo pe ale", "I am going"), pair("MO PE ALE", "i am going"), pair("Mo pe ale", "I go")]
    kept, dropped = data.merge(pairs, reserved=set())
    assert [p["translation"] for p in kept] == ["I am going", "I go"]
    assert dropped["duplicate"] == 1


def test_merge_drops_reserved_evaluation_sentences() -> None:
    pairs = [pair("Mo pe ale", "I am going"), pair("Li pe manze", "He is eating")]
    kept, dropped = data.merge(pairs, reserved={"mo pe ale"})
    assert [p["creole"] for p in kept] == ["Li pe manze"]
    assert dropped["leak"] == 1


def test_merge_keeps_first_occurrence() -> None:
    pairs = [pair("Mo pe ale", "I am going", "eng"), pair("Mo pe ale", "I am going", "fra")]
    kept, _ = data.merge(pairs, reserved=set())
    assert len(kept) == 1
    assert kept[0]["lang"] == "eng"
