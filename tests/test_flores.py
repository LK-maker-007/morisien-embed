from __future__ import annotations

import json
from pathlib import Path

import pytest

from morisien_embed import flores, xsim


def test_flores_split_pins_the_revision_and_asks_for_the_right_file(monkeypatch, tmp_path: Path) -> None:
    asked: dict[str, str] = {}

    def fake_download(repo: str, filename: str, repo_type: str, revision: str | None) -> str:
        asked.update(repo=repo, filename=filename, repo_type=repo_type, revision=revision)
        path = tmp_path / "split.jsonl"
        rows = [{"id": 2, "text": "Mo pe  ale"}, {"id": 1, "text": "Li pe manze"}]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(flores, "hf_hub_download", fake_download)
    got = flores.flores_split("mfe", "devtest")

    assert asked["repo"] == "openlanguagedata/flores_plus"
    assert asked["filename"] == "devtest/mfe_Latn.jsonl"
    assert asked["repo_type"] == "dataset"
    # an unpinned load would silently follow the branch head and break byte-stability
    assert asked["revision"] == flores.FLORES_REVISION
    # keyed by the sentence id, not by position, and whitespace-normalized
    assert got == {2: "Mo pe ale", 1: "Li pe manze"}


def test_flores_split_rejects_an_unknown_language(monkeypatch) -> None:
    monkeypatch.setattr(flores, "hf_hub_download", lambda *a, **k: pytest.fail("must not download"))
    with pytest.raises(ValueError, match="Unknown FLORES"):
        flores.flores_split("xyz", "devtest")


def test_flores_split_revision_none_tracks_the_branch_head(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, str | None] = {}

    def fake_download(repo: str, filename: str, repo_type: str, revision: str | None) -> str:
        seen["revision"] = revision
        path = tmp_path / "split.jsonl"
        path.write_text(json.dumps({"id": 1, "text": "x"}), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(flores, "hf_hub_download", fake_download)
    flores.flores_split("eng", "dev", revision=None)
    assert seen["revision"] is None


def test_fetch_errtype_downloads_once_and_reads_the_cache_after(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}
    payload = {"a perturbed sentence": {"errtype": "number_replacement", "src": "the original"}}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            calls["n"] += 1
            return json.dumps(payload).encode("utf-8")

    urls: list[str] = []

    def fake_urlopen(url: str, timeout: int):
        urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(xsim.urllib.request, "urlopen", fake_urlopen)
    assert xsim.fetch_errtype(tmp_path) == payload
    # the released artifact has a fixed name, so pin it here rather than reading it back off the module
    assert urls == ["https://dl.fbaipublicfiles.com/nllb/laser/xsimplusplus/eng_Latn_errtype.devtest.json"]
    assert calls["n"] == 1
    # the second call must come from disk, not the network
    assert xsim.fetch_errtype(tmp_path) == payload
    assert calls["n"] == 1
    assert (tmp_path / "eng_Latn_errtype.devtest.json").exists()
