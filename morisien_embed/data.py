from __future__ import annotations

import io
import json
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

KREYOL_REPO = "jhu-clsp/kreyol-mt"
KREYOL_REVISION = "4be818dfd3ef6f166a9d5d3553e2b46539507ea3"  # 2024-10-24
KREYOL_CONFIGS = {"mfe-eng": "eng", "mfe-fra": "fra"}
CREOLE_LANG = "mfe"

MORISIEN_REPO = "prajdabre/KreolMorisienMT"  # canonical id; "prajdabre/MorisienMT" is a redirect
MORISIEN_REVISION = "66c76eaf5e33b39a41c3d4c757eee3cf23b52ce5"  # 2022-06-02

SMOL_REPO = "google/smol"
SMOL_REVISION = "fdaff3a1a019f89fa30a562a85d7c1d3e9150444"  # 2026-09-03
SMOL_SUBSETS = ("smolsent", "smoldoc")
MORISIEN_PAIRS = {"en-cr": "eng", "fr-cr": "fra"}

Pair = dict[str, str]


def normalize(text: str) -> str:
    return " ".join(text.split())


def loose(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if char.isalnum())


def kreyol_mt(split: str) -> list[Pair]:
    pairs: list[Pair] = []
    for config, lang in KREYOL_CONFIGS.items():
        for row in load_dataset(KREYOL_REPO, config, split=split, revision=KREYOL_REVISION):
            entry = row["translation"]
            if entry["src_lang"] == CREOLE_LANG:
                creole, translation = entry["src_text"], entry["tgt_text"]
            else:
                creole, translation = entry["tgt_text"], entry["src_text"]
            creole, translation = normalize(creole), normalize(translation)
            if creole and translation:
                pairs.append({"creole": creole, "translation": translation, "lang": lang})
    return pairs


def morisienmt(split: str) -> list[Pair]:
    pairs: list[Pair] = []
    for pair, lang in MORISIEN_PAIRS.items():
        archive = hf_hub_download(MORISIEN_REPO, f"data/{pair}.zip", repo_type="dataset", revision=MORISIEN_REVISION)
        with zipfile.ZipFile(archive) as bundle, bundle.open(f"{pair}_{split}.jsonl") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                row = json.loads(line)
                creole, translation = normalize(row["target"]), normalize(row["input"])
                if creole and translation:
                    pairs.append({"creole": creole, "translation": translation, "lang": lang})
    return pairs


def smol(revision: str | None = SMOL_REVISION) -> list[Pair]:
    pairs: list[Pair] = []
    for subset in SMOL_SUBSETS:
        path = hf_hub_download(SMOL_REPO, f"{subset}/en_mfe.jsonl", repo_type="dataset", revision=revision)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if subset == "smolsent":
                rows = [(row["src"], row["trg"])]
            else:
                # smoldoc holds a whole document per row. strict=True turns a misaligned one into an
                # error rather than silently dropping its tail.
                rows = list(zip(row["srcs"], row["trgs"], strict=True))
            for source, target in rows:
                creole, translation = normalize(target), normalize(source)
                if creole and translation:
                    pairs.append({"creole": creole, "translation": translation, "lang": "eng"})
    return pairs


def reserved_creole(splits: tuple[str, ...] = ("dev", "test")) -> set[str]:
    reserved: set[str] = set()
    for split in splits:
        reserved.update(loose(pair["creole"]) for pair in morisienmt(split))
    return reserved


def merge(pairs: list[Pair], reserved: set[str]) -> tuple[list[Pair], Counter]:
    kept: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    dropped: Counter = Counter()
    for pair in pairs:
        if loose(pair["creole"]) in reserved:
            dropped["leak"] += 1
            continue
        key = (pair["creole"].lower(), pair["translation"].lower())
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(pair)
    return kept, dropped
