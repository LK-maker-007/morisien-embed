"""Load and merge the Mauritian Creole parallel corpora into a common schema.

Two corpora feed the model: MorisienMT (``prajdabre/MorisienMT``, CC) and Kreyòl-MT
(``jhu-clsp/kreyol-mt``). Both align Mauritian Creole (``mfe``) with English and French. Every loader
returns pairs shaped as ``{creole, translation, lang}`` with whitespace normalized and case
preserved; dedup and leak comparisons lower-case the text.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter

from datasets import load_dataset
from huggingface_hub import hf_hub_download

KREYOL_REPO = "jhu-clsp/kreyol-mt"
KREYOL_CONFIGS = {"mfe-eng": "eng", "mfe-fra": "fra"}
CREOLE_LANG = "mfe"

MORISIEN_REPO = "prajdabre/MorisienMT"
MORISIEN_PAIRS = {"en-cr": "eng", "fr-cr": "fra"}

Pair = dict[str, str]


def normalize(text: str) -> str:
    return " ".join(text.split())


def kreyol_mt(split: str) -> list[Pair]:
    """Creole↔{English,French} pairs from Kreyòl-MT's ``mfe-eng`` and ``mfe-fra`` configs."""
    pairs: list[Pair] = []
    for config, lang in KREYOL_CONFIGS.items():
        for row in load_dataset(KREYOL_REPO, config, split=split):
            entry = row["translation"]
            if entry["src_lang"] == CREOLE_LANG:
                creole, translation = entry["src_text"], entry["tgt_text"]
            else:
                creole, translation = entry["tgt_text"], entry["src_text"]
            pairs.append({"creole": normalize(creole), "translation": normalize(translation), "lang": lang})
    return pairs


def morisienmt(split: str) -> list[Pair]:
    """Creole↔{English,French} pairs from MorisienMT. ``split`` is ``train``, ``dev`` or ``test``.

    The dataset's loader script is deprecated, so the split archives are fetched and read directly.
    """
    pairs: list[Pair] = []
    for pair, lang in MORISIEN_PAIRS.items():
        archive = hf_hub_download(MORISIEN_REPO, f"data/{pair}.zip", repo_type="dataset")
        with zipfile.ZipFile(archive) as bundle, bundle.open(f"{pair}_{split}.jsonl") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                row = json.loads(line)
                creole, translation = normalize(row["target"]), normalize(row["input"])
                if creole and translation:
                    pairs.append({"creole": creole, "translation": translation, "lang": lang})
    return pairs


def reserved_creole(splits: tuple[str, ...] = ("dev", "test")) -> set[str]:
    """Lower-cased MorisienMT evaluation sentences that training must never contain."""
    reserved: set[str] = set()
    for split in splits:
        reserved.update(pair["creole"].lower() for pair in morisienmt(split))
    return reserved


def merge(pairs: list[Pair], reserved: set[str]) -> tuple[list[Pair], Counter]:
    """Drop evaluation-leaking and duplicate pairs, keeping the first occurrence of each."""
    kept: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    dropped: Counter = Counter()
    for pair in pairs:
        if pair["creole"].lower() in reserved:
            dropped["leak"] += 1
            continue
        key = (pair["creole"].lower(), pair["translation"].lower())
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(pair)
    return kept, dropped
