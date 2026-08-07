"""Loaders that normalize each parallel corpus to a common ``{creole, translation, lang}`` schema.

Two corpora feed training: MorisienMT (``prajdabre/MorisienMT``, CC) and Kreyòl-MT
(``jhu-clsp/kreyol-mt``). Both align Mauritian Creole (``mfe``) with English and French; this module
returns their pairs in one shape so the training builder can merge them. Text is whitespace-normalized
but case is preserved — callers lower-case only when comparing keys for dedup and leak removal.
"""

from __future__ import annotations

import io
import json
import zipfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download

KREYOL_REPO = "jhu-clsp/kreyol-mt"
KREYOL_CONFIGS = {"mfe-eng": "eng", "mfe-fra": "fra"}
CREOLE_LANG = "mfe"

MORISIEN_REPO = "prajdabre/MorisienMT"
MORISIEN_PAIRS = {"en-cr": "eng", "fr-cr": "fra"}


def _norm(text: str) -> str:
    return " ".join(text.split())


def kreyol_mt(split: str) -> list[dict[str, str]]:
    """Creole↔{English,French} pairs from Kreyòl-MT's ``mfe-eng`` and ``mfe-fra`` configs."""
    pairs: list[dict[str, str]] = []
    for config, lang in KREYOL_CONFIGS.items():
        for row in load_dataset(KREYOL_REPO, config, split=split):
            entry = row["translation"]
            if entry["src_lang"] == CREOLE_LANG:
                creole, translation = entry["src_text"], entry["tgt_text"]
            else:
                creole, translation = entry["tgt_text"], entry["src_text"]
            pairs.append({"creole": _norm(creole), "translation": _norm(translation), "lang": lang})
    return pairs


def morisienmt(split: str) -> list[dict[str, str]]:
    """Creole↔{English,French} pairs from MorisienMT. ``split`` is ``train``, ``dev`` or ``test``.

    The dataset's loader script is deprecated, so the split archives are fetched and read directly.
    """
    pairs: list[dict[str, str]] = []
    for pair, lang in MORISIEN_PAIRS.items():
        archive = hf_hub_download(MORISIEN_REPO, f"data/{pair}.zip", repo_type="dataset")
        with zipfile.ZipFile(archive) as bundle, bundle.open(f"{pair}_{split}.jsonl") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                row = json.loads(line)
                creole, translation = _norm(row["target"]), _norm(row["input"])
                if creole and translation:
                    pairs.append({"creole": creole, "translation": translation, "lang": lang})
    return pairs
