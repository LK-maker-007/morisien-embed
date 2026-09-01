"""Load FLORES+ splits, the independent Mauritian Creole evaluation source.

FLORES+ (``openlanguagedata/flores_plus``, CC-BY-SA-4.0, gated) carries Mauritian Creole: professionally
translated sentences aligned across languages by ``id``, sharing no source with the training corpora.
The devtest split holds 1,012 sentences. A dev split also exists and no code here has ever loaded it;
it is the untouched holdout a recipe should be selected on.

Access requires accepting the dataset's terms on the Hugging Face Hub and an ``HF_TOKEN``.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from morisien_embed.data import normalize

FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_REVISION = "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"  # 2026-07-27
LANG_FILES = {"mfe": "mfe_Latn.jsonl", "eng": "eng_Latn.jsonl", "fra": "fra_Latn.jsonl", "hat": "hat_Latn.jsonl"}


def flores_split(lang: str, split: str, *, revision: str | None = FLORES_REVISION) -> dict[int, str]:
    """Map sentence ``id`` to text for one language in one split.

    Args:
        lang (`str`): A key of `LANG_FILES`.
        split (`str`): `"dev"` or `"devtest"`.
        revision (`str | None`): Hub revision to pin. Pass `None` to track the branch head.

    Returns:
        `dict[int, str]`: Sentence id to whitespace-normalized text.
    """
    if lang not in LANG_FILES:
        raise ValueError(f"Unknown FLORES+ language {lang!r}, expected one of {sorted(LANG_FILES)}")
    path = hf_hub_download(FLORES_REPO, f"{split}/{LANG_FILES[lang]}", repo_type="dataset", revision=revision)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return {row["id"]: normalize(row["text"]) for row in rows}
