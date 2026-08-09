"""Measure Haitian-vs-Mauritian Creole confusion on FLORES+ aligned triplets.

FLORES+ translates the same sentences into Mauritian (``mfe``) and Haitian (``hat``) Creole, which
makes two probes possible for any embedding model:

- **trap**: mfe→English retrieval with every same-meaning Haitian twin injected into the corpus —
  how often does the Haitian twin outrank the correct English translation?
- **discrimination**: given the English sentence, is the Mauritian translation ranked above the
  Haitian one?

Access to FLORES+ is gated: accept the dataset's terms on the Hugging Face Hub and set ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from morisien_embed import data

FLORES_REPO = "openlanguagedata/flores_plus"
LANG_FILES = {"mfe": "mfe_Latn.jsonl", "hat": "hat_Latn.jsonl", "eng": "eng_Latn.jsonl"}


def flores_split(lang: str, split: str) -> dict[int, str]:
    path = hf_hub_download(FLORES_REPO, f"{split}/{LANG_FILES[lang]}", repo_type="dataset")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return {row["id"]: data.normalize(row["text"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--split", choices=("dev", "devtest"), default="devtest")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    splits = {lang: flores_split(lang, args.split) for lang in LANG_FILES}
    ids = sorted(splits["mfe"].keys() & splits["hat"].keys() & splits["eng"].keys())
    texts = {lang: [splits[lang][sid] for sid in ids] for lang in LANG_FILES}

    model = SentenceTransformer(args.model)
    emb = {lang: model.encode(texts[lang], batch_size=args.batch_size) for lang in LANG_FILES}

    # trap: query = mfe sentence, corpus = every English sentence plus every Haitian twin
    trap = model.similarity(emb["mfe"], model.encode(texts["eng"] + texts["hat"], batch_size=args.batch_size))
    trap_hits = int((trap.argmax(dim=1) == torch.arange(len(ids))).sum())

    # discrimination: is eng[i] closer to its Mauritian translation than to its Haitian one?
    eng_mfe = model.similarity(emb["eng"], emb["mfe"]).diagonal()
    eng_hat = model.similarity(emb["eng"], emb["hat"]).diagonal()
    disc_hits = int((eng_mfe > eng_hat).sum())

    n = len(ids)
    print(f"\n{args.model}  (flores {args.split}, {n} aligned mfe/hat/eng triplets)")
    print(f"  trap            acc@1 {trap_hits / n:.4f}  ({trap_hits}/{n})")
    print(f"  discrimination  mfe>hat {disc_hits / n:.4f}  ({disc_hits}/{n})")


if __name__ == "__main__":
    main()
