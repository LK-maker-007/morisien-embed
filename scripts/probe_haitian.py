"""Measure Haitian-vs-Mauritian Creole confusion on FLORES+ aligned triplets."""

from __future__ import annotations

import argparse

import torch
from sentence_transformers import SentenceTransformer

from morisien_embed.flores import flores_split

# The shared loader knows French too, which this probe does not use.
LANGS = ("mfe", "hat", "eng")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument(
        "--revision", default=None, help="Hub revision of the model, a branch, tag or commit. Local paths ignore it."
    )
    parser.add_argument("--split", choices=("dev", "devtest"), default="devtest")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    splits = {lang: flores_split(lang, args.split) for lang in LANGS}
    ids = sorted(splits["mfe"].keys() & splits["hat"].keys() & splits["eng"].keys())
    texts = {lang: [splits[lang][sid] for sid in ids] for lang in LANGS}

    model = SentenceTransformer(args.model, revision=args.revision)
    emb = {lang: model.encode(texts[lang], batch_size=args.batch_size) for lang in LANGS}

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
