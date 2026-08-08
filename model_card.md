---
language:
  - mfe
  - en
  - fr
license: mit
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - sentence-transformers
  - sentence-similarity
  - feature-extraction
  - mauritian-creole
  - kreol-morisien
  - matryoshka
base_model: intfloat/multilingual-e5-base
datasets:
  - prajdabre/MorisienMT
  - jhu-clsp/kreyol-mt
---

# morisien-embed

To our knowledge, the first dedicated text embedding model for **Mauritian Creole (Kreol Morisien,
`mfe`)** — the home language of roughly 90% of Mauritius (2022 census).

Fine-tuned from [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) on
effectively all publicly available Creole↔{English, French} parallel data, it outperforms every
general multilingual embedding model we evaluated — including
[LaBSE](https://huggingface.co/sentence-transformers/LaBSE), the strongest of them on this task — on
held-out Creole retrieval in both directions.

Use it for semantic search, retrieval, RAG, bitext mining, or clustering over Kreol Morisien text.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Singaraj/morisien-embed")

creole = ["Mo pe al bazar aster.", "Bann zanfan pe zwe dan lakour."]
english = ["I am going to the market now.", "The children are playing in the yard."]

similarity = model.similarity(model.encode(creole), model.encode(english))
```

Trained with Matryoshka loss, so embeddings can be truncated for faster search at a small,
measured accuracy cost (ndcg@10 on the benchmark below: 0.9591 at 256 dims, 0.9531 at 128):

```python
model = SentenceTransformer("Singaraj/morisien-embed", truncate_dim=256)
```

No prompt/prefix is required.

## Results

Creole→English retrieval on the held-out [MorisienMT](https://huggingface.co/datasets/prajdabre/MorisienMT)
test split (1,000 queries, leak-free against training data — enforced in the data pipeline by exact
matching and by a punctuation-, case- and accent-insensitive check):

| Model | Params | ndcg@10 | accuracy@1 |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | 0.16 | 0.10 |
| BAAI/bge-m3 | 568M | 0.46 | 0.36 |
| intfloat/multilingual-e5-small | 118M | 0.54 | 0.42 |
| intfloat/multilingual-e5-base | 278M | 0.64 | 0.53 |
| intfloat/multilingual-e5-large | 560M | 0.73 | 0.65 |
| sentence-transformers/LaBSE | 470M | 0.94 | 0.91 |
| **morisien-embed** | **278M** | **0.9655** | **0.9440** |

Creole→French, same protocol:

| Model | ndcg@10 | accuracy@1 |
|---|---|---|
| sentence-transformers/LaBSE | 0.9475 | 0.9130 |
| **morisien-embed** | **0.9751** | **0.9530** |

Generalization to an independent domain — [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus)
`mfe` devtest (1,012 professionally translated sentences from Wikinews, Wikijunior and Wikivoyage,
zero overlap with training data):

| Model | ndcg@10 | accuracy@1 |
|---|---|---|
| sentence-transformers/LaBSE | 0.9996 | 0.9990 |
| **morisien-embed** | **1.0000** | **1.0000** |

Both models sit at the ceiling of this benchmark — FLORES+ sentences are long and distinctive, so
1,012-way retrieval saturates. Read this as evidence of zero out-of-domain degradation, not as a
margin over LaBSE.

The contrastive stage was repeated with three random seeds over the same deterministically mined
negative set; Creole→English test ndcg@10 across seeds: **0.9653 ± 0.0002** (accuracy@1
**0.9433 ± 0.0006**). The released checkpoint is seed 42, designated before results were seen.

Every number in the tables above is reproducible from the
[training repository](https://github.com/LK-maker-007/morisien-embed) (Matryoshka figures via
`scripts/evaluate.py --truncate-dim`). The Haitian-proximity and case-sensitivity figures under
Limitations come from an internal adversarial audit of the released checkpoint.

## Training

- **Data:** 35,064 unique, leak-free Creole↔{English, French} pairs — effectively all publicly
  available Mauritian Creole parallel text — merged from
  [MorisienMT](https://huggingface.co/datasets/prajdabre/MorisienMT) (CC) and
  [Kreyòl-MT](https://huggingface.co/datasets/jhu-clsp/kreyol-mt) (mixed licenses; used for training
  only, not redistributed). Every MorisienMT dev/test sentence is removed from training by exact
  matching and by a punctuation-, case- and accent-insensitive check.
- **Recipe:** hard-negative mining with positive-aware false-negative filtering
  (`mine_hard_negatives`: 5 negatives/anchor, `range_min=10`, `relative_margin=0.05`). The margin
  filter is strict: 24,100 of the 35,064 pairs survived with a full negative set, and the released
  checkpoint's contrastive stage trained on those 24,100 tuples (the stage-1 mining model itself was
  trained on all 35,064). Contrastive training uses `CachedMultipleNegativesRankingLoss` (batch 128,
  767 in-batch negatives per anchor) wrapped in `MatryoshkaLoss` (dims 768/512/256/128/64). 3 epochs,
  lr 2e-5, warmup 10%, fp16, seed 42, single T4 GPU (~30 min contrastive + ~11 min mining).
- **Base model:** [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
  (278M parameters, MIT).

## Limitations

- **Not native-perfect.** Accuracy@1 around 0.944 means roughly one query in eighteen ranks
  a wrong translation first. Strong, but below a human bilingual speaker.
- **Register skew.** The available Creole data over-represents religious text, politics, and
  literature; highly informal or technical registers are less covered.
- **Small evaluation universe.** Retrieval is measured over ~1,000-passage corpora — standard for
  bitext benchmarks, but absolute scores would be lower against web-scale corpora.
- **One distribution family.** MorisienMT and Kreyòl-MT overlap heavily, and the only fully
  independent evaluation domain for `mfe` (FLORES+) is saturated at this corpus size — so the margin
  over LaBSE is demonstrated in-domain only.
- **Haitian Creole proximity.** Like every multilingual embedder we tested, the model embeds Haitian
  Creole close to Mauritian Creole: with same-meaning Haitian sentences injected into a FLORES-based
  corpus, mfe→eng accuracy@1 drops from 1.00 to 0.71 (the Haitian twin outranks the English
  translation) — and LaBSE degrades less on this same trap (to 0.79). On a related eng→{mfe, hat}
  discrimination test the fine-tune picks the correct Mauritian translation 306/400 times vs LaBSE's
  170/400, and wrong-meaning Haitian text is never confused — but mixed mfe/hat corpora will degrade
  retrieval.
- **Case sensitivity.** ALL-CAPS text embeds measurably differently from its lower-case form
  (cosine ≈ 0.81 to the same sentence); caps-heavy text retrieves worse.
- **Protocol note.** During recipe development the held-out test score was printed at the end of each
  training run, so recipe selection had test visibility; an internal adversarial audit bounded the
  resulting optimism at ≤ ~0.01 ndcg. The 3-seed replication was run after the recipe was frozen. Leak filtering
  reserves the Creole side of every evaluation pair; English/French target texts are not reserved, and
  an audit found 1 of 999 benchmark passages also occurring in training as the translation of a
  different Creole sentence (dropping it moves ndcg@10 by less than 0.0001). The accent-insensitive
  half of the leak check was added after the released run; it verifiably leaves the training set
  byte-identical, since the sources were already disjoint at that level.
- **Orthographic variation.** Training data mixes pre- and post-2011 (Lortograf Kreol Morisien)
  spellings; performance on older orthography is untested.

## Citation

If you use this model, please cite the data sources it builds on:
[MorisienMT](https://arxiv.org/abs/2206.02421) (Dabre & Sukhoo, 2022) and
[Kreyòl-MT](https://arxiv.org/abs/2405.05376) (Robinson et al., NAACL 2024).

```bibtex
@misc{morisien-embed,
  author = {Singaraj B},
  title = {morisien-embed: a dedicated text embedding model for Mauritian Creole},
  year = {2026},
  url = {https://huggingface.co/Singaraj/morisien-embed}
}
```
