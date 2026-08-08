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

The first dedicated text embedding model for **Mauritian Creole (Kreol Morisien, `mfe`)** — the
home language of roughly 86% of Mauritius and covered by no dedicated embedding model before this one.

Fine-tuned from [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) on all
publicly available Creole↔{English, French} parallel data, it outperforms every general multilingual
embedding model — including [LaBSE](https://huggingface.co/sentence-transformers/LaBSE), the strongest
translation-retrieval model available — on held-out Creole retrieval in both directions.

Use it for semantic search, retrieval, RAG, bitext mining, or clustering over Kreol Morisien text.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Singaraj/morisien-embed")

creole = ["Mo pe al bazar aster.", "Bann zanfan pe zwe dan lakour."]
english = ["I am going to the market now.", "The children are playing in the yard."]

similarity = model.similarity(model.encode(creole), model.encode(english))
```

Trained with Matryoshka loss, so embeddings can be truncated for faster search at a small
accuracy cost:

```python
model = SentenceTransformer("Singaraj/morisien-embed", truncate_dim=256)
```

No prompt/prefix is required.

## Results

Creole→English retrieval on the held-out [MorisienMT](https://huggingface.co/datasets/prajdabre/MorisienMT)
test split (1,000 queries, leak-free against training data — verified by exact and
punctuation-insensitive matching):

| Model | Params | ndcg@10 | accuracy@1 |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | 0.16 | 0.10 |
| BAAI/bge-m3 | 568M | 0.46 | 0.36 |
| intfloat/multilingual-e5-small | 118M | 0.54 | 0.42 |
| intfloat/multilingual-e5-base | 278M | 0.64 | 0.53 |
| intfloat/multilingual-e5-large | 560M | 0.73 | 0.65 |
| sentence-transformers/LaBSE | 470M | 0.94 | 0.91 |
| **morisien-embed** | **278M** | **[PENDING-ENG-NDCG]** | **[PENDING-ENG-ACC]** |

Creole→French, same protocol:

| Model | ndcg@10 | accuracy@1 |
|---|---|---|
| sentence-transformers/LaBSE | 0.9475 | 0.9130 |
| **morisien-embed** | **[PENDING-FRA-NDCG]** | **[PENDING-FRA-ACC]** |

Generalization to an independent domain — [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus)
`mfe` devtest (1,012 professionally translated wikinews sentences, zero overlap with training data):

| Model | ndcg@10 | accuracy@1 |
|---|---|---|
| sentence-transformers/LaBSE | [PENDING-FLORES-LABSE] | [PENDING-FLORES-LABSE-ACC] |
| **morisien-embed** | **[PENDING-FLORES-NDCG]** | **[PENDING-FLORES-ACC]** |

Training was repeated with three random seeds; Creole→English test ndcg@10 across seeds:
**[PENDING-MEAN] ± [PENDING-STD]**. The released checkpoint is seed 42, designated before results
were seen.

Every number is reproducible from the [training repository](https://github.com/LK-maker-007/morisien-embed).

## Training

- **Data:** 35,064 unique, leak-free Creole↔{English, French} pairs — effectively all publicly
  available Mauritian Creole parallel text — merged from
  [MorisienMT](https://huggingface.co/datasets/prajdabre/MorisienMT) (CC) and
  [Kreyòl-MT](https://huggingface.co/datasets/jhu-clsp/kreyol-mt) (mixed licenses; used for training
  only, not redistributed). Every MorisienMT dev/test sentence was removed from training by exact and
  fuzzy matching.
- **Recipe:** hard-negative mining with positive-aware false-negative filtering
  (`mine_hard_negatives`: 5 negatives/anchor, `range_min=10`, `relative_margin=0.05`), then
  contrastive training with `CachedMultipleNegativesRankingLoss` (batch 128, ~760 in-batch negatives
  per anchor) wrapped in `MatryoshkaLoss` (dims 768/512/256/128/64). 3 epochs, lr 2e-5, warmup 10%,
  fp16, seed 42, single T4 GPU (~30 min).
- **Base model:** [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
  (278M parameters, MIT).

## Limitations

- **Not native-perfect.** Accuracy@1 around [PENDING-ENG-ACC] means roughly one query in twenty ranks
  a wrong translation first. Strong, but below a human bilingual speaker.
- **Register skew.** The available Creole data over-represents religious text, politics, and
  literature; highly informal or technical registers are less covered.
- **Small evaluation universe.** Retrieval is measured over ~1,000-passage corpora — standard for
  bitext benchmarks, but absolute scores would be lower against web-scale corpora.
- **One distribution family.** MorisienMT and Kreyòl-MT overlap heavily; FLORES+ is the only fully
  independent evaluation domain that exists for `mfe` today.
- **Orthographic variation.** Training data mixes pre- and post-2011 (Lortograf Kreol Morisien)
  spellings; performance on older orthography is untested.

## Citation

If you use this model, please cite the data sources it builds on:
[MorisienMT](https://arxiv.org/abs/2206.02421) (Dabre & Sukhoo, 2022) and
[Kreyòl-MT](https://arxiv.org/abs/2405.05376) (Robinson et al., NAACL 2024).

```bibtex
@misc{morisien-embed,
  author = {Singaraj B},
  title = {morisien-embed: the first text embedding model for Mauritian Creole},
  year = {2026},
  url = {https://huggingface.co/Singaraj/morisien-embed}
}
```
