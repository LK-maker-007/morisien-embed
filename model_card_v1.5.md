---
language:
  - mfe
  - en
license: mit
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - sentence-transformers
  - sentence-similarity
  - feature-extraction
  - mauritian-creole
  - kreol-morisien
base_model: sentence-transformers/LaBSE
datasets:
  - prajdabre/KreolMorisienMT
  - jhu-clsp/kreyol-mt
model-index:
  - name: morisien-embed-v1.5
    results:
      - task:
          type: bitext-mining
          name: Bitext Retrieval
        dataset:
          type: prajdabre/KreolMorisienMT
          name: MorisienMT test (Creole to English)
        metrics:
          - type: ndcg_at_10
            value: 0.9654
          - type: accuracy_at_1
            value: 0.9460
      - task:
          type: bitext-mining
          name: Bitext Retrieval
        dataset:
          type: openlanguagedata/flores_plus
          name: FLORES+ devtest with xSIM++ distractors (Creole to English)
        metrics:
          - type: ndcg_at_10
            value: 0.8616
          - type: accuracy_at_1
            value: 0.7239
---

# morisien-embed-v1.5

A sentence embedding model for Mauritian Creole (Kreol Morisien, `mfe`), fine-tuned from
[LaBSE](https://huggingface.co/sentence-transformers/LaBSE).

This supersedes [morisien-embed](https://huggingface.co/Singaraj/morisien-embed) for Creole to
English retrieval. That model is left in place: it is the checkpoint described in the preprint and
in the MTEB task, and its weights are unchanged.

## What changed

The first version was fine-tuned from multilingual-e5-base. On a benchmark hard enough to separate
models it scores below untrained LaBSE, so its advantage did not hold outside the domain it was
trained on. This version starts from LaBSE instead and is the first configuration measured that
beats untrained LaBSE on that benchmark.

The gain is modest and the honest figures are below.

## Results

Creole to English. The in-domain split is the held-out MorisienMT test set. The out-of-domain pool
is FLORES+ devtest with distractors from the released
[xSIM++](https://arxiv.org/abs/2306.12907) augmentation, 996 queries over 45,029 passages, where
each distractor is a gold passage with one meaning-critical token changed.

| model | in-domain nDCG@10 | out-of-domain nDCG@10 |
|---|---|---|
| multilingual-e5-base | 0.6402 | 0.6691 |
| LaBSE | 0.9393 | 0.8449 |
| morisien-embed | 0.9661 | 0.8426 |
| **morisien-embed-v1.5** | **0.9654** | **0.8616** |

Three seeds of this recipe give 0.9655 (sd 0.0005) in-domain and 0.8597 (sd 0.0021) out of domain.
The released checkpoint is seed 42. Run-to-run drift on an identical command reaches 0.0024 on the
out-of-domain pool, which is worth knowing before reading small differences.

Against LaBSE the fine-tune is worth +0.026 in-domain and +0.015 out of domain. Against the first
version it holds the in-domain score and adds 0.019 out of domain.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Singaraj/morisien-embed-v1.5")
creole = model.encode(["Mo pe al bazar aster."])
english = model.encode(["I am going to the market now."])
```

No query or passage prefix is used at training or inference.

## Training

LaBSE fine-tuned on 35,064 Creole to English and Creole to French pairs from MorisienMT and the
Mauritian portion of Kreyol-MT. A first-stage model mines hard negatives with positive-aware
filtering, five per anchor at a minimum rank of 10 and a relative margin of 0.05, which leaves
24,096 pairs with a full negative set. The released checkpoint trains on those with a cached
multiple-negatives ranking loss at batch 128, wrapped in a Matryoshka loss over 768, 512, 256, 128
and 64 dimensions, for three epochs at learning rate 2e-5 on one T4.

Code, data construction and the evaluation harness are at
[LK-maker-007/morisien-embed](https://github.com/LK-maker-007/morisien-embed).

## What is known about its weaknesses

**The margin over LaBSE is small.** 0.015 nDCG@10 out of domain, against a measured run-to-run
drift of 0.0024. It is consistent across three seeds with no overlap, and it is not large.

**Most of what the model knows comes from LaBSE.** LaBSE lists 110 languages and Mauritian Creole is
not among them, but Haitian Creole is. Its starting point on this language is transfer from a
related French-lexified creole. Fine-tuning is worth 0.174 out of domain when applied to
multilingual-e5-base and 0.015 when applied to LaBSE, which is what that head start looks like.

**Haitian Creole is close enough to confuse it.** On FLORES+ triplets, injecting a same-meaning
Haitian twin for every sentence drops Creole to English accuracy@1 substantially for every
multilingual model tested. Mixed Mauritian and Haitian corpora will degrade retrieval.

**The training corpus is largely a dictionary.** 22,164 of the 35,064 pairs are a single Creole
word, and the median Creole side is one word. Removing them makes the model worse, and so does
adding 2,468 sentence-level pairs from `google/smol`, so the corpus appears to sit near a local
optimum for this recipe rather than being straightforwardly improvable.

**Both this model and LaBSE fail the same way.** On the out-of-domain pool, distractors made by
reversing a causal relation are 4.4% of the corpus and cause about 51% of the errors for both, while
entity substitutions are 88% of the corpus and cause under a third. Neither model reads causal
direction reliably.

**Not measured for this checkpoint.** Creole to French retrieval, the MTEB
MorisienMTBitextMining task, and Matryoshka truncation quality. All three are reported for the first
version and none has been rerun here.

**Sequence length is 256 tokens**, LaBSE's default, against 512 for the first version.

## Citation

The preprint describes the first version. A revision covering this one is in preparation.

```bibtex
@misc{morisien-embed,
  title  = {morisien-embed: A Dedicated Text Embedding Model and Benchmark for Mauritian Creole},
  author = {Singaraj B},
  year   = {2026},
  doi    = {10.5281/zenodo.21877806}
}
```
