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

### Other retrieval directions

Held-out MorisienMT test, nDCG@10, all three measured in one run in which `morisien-embed`
reproduces its published scores exactly.

| direction | LaBSE | morisien-embed | **morisien-embed-v1.5** |
|---|---|---|---|
| Creole to English | 0.9393 | 0.9655 | **0.9658** |
| Creole to French | 0.9475 | 0.9751 | **0.9738** |
| English to Creole | 0.9247 | 0.9588 | **0.9642** |

The in-domain figure is 0.9658 here against 0.9654 on the training run's own record, with an
identical accuracy@1 of 0.9460. That gap is a CPU against GPU difference and is smaller than the
0.0024 run-to-run drift.

### MTEB MorisienMTBitextMining

F1 on task revision `45f511e8`. LaBSE and `morisien-embed` reproduce their published means of 0.848
and 0.925, so all three rows are on the same footing.

| model | mfe>eng | eng>mfe | mfe>fra | fra>mfe | mean |
|---|---|---|---|---|---|
| LaBSE | 0.882 | 0.845 | 0.886 | 0.779 | 0.848 |
| morisien-embed | 0.927 | 0.909 | **0.939** | 0.924 | 0.925 |
| **morisien-embed-v1.5** | **0.930** | **0.920** | 0.933 | **0.928** | **0.928** |

The four subsets are built over one set of 999 Creole sentences rather than four independent
samples, so the mean is not an average of four independent measurements.

### Matryoshka truncation

Creole to English, nDCG@10. Half the embedding costs about 0.005.

| dimensions | 768 | 512 | 256 | 128 | 64 |
|---|---|---|---|---|---|
| morisien-embed-v1.5 | 0.9658 | 0.9643 | 0.9612 | 0.9565 | 0.9335 |

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
reversing a causal relation are 4.35% of the corpus and cause 49.7% of this model's errors and 46.8%
of LaBSE's, while entity substitutions are 88.2% of the corpus and cause about a third of each.
Neither model reads causal direction reliably.

**On the metric the benchmark defines.** xSIM++ scores with a margin-based similarity rather than
plain cosine, and the error rates above use it: 0.2932 for this model against 0.3343 for untrained
LaBSE, McNemar exact p = 0.00083 and a paired bootstrap 95% interval of [-0.0653, -0.0181] over 996
queries. Under plain cosine the same comparison is 0.2751 against 0.3072, so the advantage is
slightly larger on the specified scoring than on the one the first version reported.

**Sequence length is 256 tokens**, LaBSE's default, against 512 for the first version. Every
benchmark used here is single sentences, so nothing measured exercises the difference.

**Requires sentence-transformers 6.0 or newer.** This checkpoint was serialised by a build that
writes `Normalize` as `sentence_transformers.base.modules.normalize`, a path that does not exist
before 6.0, so older installs raise `ModuleNotFoundError` on load. The first version has no such
constraint.

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
