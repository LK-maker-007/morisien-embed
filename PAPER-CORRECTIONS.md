# What the v2 measurements change in the paper

**Status: applied.** `paper/morisien-embed.tex` is the revision. Every claim below is addressed
there, and the corpus figures that had no recorded measurement are now produced by
`scripts/audit_corpus.py` and stored in `results/phase-d-corpus-audit.json`.

One entry per claim. Each gives the line in `paper/morisien-embed.tex`, what the measurement says,
and what the sentence should become. Evidence is in `results/`, produced by committed scripts.

---

## 1. "effectively all publicly available parallel text"

**Lines 41 and 101.** Untrue. `google/smol` carries 863 `smolsent` plus 1,609 `smoldoc` segments,
2,472 gold Mauritian Creole pairs, CC-BY-4.0, and none of it is in the training corpus. Only 9 of
those 2,472 appear in `train.jsonl` under the loose leak key, so it was available and independent.
It was used here as an evaluation set, which did not work: see the SMOL entry below.

**Replace with:** the corpus is MorisienMT merged with the Mauritian portion of Kreyol-MT, 35,064
pairs. Name SMOL as public parallel text the corpus does not include, and held out here.

## 2. "35,064 unique sentence pairs"

**Line 101.** 22,164 of the 35,064 rows, 63.21%, are a single Creole word. The median Creole side is
one word. Calling the corpus sentence pairs misdescribes what the model trains on, and it matters
because every benchmark is built from sentences.

**Replace with:** 35,064 pairs, of which 22,164 are single-word entries from bilingual dictionary
data and 9,183 are six words or longer.

**And report C2**, because the obvious inference is wrong. Training on only the 9,183 longer rows
scores 0.8547 on the hard benchmark against 0.8613 for a random 9,183 drawn from the full set, three
seeds each, ranges not overlapping. The dictionary helps. A random quarter of the corpus also scores
+0.0024 against the whole of it, inside seed noise, so composition matters and volume does not.
See `results/phase-c2-dictionary-ablation.json`.

## 3. "removed from the training pool by two filters"

**Line 107.** The filters remove nothing. Running them drops 0 of 69,525 rows, because MorisienMT's
test split and the training sources were already disjoint. The sentence claims a safeguard did work
it never had to do. Confirmed by `scripts/audit_corpus.py`: deduplication accounts for all 34,461
dropped rows and the leak filter for none.

**Replace with:** the filters are a check rather than a removal step. Verified absent is a stronger
claim than removed, and it is the true one.

## 4. FLORES+ "is saturated at the available corpus size"

**Lines 77 and 209.** True of the pool as v1 built it, and no longer a limit. A char n-gram TF-IDF
baseline with no neural model scores 0.8221 accuracy@1 on that pool, so it was measuring surface
overlap. xSIM++ (Chen et al., ACL 2023) supplies rule-based English-side distractors, each the gold
passage with one semantically critical token changed. The resulting pool is 996 queries over 45,029
passages, the same lexical baseline drops to 0.3273, and LaBSE falls from 0.9996 to 0.6928 accuracy@1.

**Replace with:** the saturation was a property of the pool, not of the language, and describe the
hardened one. See `results/phase-a-xsim-eng.json`.

## 5. The margin over LaBSE

**Lines 44 and 76.** The in-domain numbers hold and reproduce. What the paper cannot say once a
benchmark with headroom exists is that the model is better in general.

On the hardened pool the released recipe scores 0.8426 nDCG@10 over three seeds and untrained LaBSE
scores 0.8449. The published model is behind the baseline it is compared against, out of domain.

**Add:** the in-domain margin is real and does not transfer. This is the central correction.

## 6. "0.9653 +/- 0.0002 over three seeds"

**Lines 44 and 170.** The interval is seed variance with mined negatives held fixed. It is not the
uncertainty on the score. A paired bootstrap over queries gives roughly +/-0.010, fifty times wider,
and run-to-run drift on an identical command reaches 0.0024 on the hard benchmark.

**Replace with:** report seed spread and sampling interval separately and say which is which.

## 7. "an internal adversarial audit bounded the optimism at <= 0.01 nDCG"

**Line 216.** No committed script produces this bound and the audit is not described well enough to
repeat. Either give the procedure and the code, or withdraw the number and state the exposure
plainly: the test score was printed at the end of every training run during recipe development.

FLORES+ ships a 997-sentence `dev` split that no code in the repository has ever downloaded. It is
the clean holdout a recipe should be selected on.

## 8. "Dataset loads are pinned to exact revisions, so the pipeline is byte-stable"

**Line 129.** Untrue when written: FLORES+ was fetched with no `revision=` in three places. True now.
`morisien_embed/flores.py` pins it, both scripts import that loader, and building the FLORES
benchmark twice gives identical md5 sums for queries, corpus and qrels. Models now take `--revision`.

**No change needed to the sentence.** The code changed to match it.

## 9. The MTEB task description

**Lines 133 to 137.** Four directional subsets of 1,000 pairs each reads as four independent sets.
The four share one set of 1,000 Creole sentences, selected as the longest in the source corpus, with
selected from the long tail of the corpus: every one is at least 10 words against a training median
of one. 97.01% of test tokens are already in the training vocabulary and 74.10% of test sentences
contain no out-of-vocabulary token at all, under punctuation-insensitive tokenisation. (The figures
first recorded here, 97.05% and 73.9%, came from a measurement whose tokenisation was not written
down. Re-running it produces the values above; counting raw whitespace tokens instead gives 92.83%
and 45.60%. The paper quotes the punctuation-insensitive pair and states the method.) The claim that
roughly a third is Bible text is dropped: no committed measurement supports it.

**Replace with:** an honest description of what the four subsets share and how the sentences were
chosen.

---

## New results the paper does not contain

**The instrument.** 996 queries over 45,029 passages, built by a published method from Meta's
released augmentation. Two conditions were written before it was built: a lexical baseline below
0.50, and a strong model below ceiling. It scores 0.3273 and 0.6928.

**What each component is worth**, out of domain. Batch 48 to 128 is +0.0053. Hard-negative mining
with the cached loss is +0.0358. Matryoshka is -0.0013. The abstract credits three techniques; one
carries the result and one does nothing for retrieval quality, though it still buys truncation.

**The base model.** Fine-tuning LaBSE rather than multilingual-e5-base scores 0.8597 against 0.8426
over three seeds each with no overlap, McNemar exact p = 0.0263, paired bootstrap +0.0154 with a 95%
interval of [+0.0011, +0.0300]. It is the only full fine-tune measured that beats untrained LaBSE out
of domain.

**Why that gain is small.** LaBSE lists 110 languages and `mfe` is not among them, but `ht` is.
Its starting point on Mauritian Creole is transfer from a related French-lexified creole rather than
Creole supervision. Fine-tuning is worth +0.1735 out of domain on e5-base and +0.0148 on LaBSE, and
the e5 version still lands below untrained LaBSE while the LaBSE version lands above it. The existing
Haitian probe already measures the effect the fine-tune actually has: Mauritian preferred over
Haitian 709 times out of 1,012 against LaBSE's 351.

**What the models get wrong.** Reporting the xSIM++ error rate rather than nDCG makes the numbers
comparable to other xSIM++ results: 0.3143 for the released model, 0.3072 for LaBSE, 0.5361 for
e5-base. The breakdown is the finding. Causality alternation is 4.4% of the distractors and causes
about 51% of the errors for both models, an enrichment of 11.7x, while entity replacement is 88.2% of
the pool and causes under a third. Both models handle a swapped entity and fail on a reversed causal
direction.

**A caveat on that diagnostic.** Reading the distractor text shows entity replacement often produces
a sentence no fluent reader would accept, for example "Abderrahman Bouanane Flow is the study of the
movement of individual drivers". A model can reject those without understanding meaning, which is the
likeliest reason they are so under-represented in the errors.

**LoRA.** An r=16 adapter on LaBSE trains 3.27M of 474M parameters, scores 0.8628 out of domain, the
highest measured, with a seed spread of 0.0001 against 0.0021 for the same recipe fine-tuned in full.
It costs 0.0108 in domain, so it is reported as an ablation rather than shipped. On multilingual-e5-base
the same adapter is the worst arm measured at 0.7957, so the effect belongs to the combination.

**A benchmark that did not work.** SMOL was built as a second held-out domain and failed the second
exit condition: the lexical baseline passes at 0.3643 but LaBSE scores 0.9740, leaving no headroom.
Its distractors come from a different corpus and are separable by domain alone. Recorded in
`results/phase-a2-smol.json` because the construction error is worth reporting.

---

## What this does to the claim

v1 says a dedicated Creole model beats the general multilingual baselines. On a benchmark that can
separate models, it does not.

What survives is smaller and true. There is now an evaluation instrument for a language that had
none. The published model wins in domain and does not transfer. The base model was the wrong choice
and changing it is worth +0.0171 out of domain, which is the only configuration measured that beats
LaBSE at all. Of the three techniques the abstract credits, one carries the result. And both the
model and LaBSE fail on the same thing, reversed causal direction, at nearly twelve times its share
of the pool.
