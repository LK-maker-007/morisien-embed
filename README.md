# morisien-embed

[![CI](https://github.com/LK-maker-007/morisien-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/LK-maker-007/morisien-embed/actions/workflows/ci.yml)

To our knowledge, the first dedicated embedding model for **Mauritian Creole (Kreol Morisien)**, the
only language spoken at home by 90.0% of people in Mauritius
([2022 census](https://statsmauritius.govmu.org/Documents/Statistics/ESI/2022/EI1687/2022%20Population%20Census_Main%20Results_18112022.pdf)),
which general multilingual embedding models don't reliably cover.

**Models.** Two are released, both MIT:

| | base | params | use |
|---|---|---|---|
| [**morisien-embed-v1.5**](https://huggingface.co/Singaraj/morisien-embed-v1.5) | LaBSE | 471M | current, the only one ahead of LaBSE out of domain |
| [morisien-embed](https://huggingface.co/Singaraj/morisien-embed) | multilingual-e5-base | 278M | smaller, superseded |

Trained on 35,064 Creole↔{English,French} pairs from MorisienMT and Kreyòl-MT. That is not all the
public Creole parallel text: `google/smol` holds further pairs the training set does not contain,
enough to build a 2,462-query benchmark from, held out deliberately and
[measured](results/phase-c5-smol-training.json), in one run, as making the model worse when added.

**Paper:** [morisien-embed on Zenodo](https://doi.org/10.5281/zenodo.21877805) (concept DOI
10.5281/zenodo.21877805, always resolves to the newest version).

**Demo:** [in-browser Space](https://huggingface.co/spaces/Singaraj/morisien-embed-demo), which runs
v1.5 locally in the browser via ONNX, no server.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Singaraj/morisien-embed-v1.5")

creole = ["Mo pe al bazar aster.", "Bann zanfan pe zwe dan lakour."]
english = ["I am going to the market now.", "The children are playing in the yard."]

similarity = model.similarity(model.encode(creole), model.encode(english))
```

No prompt or prefix is required. v1 was trained with Matryoshka loss, so its embeddings can be
truncated for faster search at a small accuracy cost:
`SentenceTransformer("Singaraj/morisien-embed", truncate_dim=256)`. v1.5 also publishes ONNX
exports: `onnx/model.onnx` (fp16, transformer only) and `browser/model.onnx` (full pipeline, for
in-browser use).

## Results: Creole→English retrieval, held-out MorisienMT test (1,000 queries)

| model | params | ndcg@10 | acc@1 |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | 0.16 | 0.10 |
| BAAI/bge-m3 | 568M | 0.46 | 0.36 |
| intfloat/multilingual-e5-small | 118M | 0.54 | 0.42 |
| intfloat/multilingual-e5-base | 278M | 0.64 | 0.53 |
| intfloat/multilingual-e5-large | 560M | 0.73 | 0.65 |
| sentence-transformers/LaBSE | 471M | 0.94 | 0.91 |
| morisien-embed | 278M | 0.9655 | 0.9440 |
| **morisien-embed-v1.5** | **471M** | **0.9658** | **0.9460** |

- Three seeds of each recipe give ndcg@10 **0.9661** (sd 0.0013) for v1 and **0.9655** (sd 0.0005)
  for v1.5 ([record](results/phase-c-release-decision.json)). That is seed spread with the mined
  negatives held fixed, not a confidence interval on the score.
- v1 Creole→French: **0.9751** vs LaBSE's 0.9475. English→Creole: **0.9588** vs LaBSE's 0.9247.
- v1.5 across all three directions: Creole→English acc@1 **0.9460**, Creole→French **0.9490**,
  English→Creole **0.9389**.
- FLORES+ `mfe` (independent domain, 1,012 unseen sentences): perfect 1.0000 retrieval, though LaBSE
  also sits at that ceiling (0.9996), so the out-of-domain comparison is saturated rather than won.
- The E5 baselines are scored without their `query:`/`passage:` prompts, because the prompts score
  lower here: ndcg@10 falls to 0.52, 0.61 and 0.67 for small, base and large with
  `--query-prompt "query: " --corpus-prompt "passage: "`. The other baselines have no prompt
  convention. Reproduce any number with `scripts/evaluate.py`.

## MTEB: MorisienMTBitextMining

The held-out MorisienMT test split is now a task in [MTEB](https://github.com/embeddings-benchmark/mteb),
`MorisienMTBitextMining`, the first Mauritian Creole task in the benchmark. Both models are
registered in MTEB, and their scores are public in MTEB's results repository
([#671](https://github.com/embeddings-benchmark/results/pull/671),
[#717](https://github.com/embeddings-benchmark/results/pull/717)).

Bitext-mining F1 across the four directional subsets:

| model | mfe→eng | eng→mfe | mfe→fra | fra→mfe | avg |
|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 0.358 | 0.454 | 0.475 | 0.495 | 0.446 |
| sentence-transformers/LaBSE | 0.882 | 0.845 | 0.886 | 0.779 | 0.848 |
| morisien-embed | 0.927 | 0.909 | **0.939** | 0.924 | 0.925 |
| **morisien-embed-v1.5** | **0.930** | **0.920** | 0.933 | **0.928** | **0.928** |

This is bitext-mining F1, a different metric from the ndcg@10 retrieval numbers above. Both models
are trained on the MorisienMT corpus this split is drawn from, so MTEB records them as in-domain
(via `training_datasets`), not zero-shot.

## xSIM++: hard negatives

The retrieval and MTEB numbers above are both in-domain and, at the top of the table, close to
saturated. To separate the strong models, the benchmark builder in `scripts/build_xsim_benchmark.py`
applies the released [xSIM++](https://arxiv.org/abs/2306.12907) augmentation to FLORES+,
generating adversarial distractors that differ from the gold sentence by an entity swap, a number
change, or a reversed causal clause. 996 queries against a pool of 45,029 passages.

Error rate under the `ratio` margin, which is the mode LASER's reference `xsim.py` defaults to and
the one comparable to published xSIM++ figures. Lower is better:

| model | error rate | errors |
|---|---|---|
| intfloat/multilingual-e5-base | 0.4990 | 497 / 996 |
| morisien-embed | 0.3504 | 349 / 996 |
| sentence-transformers/LaBSE | 0.3343 | 333 / 996 |
| **morisien-embed-v1.5** | **0.2932** | **292 / 996** |

v1.5 against its own untrained base, LaBSE, is significant: exact McNemar **p = 0.00083**, bootstrap
95% interval on the difference **[−0.065, −0.018]**, which excludes zero.

These error rates were originally published scored with plain cosine similarity, which is LASER's
`absolute` mode and is *not* comparable to published xSIM++ numbers. On re-reading the method the
scoring was corrected to the margin the method specifies and every model was rescored; the
conclusions held and two of them strengthened. `scripts/xsim_score.py --margin ratio distance
absolute` reproduces all three modes, and `scripts/paired_test.py` reproduces the significance tests.

## Data

- **Training:** 35,064 Creole↔{English,French} pairs, merged from MorisienMT (MIT) and Kreyòl-MT.
  Every MorisienMT dev/test sentence is checked against them by exact matching and by a
  punctuation-, case- and accent-insensitive match (both under test). The check removes 0 of the
  69,525 raw rows, because the evaluation splits and the training sources were already disjoint
  ([audit](results/phase-d-corpus-audit.json)). Hard-negative mining keeps roughly 24,100 of the pairs
  for the contrastive stage; recorded runs range from 24,096 to 24,101. Only the trained models are
  released, never the training data; regenerate it with `scripts/build_training.py`.
- **Benchmark:** the held-out MorisienMT test split (MIT-licensed, redistributable), the basis for the
  `MorisienMTBitextMining` task on MTEB (above).

## Reproduce

[`requirements-lock.txt`](requirements-lock.txt) pins the library versions that produced v1; v1.5's
training environment is not recorded here. `pip install -e .` installs compatible current versions,
and on sentence-transformers 5.7.0 both models reproduce their in-domain scores above exactly.

```bash
pip install -e .  # CPU-only? install torch from https://download.pytorch.org/whl/cpu first (5x smaller)

python scripts/build_training.py                        # -> data/processed/train.jsonl (35K pairs)
python scripts/build_benchmark.py --target eng          # -> benchmark/data/eng
python scripts/evaluate.py sentence-transformers/LaBSE  # any baseline on the benchmark
python scripts/build_flores_benchmark.py --target eng   # independent benchmark; gated, so accept the
                                                        # FLORES+ terms on the Hub and set HF_TOKEN

# stage 1, the model that mines hard negatives for both releases
python scripts/train.py --base intfloat/multilingual-e5-base \
  --batch-size 48 --epochs 3 --no-checkpoints --output-dir models/e5-base

# stage 2: v1 starts from multilingual-e5-base, v1.5 from LaBSE
python scripts/train.py --base intfloat/multilingual-e5-base \
  --mine-with models/e5-base/final \
  --num-negatives 5 --range-min 10 --relative-margin 0.05 \
  --matryoshka --batch-size 128 --mini-batch-size 32 --epochs 3 \
  --seed 42 --no-checkpoints --output-dir models/morisien-embed
python scripts/train.py --base sentence-transformers/LaBSE \
  --mine-with models/e5-base/final \
  --num-negatives 5 --range-min 10 --relative-margin 0.05 \
  --matryoshka --batch-size 128 --mini-batch-size 32 --epochs 3 \
  --seed 42 --no-checkpoints --output-dir models/morisien-embed-v1.5
```

Training runs on a single free Kaggle T4: stage 1 takes about 12 minutes and each stage-2 run 30 to
35. Everything else runs on CPU; scoring one model on the 1,000-query benchmark takes from under a
minute to about 5 minutes on an 8-core machine, depending on model size. To smoke-test the training
loop without a GPU, add `--no-fp16 --limit 64`. Dataset loads are pinned to exact Hub revisions, so
rebuilds are byte-stable. Model loads take `--revision` (`--base-revision` in `train.py`) and follow
the branch head when it is omitted.

## Status

Both models are trained, validated and published; v1.5 is current. Validation covers 3 seeds, three
retrieval directions, an independent-domain FLORES+ check, and the xSIM++ hard-negative pool above.
The task and both model entries are merged into MTEB, with v1 marked `superseded_by` v1.5.

Known limits, each recorded under `results/`: 22,164 of the 35,064 training pairs (63%) are a single
Creole word ([audit](results/phase-d-corpus-audit.json)), and training without them makes the model
worse ([ablation](results/phase-c2-dictionary-ablation.json)); plain FLORES+ cannot separate strong
models, since a character n-gram baseline with no neural model scores 0.8221 accuracy@1 on it
([lexical control](results/phase-a-xsim-eng.json)); and adding `google/smol` made the model worse in
the one run made ([SMOL run](results/phase-c5-smol-training.json)). MorisienMT's authors built it
from books translated from English, the Bible among them, plus basic sentences written by hand
([Dabre and Sukhoo 2022](https://arxiv.org/abs/2206.02421), §4). Slang and SMS-style spelling are
untested.

## Citation

If you use this model, please cite the report along with the datasets it builds on:

```bibtex
@misc{morisien-embed,
  author    = {Singaraj B},
  title     = {morisien-embed: Text Embedding Models and Evaluation for Mauritian Creole (Kreol Morisien)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21877805},
  url       = {https://doi.org/10.5281/zenodo.21877805},
}

@article{dabre2022morisienmt,
  author  = {Dabre, Raj and Sukhoo, Aneerav},
  title   = {MorisienMT: A Dataset for Mauritian Creole Machine Translation},
  journal = {arXiv preprint arXiv:2206.02421},
  year    = {2022},
}

@inproceedings{robinson2024kreyol,
  author    = {Robinson, Nathaniel R. and others},
  title     = {Krey{\`o}l-MT: Building MT for Latin American, Caribbean and Colonial African Creole Languages},
  booktitle = {NAACL},
  year      = {2024},
}
```

---

By **Singaraj B**, [LK-maker-007](https://github.com/LK-maker-007) on GitHub,
[Singaraj](https://huggingface.co/Singaraj) on Hugging Face.
