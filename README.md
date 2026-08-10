# morisien-embed

[![CI](https://github.com/LK-maker-007/morisien-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/LK-maker-007/morisien-embed/actions/workflows/ci.yml)

To our knowledge, the first dedicated embedding model for **Mauritian Creole (Kreol Morisien)** — the
home language of roughly 90% of Mauritius (2022 census), which general multilingual embedding models
don't reliably cover.

**Model:** [Singaraj/morisien-embed](https://huggingface.co/Singaraj/morisien-embed) · fine-tuned
from multilingual-e5-base on effectively all publicly available Creole parallel data.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Singaraj/morisien-embed")

creole = ["Mo pe al bazar aster.", "Bann zanfan pe zwe dan lakour."]
english = ["I am going to the market now.", "The children are playing in the yard."]

similarity = model.similarity(model.encode(creole), model.encode(english))
```

No prompt or prefix is required. Trained with Matryoshka loss, so embeddings can be truncated for
faster search at a small accuracy cost: `SentenceTransformer("Singaraj/morisien-embed", truncate_dim=256)`.

## Results — Creole→English retrieval, held-out MorisienMT test (1,000 queries)

| model | params | ndcg@10 | acc@1 |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | 0.16 | 0.10 |
| BAAI/bge-m3 | 568M | 0.46 | 0.36 |
| intfloat/multilingual-e5-small | 118M | 0.54 | 0.42 |
| intfloat/multilingual-e5-base | 278M | 0.64 | 0.53 |
| intfloat/multilingual-e5-large | 560M | 0.73 | 0.65 |
| sentence-transformers/LaBSE | 470M | 0.94 | 0.91 |
| **morisien-embed** | **278M** | **0.9655** | **0.9440** |

- Stable across 3 seeds: ndcg@10 **0.9653 ± 0.0002**.
- Creole→French: **0.9751** vs LaBSE's 0.9475.
- English→Creole (reversed direction): **0.9588** vs LaBSE's 0.9247.
- FLORES+ `mfe` (independent domain, 1,012 unseen sentences): perfect 1.0000 retrieval — though LaBSE
  also sits at that ceiling (0.9996), so the out-of-domain comparison is saturated rather than won.
- E5 baselines were ablated with and without their `query:`/`passage:` prompts on an earlier
  iteration of the benchmark (no prefix won every time); final-benchmark numbers use the winning
  no-prefix configuration. The other baselines have no prompt convention. Reproduce any number with
  `scripts/evaluate.py`.

## MTEB — MorisienMTBitextMining

The held-out MorisienMT test split is now a task in [MTEB](https://github.com/embeddings-benchmark/mteb),
`MorisienMTBitextMining` — the first Mauritian Creole task in the benchmark. The model is registered in
MTEB and its scores are on the [leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

Bitext-mining F1 across the four directional subsets:

| model | mfe→eng | eng→mfe | mfe→fra | fra→mfe | avg |
|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 0.358 | 0.454 | 0.475 | 0.495 | 0.446 |
| sentence-transformers/LaBSE | 0.882 | 0.845 | 0.886 | 0.779 | 0.848 |
| **morisien-embed** | **0.927** | **0.909** | **0.939** | **0.924** | **0.925** |

This is bitext-mining F1, a different metric from the ndcg@10 retrieval numbers above. morisien-embed is
trained on the MorisienMT corpus this split is drawn from, so MTEB records it as in-domain
(via `training_datasets`), not zero-shot.

## Data

- **Training:** 35,064 leak-free Creole↔{English,French} pairs, merged from MorisienMT (MIT) and
  Kreyòl-MT, with every MorisienMT dev/test sentence removed — enforced by exact matching plus a
  punctuation-, case- and accent-insensitive check (both under test). Hard-negative mining keeps
  24,100 of these for the released model's contrastive stage. Only the trained model is released,
  never the data; regenerate it with `scripts/build_training.py`.
- **Benchmark:** the held-out MorisienMT test split (MIT-licensed, redistributable), the basis for the
  `MorisienMTBitextMining` task on MTEB (above).

## Reproduce

The exact library versions that produced the released checkpoint are pinned in
[`requirements-lock.txt`](requirements-lock.txt); `pip install -e .` alone installs compatible
current versions, which reproduce the reported metrics.

```bash
pip install -e .  # CPU-only? install torch from https://download.pytorch.org/whl/cpu first (5x smaller)

python scripts/build_training.py                        # -> data/processed/train.jsonl (35K pairs)
python scripts/build_benchmark.py --target eng          # -> benchmark/data/eng
python scripts/evaluate.py sentence-transformers/LaBSE  # any baseline on the benchmark
python scripts/build_flores_benchmark.py --target eng   # independent benchmark; gated — accept the
                                                        # FLORES+ terms on the Hub and set HF_TOKEN

python scripts/train.py --base intfloat/multilingual-e5-base \
  --batch-size 48 --epochs 3 --no-checkpoints --output-dir models/e5-base
python scripts/train.py --base intfloat/multilingual-e5-base \
  --mine-with models/e5-base/final \
  --num-negatives 5 --range-min 10 --relative-margin 0.05 \
  --matryoshka --batch-size 128 --mini-batch-size 32 --epochs 3 \
  --seed 42 --no-checkpoints --output-dir models/morisien-embed
```

Training runs on a single free Kaggle T4 (~45 min total); everything else runs on CPU (a full LaBSE
evaluation takes ~45 min on an 8-core machine). To smoke-test the training loop without a GPU, add
`--no-fp16 --limit 64`. Dataset loads are pinned to exact Hub revisions, so rebuilds are byte-stable.

## Status

Model trained, validated (3 seeds, three retrieval directions, independent-domain check) and
published. The task, model, and results are merged into MTEB — see the section above.

## Citation

If you use this model, please cite it along with the datasets it builds on:

```bibtex
@misc{morisien-embed,
  author = {Singaraj B},
  title  = {morisien-embed: a dedicated text embedding model for Mauritian Creole},
  year   = {2026},
  url    = {https://huggingface.co/Singaraj/morisien-embed},
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

By **Singaraj B** — [LK-maker-007](https://github.com/LK-maker-007) on GitHub,
[Singaraj](https://huggingface.co/Singaraj) on Hugging Face.
