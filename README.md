# morisien-embed

[![CI](https://github.com/LK-maker-007/morisien-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/LK-maker-007/morisien-embed/actions/workflows/ci.yml)

To our knowledge, the first dedicated embedding model for **Mauritian Creole (Kreol Morisien)** — the
home language of roughly 90% of Mauritius (2022 census), which general multilingual embedding models
don't reliably cover.

**Model:** [Singaraj/morisien-embed](https://huggingface.co/Singaraj/morisien-embed) · fine-tuned
from multilingual-e5-base on effectively all publicly available Creole parallel data.

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

## Data

- **Training:** 35,064 leak-free Creole↔{English,French} pairs, merged from MorisienMT (CC) and
  Kreyòl-MT, with every MorisienMT dev/test sentence removed — enforced by exact matching plus a
  punctuation-, case- and accent-insensitive check (both under test). Hard-negative mining keeps
  24,100 of these for the released model's contrastive stage. Only the trained model is released,
  never the data; regenerate it with `scripts/build_training.py`.
- **Benchmark:** the held-out MorisienMT test split (CC-licensed, redistributable), the intended basis
  for a Mauritian Creole bitext-mining task on MMTEB.

## Reproduce

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
published. Next: contributing the Kreol Morisien retrieval task to MMTEB.

---

By **Singaraj B** — [LK-maker-007](https://github.com/LK-maker-007) on GitHub,
[Singaraj](https://huggingface.co/Singaraj) on Hugging Face.
