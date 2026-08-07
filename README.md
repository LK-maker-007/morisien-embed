# morisien-embed

An embedding model for **Mauritian Creole (Kreol Morisien)** — the language ~86% of Mauritius speaks,
which today's multilingual embedding models don't reliably cover.

## Why

There is no dedicated embedding model for Kreol Morisien, and general multilingual models are weak on
it. This project fine-tunes an efficient multilingual base into a Creole specialist and measures it
honestly against the strongest general model.

## Baselines — Creole→English retrieval, MorisienMT test (1,000 pairs)

| model | params | ndcg@10 | acc@1 |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | 0.16 | 0.10 |
| BAAI/bge-m3 | 568M | 0.46 | 0.36 |
| intfloat/multilingual-e5-small *(base for fine-tuning)* | 118M | 0.54 | 0.42 |
| intfloat/multilingual-e5-base | 278M | 0.64 | 0.53 |
| intfloat/multilingual-e5-large | 560M | 0.73 | 0.65 |
| **sentence-transformers/LaBSE** | 470M | **0.94** | **0.91** |

Every model is evaluated under its best prompt configuration; reproduce with `scripts/evaluate.py`.
General multilingual models — including the SOTA BGE-M3 — fall far short; only LaBSE, built for
translation-pair retrieval, is competitive. Beating **0.94 ndcg@10** is the goal.

## Data

- **Training:** 35,064 leak-free Creole↔{English,French} pairs, merged from MorisienMT (CC) and
  Kreyòl-MT, with every MorisienMT dev/test sentence removed — verified disjoint by exact and
  punctuation-insensitive match. Only the trained model is released, never the data; regenerate it
  with `scripts/build_training.py`.
- **Benchmark:** the held-out MorisienMT test split (CC-licensed, redistributable), the intended basis
  for a Mauritian Creole bitext-mining task on MMTEB.

## Usage

```bash
pip install -e .

python scripts/build_training.py                       # -> data/processed/train.jsonl (35K pairs)
python scripts/build_benchmark.py --target eng         # -> benchmark/data/eng
python scripts/evaluate.py sentence-transformers/LaBSE # baseline on the benchmark
python scripts/train.py --base intfloat/multilingual-e5-small --batch-size 128 --epochs 3
```

Training runs on a single GPU (free Kaggle T4/P100); everything else runs on CPU.

## Status

Data pipeline, benchmark, and fine-tuning script are complete and tested. Training is in progress; the
goal is to beat LaBSE (0.94 ndcg@10) on the held-out benchmark and contribute the Kreol Morisien task
to MMTEB.
