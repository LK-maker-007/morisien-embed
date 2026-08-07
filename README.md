# morisien-embed

An embedding model for **Mauritian Creole (Kreol Morisien)** — the language ~86% of Mauritius speaks,
which today's multilingual embedding models don't reliably cover.

## Why

There is no dedicated embedding model for Kreol Morisien, and general multilingual models are weak on
it. On a held-out Creole→English retrieval benchmark (the MorisienMT test split, 1,000 pairs),
`multilingual-e5-small` reaches only **0.54 ndcg@10**. The one strong general model is **LaBSE**
(**0.94 ndcg@10**), because it was built for translation-pair retrieval — so LaBSE is the bar to beat.
This project fine-tunes an efficient multilingual base into a Creole specialist and measures it
honestly against that bar.

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
