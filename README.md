# morisien-embed

An embedding model for **Mauritian Creole (Kreol Morisien)** — the language ~86.5% of Mauritius speaks,
which today's embedding models don't cover.

## Why

If you want to build semantic search, RAG, or a chatbot in Kreol Morisien, there is currently **no
dedicated embedding model**. The multilingual giants (BGE-M3, LaBSE, multilingual-e5) don't list
Mauritian Creole (`mfe`) as a supported language, and their discrimination on it is weak
(measured separation ~0.05 on parallel pairs — they lump most Creole text into a narrow band).

This project fine-tunes a small, efficient, multilingual base into a Creole specialist, and builds a
retrieval benchmark to measure it — so the improvement is provable, not asserted.

## Honest scope (no overclaiming)

- **First** dedicated Kreol Morisien embedding model (verified: none exists on the HF Hub).
- Target: **beat the general multilingual models on Creole retrieval**, on a hard benchmark we build.
- Margin expected **moderate, not massive** — Mauritian Creole is French-based, so the giants get
  partial transfer for free. The win shows up on hard retrieval / discrimination, not on trivial pairs.
- Small model by design (~118M): efficient, deployable, free to train — and specialization beats scale
  on a narrow niche.

## Approach

- **Base:** `intfloat/multilingual-e5-small` (~118M, multilingual incl. French)
- **Data:** 30,472 unique Creole↔English/French pairs from Kreyòl-MT (`jhu-clsp/kreyol-mt`,
  `mfe-eng` + `mfe-fra`). Run `python data/download.py` to regenerate; the data is not committed.
- **Training:** sentence-transformers, `MultipleNegativesRankingLoss`, free Kaggle GPU
- **Eval:** a purpose-built hard Kreol Morisien retrieval benchmark (candidate for MMTEB contribution)

## Status

Data pipeline done (30,472 train / 1,122 test pairs). Building the retrieval benchmark next.
