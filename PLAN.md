# Build plan — morisien-embed

Honest, phased. Each phase has a clear exit check before moving on. GPU work runs on free Kaggle
(T4/P100, 30 hrs/week); a CPU laptop handles data prep and eval.

## Phase 0 — Foundation (DONE)
- [x] Data de-risk: 135K clean EN↔Creole pairs found (`hramphul/Kreol-Morisien`), verified quality.
- [x] Baseline: multilingual-e5-small on Creole → separation ~0.05 (weak discrimination = room to beat).
- [x] Confirmed no existing Creole embedding model on HF Hub.
- [x] Repo scaffold.

## Phase 1 — Data preparation  (laptop, CPU)
- [ ] Download + cache the datasets (`data/download.py`): hramphul/Kreol-Morisien (135K),
      the 922 curated pairs, KreolMorisienMT (check for a usable parallel config).
- [ ] Clean + dedup: drop empties, near-duplicates, length outliers, obvious noise.
- [ ] Build the training set: (english, creole) positive pairs for MultipleNegativesRankingLoss.
- [ ] **Hold out** a test split BEFORE training (never train on it) — this seeds the benchmark.
- Exit check: N clean pairs (target: 100K+), a clean held-out test set, quality spot-checked.

## Phase 2 — The benchmark FIRST  (laptop, CPU)  ← do this before training
Reasoning: the win is "moderate", so the benchmark must be HARD enough to show it. Build it before
training so we're not tempted to fit the eval.
- [ ] A retrieval task: Creole query -> correct English (or Creole) passage among many distractors.
      Distractors must be *semantically close* (hard negatives), not random, or everything scores 100%.
- [ ] Optionally a bitext-mining task (EN<->Creole alignment) and an STS-style task.
- [ ] Run the giants (multilingual-e5, LaBSE, paraphrase-multilingual-MiniLM) on it -> baseline table.
- Exit check: a benchmark where the giants score *clearly below ceiling* (proves it's hard enough).

## Phase 3 — Train  (Kaggle GPU)
- [ ] `training/train.py`: base = multilingual-e5-small, loss = MultipleNegativesRankingLoss,
      in-batch negatives, reasonable batch size, 1-3 epochs, eval each epoch on Phase-2 benchmark.
- [ ] Track: does separation / retrieval accuracy beat the giants' baseline on the HARD benchmark?
- Exit check: our model beats every giant on the Phase-2 benchmark by a clear, honest margin.

## Phase 4 — Publish  (HF Hub)
- [ ] Push model + a rigorous model card: what it is, the benchmark, the numbers vs giants, limits.
- [ ] Push the benchmark as a dataset.
- [ ] Honest claims only: "first Kreol Morisien embedder; beats general multilingual models on
      Creole retrieval by X on our benchmark." No inflation.

## Phase 5 — (Stretch) MMTEB contribution
- [ ] Package the benchmark as an MMTEB task and submit a PR (like AfriMTEB did).
- [ ] If accepted, the model ranks on an *official* leaderboard -> the strong showcase outcome.

## Kill criteria (be honest, don't sink weeks into a dead project)
- Phase 1: if <30K usable clean pairs after cleaning -> too thin, reconsider.
- Phase 2: if the giants already ace a genuinely-hard benchmark -> no room to beat, stop.
- Phase 3: if after tuning we can't clearly beat the giants on the hard benchmark -> don't publish a
  model that isn't actually better; write up the negative result instead.
