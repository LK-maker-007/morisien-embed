# Contributing

## Code standard

- Run `ruff check .` and `ruff format .` before every commit. CI will reject unformatted code.
- Line length 100, type hints on public functions, Python 3.10+.
- Comments explain *why*, not *what*. No comments that restate the code.
- Small, single-responsibility functions. No dead code, no defensive checks against inputs that
  cannot occur.
- Every script must run end-to-end and be verified before it is committed. Data pipelines are
  seeded for reproducibility.

## Reproducibility

- All randomness (splits, shuffles, training) takes an explicit seed; default `42`.
- Data artifacts are not committed — they are regenerated from `data/download.py`. Only source
  code is versioned.
