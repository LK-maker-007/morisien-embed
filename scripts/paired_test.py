"""Paired significance tests between two models on the same queries.

An error rate on its own cannot say whether one model beats another: 292 misses against 333 could be
41 queries the better model genuinely handles, or the same queries falling either way. Both tests
here are paired, so they use the fact that the two models saw identical queries.

McNemar's exact test looks only at the queries where the two disagree. Under the null the winner of
each disagreement is a coin flip, so the two-sided p-value is the exact binomial tail, computed with
``math.comb`` rather than a normal approximation because the discordant count is small.

The bootstrap resamples queries with replacement and reports the 95 percentile interval on the
difference in error rate. It answers a different question from McNemar, how far the estimate would
move on another sample of queries, and the two are reported together because agreement between them
is what makes a difference credible.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def mcnemar_exact(hits_a: np.ndarray, hits_b: np.ndarray) -> dict[str, float | int]:
    """Two-sided exact McNemar on paired hit vectors, where 1 is a correct retrieval."""
    a_only = int(np.sum((hits_a == 1) & (hits_b == 0)))
    b_only = int(np.sum((hits_a == 0) & (hits_b == 1)))
    n = a_only + b_only
    if n == 0:
        return {"a_wins": 0, "b_wins": 0, "discordant": 0, "exact_two_sided_p": 1.0}
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return {
        "a_wins": a_only,
        "b_wins": b_only,
        "discordant": n,
        "exact_two_sided_p": round(min(1.0, 2 * tail), 6),
    }


def paired_bootstrap(hits_a: np.ndarray, hits_b: np.ndarray, resamples: int, seed: int) -> dict[str, float | int]:
    """Percentile interval on the error-rate difference, a minus b. Negative means a errs less."""
    rng = np.random.default_rng(seed)
    errors_a, errors_b = 1 - hits_a, 1 - hits_b
    n = len(hits_a)
    diffs = np.empty(resamples)
    for i in range(resamples):
        take = rng.integers(0, n, n)
        diffs[i] = errors_a[take].mean() - errors_b[take].mean()
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": round(float(errors_a.mean() - errors_b.mean()), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "resamples": resamples,
        "seed": seed,
    }


def load(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Hit vector and, when the file carries them, the query ids it is ordered by.

    Files written before query ids were saved have only the hits. They are still usable, but the
    alignment between the two models cannot be checked, so the caller is told rather than left to
    assume it held.
    """
    data = np.load(path, allow_pickle=True)
    return data["hits"], (data["qids"] if "qids" in data.files else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="per-query npz for model A")
    parser.add_argument("b", type=Path, help="per-query npz for model B")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    hits_a, qids_a = load(args.a)
    hits_b, qids_b = load(args.b)
    # Comparing vectors that are not the same queries in the same order would silently produce a
    # number, so this is checked rather than assumed.
    if qids_a is not None and qids_b is not None:
        if not np.array_equal(qids_a, qids_b):
            raise SystemExit(f"{args.a} and {args.b} do not cover the same queries in the same order")
        alignment = "verified by query id"
    else:
        if len(hits_a) != len(hits_b):
            raise SystemExit(f"{args.a} has {len(hits_a)} queries, {args.b} has {len(hits_b)}")
        alignment = "NOT verified: at least one file predates query ids, only lengths were compared"

    report = {
        "a": str(args.a),
        "b": str(args.b),
        "queries": int(len(hits_a)),
        "alignment": alignment,
        "error_rate_a": round(float(1 - hits_a.mean()), 4),
        "error_rate_b": round(float(1 - hits_b.mean()), 4),
        "mcnemar": mcnemar_exact(hits_a, hits_b),
        "bootstrap_error_rate": paired_bootstrap(hits_a, hits_b, args.resamples, args.seed),
    }
    print(json.dumps(report, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
