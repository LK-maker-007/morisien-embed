from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_test.py"


def load_module():
    spec = importlib.util.spec_from_file_location("paired_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vectors(a_only: int, b_only: int, both: int = 0, neither: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build paired hit vectors with a chosen disagreement pattern."""
    hits_a = np.array([1] * a_only + [0] * b_only + [1] * both + [0] * neither)
    hits_b = np.array([0] * a_only + [1] * b_only + [1] * both + [0] * neither)
    return hits_a, hits_b


@pytest.mark.parametrize(
    ("a_only", "b_only", "expected"),
    [
        (1, 0, 1.0),  # 2 * C(1,0)/2^1
        (5, 0, 0.0625),  # 2 * C(5,0)/2^5 = 2/32
        (10, 0, 0.001953),  # 2 * 1/1024
        (6, 1, 0.125),  # 2 * (C(7,0)+C(7,1))/2^7 = 16/128
        (5, 5, 1.0),  # symmetric, the doubled tail exceeds 1 and is capped
    ],
)
def test_mcnemar_matches_the_binomial_tail(a_only: int, b_only: int, expected: float) -> None:
    module = load_module()
    hits_a, hits_b = vectors(a_only, b_only, both=20, neither=20)

    got = module.mcnemar_exact(hits_a, hits_b)

    assert got["a_wins"] == a_only
    assert got["b_wins"] == b_only
    assert got["discordant"] == a_only + b_only
    assert got["exact_two_sided_p"] == pytest.approx(expected, abs=1e-6)


def test_mcnemar_ignores_agreements() -> None:
    """Only disagreements carry information, so padding with agreements must not move the p-value."""
    module = load_module()
    lean = module.mcnemar_exact(*vectors(6, 1))
    padded = module.mcnemar_exact(*vectors(6, 1, both=500, neither=500))

    assert lean == padded


def test_mcnemar_on_identical_models_is_one() -> None:
    module = load_module()
    hits = np.array([1, 0, 1, 1, 0])

    assert module.mcnemar_exact(hits, hits)["exact_two_sided_p"] == 1.0


def test_bootstrap_sign_and_determinism() -> None:
    """A model that errs less must give a negative difference, and the seed must fix the interval.

    A thousand queries rather than a hundred: at a hundred the interval is coarse enough that two
    seeds round to the same four decimals, which would make the last assertion flaky rather than
    wrong.
    """
    module = load_module()
    hits_a = np.array([1] * 900 + [0] * 100)  # 10% error
    hits_b = np.array([1] * 600 + [0] * 400)  # 40% error

    first = module.paired_bootstrap(hits_a, hits_b, resamples=2000, seed=0)
    again = module.paired_bootstrap(hits_a, hits_b, resamples=2000, seed=0)
    other = module.paired_bootstrap(hits_a, hits_b, resamples=2000, seed=1)

    assert first["diff"] == pytest.approx(-0.30, abs=1e-9)
    assert first["ci95"][1] < 0, "an interval excluding zero should stay below it"
    assert first == again, "same seed must reproduce"
    assert other["ci95"] != first["ci95"], "a different seed should move the interval"


def test_bootstrap_on_equal_models_straddles_zero() -> None:
    module = load_module()
    hits = np.array([1] * 70 + [0] * 30)

    got = module.paired_bootstrap(hits, hits.copy(), resamples=2000, seed=0)

    assert got["diff"] == 0.0
    assert got["ci95"] == [0.0, 0.0]
