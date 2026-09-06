"""Tests for the xSIM margin scoring in ``scripts/xsim_score.py``.

The reference is LASER's ``source/xsim.py``. Its ``_score_knn`` reranks the ``k`` nearest cosine
neighbours by a margin that divides (ratio) or subtracts (distance) the mean of the two sides' own
nearest-neighbour cosines. Getting any part of that wrong produces a plausible number that no
published xSIM++ result can be compared against, which is the failure these tests exist to catch, so
the reference is transcribed here rather than imported.

Checked by mutation. Reverting any of these fails at least one test: which side the denominator
averages, ratio against distance, the mean over the k neighbours, the argmax, and how many
neighbours are reranked. Dropping the division by two is equivalent under ``ratio``, where it scales
every candidate's score by the same constant and cannot change the argmax, so it is caught only by
the ``distance`` parametrization. That is why both margins are parametrized rather than just the
default.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "xsim_score.py"


def load_module():
    spec = importlib.util.spec_from_file_location("xsim_score", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def laser_reference(x: np.ndarray, y: np.ndarray, margin: str, k: int) -> np.ndarray:
    """Transcription of LASER's ``_score_knn`` and ``score_margin``, deliberately literal."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    if margin == "absolute":
        return (x @ y.T).argmax(axis=1)
    sim_xy = x @ y.T
    idx_xy = np.argsort(-sim_xy, axis=1)[:, :k]
    cos_xy = np.take_along_axis(sim_xy, idx_xy, axis=1)
    cos_yx = np.sort(y @ x.T, axis=1)[:, ::-1][:, :k]
    avg_xy, avg_yx = cos_xy.mean(axis=1), cos_yx.mean(axis=1)
    scores = np.zeros((x.shape[0], k))
    for i in range(x.shape[0]):
        for j in range(k):
            a, b = cos_xy[i, j], (avg_xy[i] + avg_yx[idx_xy[i, j]]) / 2
            scores[i, j] = a / b if margin == "ratio" else a - b
    best = scores.argmax(axis=1)
    return np.array([idx_xy[i, best[i]] for i in range(x.shape[0])])


def normalized(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array / np.linalg.norm(array, axis=1, keepdims=True))


@pytest.mark.parametrize("margin", ["ratio", "distance", "absolute"])
@pytest.mark.parametrize(("queries", "passages", "dim", "k"), [(50, 400, 32, 4), (17, 90, 16, 2), (30, 250, 24, 8)])
def test_retrieve_matches_the_laser_reference(margin: str, queries: int, passages: int, dim: int, k: int) -> None:
    module = load_module()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(queries, dim)).astype(np.float32)
    y = rng.normal(size=(passages, dim)).astype(np.float32)

    got = np.array(module.retrieve(normalized(x), normalized(y), margin, k))

    assert (got == laser_reference(x.copy(), y.copy(), margin, k)).all()


def test_margin_reranks_only_the_k_nearest_neighbours() -> None:
    """The margin cannot promote a passage outside the top ``k`` by cosine, however good its margin."""
    module = load_module()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(25, 16)).astype(np.float32)
    y = rng.normal(size=(200, 16)).astype(np.float32)
    q, c = normalized(x), normalized(y)
    k = 4

    top_k = (q @ c.T).topk(k, dim=1).indices
    chosen = module.retrieve(q, c, "ratio", k)

    assert all(pick in set(row.tolist()) for pick, row in zip(chosen, top_k, strict=True))


def test_absolute_is_plain_nearest_neighbour_and_ignores_k() -> None:
    """``absolute`` is the pre-margin behaviour, and it must not depend on k."""
    module = load_module()
    rng = np.random.default_rng(3)
    q = normalized(rng.normal(size=(20, 12)).astype(np.float32))
    c = normalized(rng.normal(size=(150, 12)).astype(np.float32))

    expected = (q @ c.T).argmax(dim=1).tolist()

    assert module.retrieve(q, c, "absolute", 4) == expected
    assert module.retrieve(q, c, "absolute", 9) == expected


def test_ratio_and_absolute_disagree_somewhere() -> None:
    """A margin that never changed the answer would make the whole distinction untestable."""
    module = load_module()
    rng = np.random.default_rng(11)
    q = normalized(rng.normal(size=(120, 16)).astype(np.float32))
    c = normalized(rng.normal(size=(600, 16)).astype(np.float32))

    assert module.retrieve(q, c, "ratio", 4) != module.retrieve(q, c, "absolute", 4)


@pytest.mark.parametrize("margin", ["ratio", "distance", "absolute"])
@pytest.mark.parametrize("chunk", [7, 32, 99, 1000])
def test_chunking_does_not_change_the_answer(margin: str, chunk: int) -> None:
    """The pool is streamed in chunks, so the running top-k merge must survive any chunk boundary.

    With the default chunk size no test pool is large enough to take the loop round twice, which
    would leave the merge unexercised. Forcing small chunks is what makes it real.
    """
    module = load_module()
    module.CHUNK = chunk
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 16)).astype(np.float32)
    y = rng.normal(size=(500, 16)).astype(np.float32)

    got = np.array(module.retrieve(normalized(x), normalized(y), margin, 4))

    assert (got == laser_reference(x.copy(), y.copy(), margin, 4)).all()


def test_retrieve_many_agrees_with_one_margin_at_a_time() -> None:
    """Scoring every margin from one pass must equal scoring each on its own."""
    module = load_module()
    rng = np.random.default_rng(13)
    q = normalized(rng.normal(size=(60, 16)).astype(np.float32))
    c = normalized(rng.normal(size=(700, 16)).astype(np.float32))
    margins = ("ratio", "distance", "absolute")

    together = module.retrieve_many(q, c, margins, 4)

    assert together == {m: module.retrieve(q, c, m, 4) for m in margins}
