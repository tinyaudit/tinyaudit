"""Tests for central RNG seeding and audit() determinism."""

from __future__ import annotations

import importlib.util
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from tinyaudit import audit
from tinyaudit._seed import seed_everything

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _draw() -> tuple[float, float, object]:
    """One draw from each global RNG that seed_everything covers."""
    r = random.random()
    n = float(np.random.rand())
    t: object = None
    if _HAS_TORCH:
        import torch

        t = torch.rand(1).item()
    return r, n, t


def test_seed_everything_resets_global_state() -> None:
    seed_everything(0)
    first = _draw()

    seed_everything(0)
    second = _draw()

    assert first == second


def test_different_seeds_diverge() -> None:
    seed_everything(0)
    a = _draw()
    seed_everything(1)
    b = _draw()

    assert a != b


def test_seed_everything_sets_pythonhashseed() -> None:
    seed_everything(1234)
    assert os.environ["PYTHONHASHSEED"] == "1234"


def test_seed_everything_is_callable_without_error() -> None:
    # Covers the happy path (and, when torch is absent, the lazy-import guard).
    seed_everything(7)


def _fit(X, y) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X.to_numpy(), y.to_numpy())
    return clf


def _metric_values(card) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in card.fairness.metrics:
        out[f"fairness.{m.name}"] = m.value
    if card.uncertainty is not None:
        for m in card.uncertainty.metrics:
            out[f"uncertainty.{m.name}"] = m.value
    return out


def test_audit_is_deterministic_across_same_seed_runs(binary_classification) -> None:
    X, y, sensitive = binary_classification

    card_a = audit(_fit(X, y), data=(X, y), sensitive=sensitive, seed=42)
    card_b = audit(_fit(X, y), data=(X, y), sensitive=sensitive, seed=42)

    va = _metric_values(card_a)
    vb = _metric_values(card_b)

    assert va.keys() == vb.keys()
    for key in va:
        assert va[key] == vb[key], f"{key} differed: {va[key]} vs {vb[key]}"
