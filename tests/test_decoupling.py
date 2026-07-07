"""Regression test for the decoupling experiment's row reshaping.

The experiment script itself runs on real Adult (too slow for the unit suite),
so this covers the pure ``decoupling_rows`` reshaping against a small audit card
built from a synthetic frame: one row per sensitive group, each pairing the
point-fairness signal (selection rate) with the uncertainty signals (mean
entropy, ECE).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tinyaudit import audit

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_decoupling.py"


def _load_run_decoupling():
    spec = importlib.util.spec_from_file_location("run_decoupling", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(n: int = 400, d: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    sensitive = rng.integers(0, 2, size=n)
    X = rng.normal(0, 1, size=(n, d))
    w = rng.normal(0, 1, size=d)
    logits = X @ w + np.where(sensitive == 1, 0.6, -0.6)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    cols = [f"x{i}" for i in range(d)]
    s = pd.Series(np.where(sensitive == 1, "A", "B"))
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label"), s


def test_decoupling_rows_shape_and_keys():
    mod = _load_run_decoupling()
    X, y, s = _synthetic()
    clf = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y.to_numpy())
    card = audit(clf, data=(X, y), sensitive=s, methods=["fairness", "uncertainty"], seed=0)

    rows = mod.decoupling_rows(card, sensitive="grp", compression="none", seed=0)

    # One row per sensitive group.
    assert {r["group"] for r in rows} == {"A", "B"}
    assert len(rows) == 2

    for r in rows:
        assert set(mod._FIELDS) == set(r)  # every declared column is populated
        assert r["compression"] == "none"
        assert r["n"] > 0
        # Point-fairness and uncertainty-fairness signals are both present and finite.
        assert 0.0 <= r["selection_rate"] <= 1.0
        assert r["mean_entropy"] == r["mean_entropy"]  # not nan
        assert r["ece"] == r["ece"]  # not nan


def test_decoupling_rows_handles_missing_uncertainty():
    """Fairness-only card: uncertainty signals become nan, rows still emitted."""
    mod = _load_run_decoupling()
    X, y, s = _synthetic()
    clf = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y.to_numpy())
    card = audit(clf, data=(X, y), sensitive=s, methods=["fairness"], seed=0)
    assert card.uncertainty is None

    rows = mod.decoupling_rows(card, sensitive="grp", compression="none", seed=0)
    assert {r["group"] for r in rows} == {"A", "B"}
    for r in rows:
        assert np.isnan(r["mean_entropy"])
        assert np.isnan(r["ece"])
