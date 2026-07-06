"""Tests for the DeepEnsemble estimator, focused on perturb construction.

Retrain mode is exercised indirectly through the end-to-end pipeline tests;
these target the weight-perturbation ("anchored") construction added to fix
issue #5 (compression not affecting uncertainty). The contracts checked here:

* perturb members are distinct from each other and from the base model,
* the pruning mask is preserved exactly for LogisticRegression and MLPClassifier,
* the same seed reproduces identical member weights,
* end to end, uncertainty now moves when the audited model is pruned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from tinyaudit import audit
from tinyaudit.compress import magnitude_prune
from tinyaudit.models.sklearn import SklearnModel
from tinyaudit.uncertainty.ensemble import DeepEnsemble, PerturbNotSupportedError

# --------------------------------------------------------------------------- #
# Shared synthetic data (many features so 0.9 pruning still leaves weights).
# --------------------------------------------------------------------------- #


def _synthetic(n: int = 500, d: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    sensitive = rng.integers(0, 2, size=n)
    X = rng.normal(0, 1, size=(n, d))
    # A dense signal so several coefficients are non-trivial.
    w = rng.normal(0, 1, size=d)
    logits = X @ w + np.where(sensitive == 1, 0.6, -0.6)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    cols = [f"x{i}" for i in range(d)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label"), pd.Series(sensitive)


def _fit_logreg(X, y) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000)
    clf.fit(np.asarray(X), np.asarray(y))
    return clf


# --------------------------------------------------------------------------- #
# Members are distinct
# --------------------------------------------------------------------------- #


def test_perturb_members_differ_from_each_other_and_base():
    X, y, _ = _synthetic()
    base = SklearnModel(_fit_logreg(X, y))

    ens = DeepEnsemble(n_members=5, seed=0, construction="perturb", perturb_scale=0.1)
    ens.fit(base, np.asarray(X), np.asarray(y))

    feats = np.asarray(X)
    base_proba = base.predict_proba(feats)
    member_probas = [m.predict_proba(feats) for m in ens.members]

    # No member reproduces the base model exactly.
    for p in member_probas:
        assert not np.allclose(p, base_proba)
    # Members are not all identical to one another.
    assert not all(np.allclose(member_probas[0], p) for p in member_probas[1:])


# --------------------------------------------------------------------------- #
# Mask preservation
# --------------------------------------------------------------------------- #


def test_perturb_preserves_pruning_mask_logreg():
    X, y, _ = _synthetic()
    pruned = magnitude_prune(SklearnModel(_fit_logreg(X, y)), 0.9)
    pruned_coef = np.asarray(pruned.estimator.coef_)
    zero_mask = pruned_coef == 0.0
    assert zero_mask.any()  # 0.9 sparsity really did zero weights

    ens = DeepEnsemble(n_members=5, seed=1, construction="perturb")
    ens.fit(pruned, np.asarray(X), np.asarray(y))

    for m in ens.members:
        coef = np.asarray(m.estimator.coef_)
        # Every pruned-away weight is still exactly zero.
        assert np.all(coef[zero_mask] == 0.0)
        # Surviving weights were actually perturbed (not a no-op).
        assert not np.allclose(coef[~zero_mask], pruned_coef[~zero_mask])


def test_perturb_preserves_pruning_mask_mlp():
    X, y, _ = _synthetic()
    clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=0)
    clf.fit(np.asarray(X), np.asarray(y))
    pruned = magnitude_prune(SklearnModel(clf), 0.9)
    zero_masks = [np.asarray(c) == 0.0 for c in pruned.estimator.coefs_]
    assert any(m.any() for m in zero_masks)

    ens = DeepEnsemble(n_members=4, seed=2, construction="perturb")
    ens.fit(pruned, np.asarray(X), np.asarray(y))

    for member in ens.members:
        for coef, mask in zip(member.estimator.coefs_, zero_masks, strict=True):
            assert np.all(np.asarray(coef)[mask] == 0.0)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_perturb_is_deterministic_across_builds():
    X, y, _ = _synthetic()
    base = SklearnModel(_fit_logreg(X, y))

    def build():
        ens = DeepEnsemble(n_members=3, seed=7, construction="perturb")
        ens.fit(base, np.asarray(X), np.asarray(y))
        return [np.asarray(m.estimator.coef_) for m in ens.members]

    first, second = build(), build()
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Unreachable weights (onnx-like) raise, do not silently retrain
# --------------------------------------------------------------------------- #


def test_perturb_raises_on_unreachable_weights():
    class _FakeOnnx:
        framework = "onnx"

        def predict(self, X):  # pragma: no cover - not reached
            return np.zeros(len(X))

        def predict_proba(self, X):  # pragma: no cover - not reached
            return np.zeros((len(X), 2))

    ens = DeepEnsemble(n_members=3, construction="perturb")
    with pytest.raises(PerturbNotSupportedError):
        ens.fit(_FakeOnnx(), np.zeros((4, 3)), np.zeros(4))


# --------------------------------------------------------------------------- #
# End-to-end regression: uncertainty now tracks compression (issue #5)
# --------------------------------------------------------------------------- #


def _mean_entropy(card) -> float:
    return next(
        m.value for m in card.uncertainty.metrics if m.name == "mean_group_predictive_entropy"
    )


def test_uncertainty_tracks_pruning_end_to_end():
    X, y, s = _synthetic()

    base = audit(_fit_logreg(X, y), data=(X, y), sensitive=s, methods=["fairness", "uncertainty"])
    pruned = audit(
        _fit_logreg(X, y),
        data=(X, y),
        sensitive=s,
        compression="prune:0.9",
        methods=["fairness", "uncertainty"],
    )

    assert base.uncertainty is not None
    assert pruned.uncertainty is not None

    base_h = _mean_entropy(base)
    pruned_h = _mean_entropy(pruned)
    # The bug was that these were identical (~0.325 either way). They must now
    # differ by a non-trivial margin because members derive from the compressed
    # weights.
    assert abs(base_h - pruned_h) > 1e-3
