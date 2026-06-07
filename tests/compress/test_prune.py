"""Tests for magnitude_prune in src/tinyaudit/compress/prune.py.

Coverage targets:
  - sparsity=0.0 returns a functionally identical model (predict/predict_proba)
  - sparsity=0.5 produces >= 50 % zeros in the prunable weight arrays
  - sparsity >= 1.0 raises ValueError
  - unsupported sklearn estimator (e.g. DecisionTreeClassifier) raises RuntimeError
  - works correctly for both LogisticRegression and MLPClassifier
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from tinyaudit.compress.prune import magnitude_prune
from tinyaudit.models.sklearn import SklearnModel

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def X_y():
    """Small deterministic binary dataset."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))
    y = (X[:, 0] + rng.standard_normal(120) > 0).astype(int)
    return X.astype(np.float64), y


@pytest.fixture()
def lr_model(X_y):
    X, y = X_y
    clf = LogisticRegression(max_iter=500, random_state=0).fit(X, y)
    return SklearnModel(clf)


@pytest.fixture()
def mlp_model(X_y):
    X, y = X_y
    clf = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500, random_state=0).fit(X, y)
    return SklearnModel(clf)


@pytest.fixture()
def dt_model(X_y):
    X, y = X_y
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    return SklearnModel(clf)


# --------------------------------------------------------------------------- #
# sparsity >= 1.0 raises ValueError
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_sparsity", [1.0, 1.5, 2.0, 100.0])
def test_sparsity_ge_one_raises_value_error(lr_model, bad_sparsity: float) -> None:
    with pytest.raises(ValueError, match="sparsity"):
        magnitude_prune(lr_model, bad_sparsity)


def test_negative_sparsity_raises_value_error(lr_model) -> None:
    with pytest.raises(ValueError, match="sparsity"):
        magnitude_prune(lr_model, -0.1)


# --------------------------------------------------------------------------- #
# sparsity=0.0 is a functional passthrough (deep copy, same outputs)
# --------------------------------------------------------------------------- #


def test_zero_sparsity_lr_identical_predictions(lr_model, X_y) -> None:
    X, _ = X_y
    pruned = magnitude_prune(lr_model, 0.0)
    np.testing.assert_array_equal(lr_model.predict(X), pruned.predict(X))
    np.testing.assert_allclose(lr_model.predict_proba(X), pruned.predict_proba(X), atol=0.0)


def test_zero_sparsity_mlp_identical_predictions(mlp_model, X_y) -> None:
    X, _ = X_y
    pruned = magnitude_prune(mlp_model, 0.0)
    np.testing.assert_array_equal(mlp_model.predict(X), pruned.predict(X))
    np.testing.assert_allclose(mlp_model.predict_proba(X), pruned.predict_proba(X), atol=0.0)


def test_zero_sparsity_does_not_mutate_original(lr_model) -> None:
    """The returned model is a *copy*; the original coef_ must be unchanged."""
    import copy

    original_coef = copy.deepcopy(lr_model.estimator.coef_)
    pruned = magnitude_prune(lr_model, 0.0)
    # Mutate the pruned model's coef_ and confirm the original is untouched.
    pruned.estimator.coef_[:] = 999.0
    np.testing.assert_array_equal(lr_model.estimator.coef_, original_coef)


# --------------------------------------------------------------------------- #
# sparsity=0.5 produces >= 50 % zeros
# --------------------------------------------------------------------------- #


def test_half_sparsity_lr_zeros(lr_model) -> None:
    pruned = magnitude_prune(lr_model, 0.5)
    coef = np.asarray(pruned.estimator.coef_)
    zero_frac = (coef == 0.0).mean()
    assert zero_frac >= 0.50, f"Expected >= 50 % zeros, got {zero_frac:.2%}"


def test_half_sparsity_mlp_zeros(mlp_model) -> None:
    pruned = magnitude_prune(mlp_model, 0.5)
    all_weights = np.concatenate([np.asarray(c).reshape(-1) for c in pruned.estimator.coefs_])
    zero_frac = (all_weights == 0.0).mean()
    assert zero_frac >= 0.50, f"Expected >= 50 % zeros, got {zero_frac:.2%}"


def test_half_sparsity_lr_still_predicts(lr_model, X_y) -> None:
    X, _ = X_y
    pruned = magnitude_prune(lr_model, 0.5)
    preds = pruned.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds.tolist()).issubset({0, 1})


def test_half_sparsity_mlp_still_predicts(mlp_model, X_y) -> None:
    X, _ = X_y
    pruned = magnitude_prune(mlp_model, 0.5)
    preds = pruned.predict(X)
    assert preds.shape == (len(X),)


# --------------------------------------------------------------------------- #
# Sparsity sweep spot-checks (30 %, 70 %, 90 %)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sparsity", [0.30, 0.70, 0.90])
def test_lr_sparsity_sweep(lr_model, sparsity: float) -> None:
    pruned = magnitude_prune(lr_model, sparsity)
    coef = np.asarray(pruned.estimator.coef_)
    zero_frac = (coef == 0.0).mean()
    assert (
        zero_frac >= sparsity
    ), f"sparsity={sparsity}: expected >= {sparsity:.0%} zeros, got {zero_frac:.2%}"


@pytest.mark.parametrize("sparsity", [0.30, 0.70, 0.90])
def test_mlp_sparsity_sweep(mlp_model, sparsity: float) -> None:
    pruned = magnitude_prune(mlp_model, sparsity)
    all_weights = np.concatenate([np.asarray(c).reshape(-1) for c in pruned.estimator.coefs_])
    zero_frac = (all_weights == 0.0).mean()
    assert (
        zero_frac >= sparsity
    ), f"sparsity={sparsity}: expected >= {sparsity:.0%} zeros, got {zero_frac:.2%}"


# --------------------------------------------------------------------------- #
# Unsupported sklearn family raises RuntimeError
# --------------------------------------------------------------------------- #


def test_unsupported_sklearn_family_raises_runtime_error(dt_model) -> None:
    with pytest.raises(RuntimeError, match="DecisionTreeClassifier"):
        magnitude_prune(dt_model, 0.3)


def test_unsupported_family_error_names_supported(dt_model) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        magnitude_prune(dt_model, 0.3)
    msg = str(exc_info.value)
    assert "LogisticRegression" in msg or "MLPClassifier" in msg


# --------------------------------------------------------------------------- #
# Protocol compliance: returned model satisfies AuditedModel
# --------------------------------------------------------------------------- #


def test_pruned_lr_satisfies_protocol(lr_model, X_y) -> None:
    from tinyaudit.models.base import AuditedModel

    X, _ = X_y
    pruned = magnitude_prune(lr_model, 0.5)
    assert isinstance(pruned, AuditedModel)
    assert pruned.framework == "sklearn"
    assert isinstance(pruned.n_params, int)
    assert pruned.n_params > 0


def test_pruned_mlp_satisfies_protocol(mlp_model, X_y) -> None:
    from tinyaudit.models.base import AuditedModel

    X, _ = X_y
    pruned = magnitude_prune(mlp_model, 0.5)
    assert isinstance(pruned, AuditedModel)
    assert pruned.framework == "sklearn"
    assert isinstance(pruned.n_params, int)


# --------------------------------------------------------------------------- #
# n_params is preserved (pruning zeroes weights, does not remove them)
# --------------------------------------------------------------------------- #


def test_lr_n_params_unchanged_after_prune(lr_model) -> None:
    pruned = magnitude_prune(lr_model, 0.5)
    assert pruned.n_params == lr_model.n_params


def test_mlp_n_params_unchanged_after_prune(mlp_model) -> None:
    pruned = magnitude_prune(mlp_model, 0.5)
    assert pruned.n_params == mlp_model.n_params
