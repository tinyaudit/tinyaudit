"""Tests for the scikit-learn adapter conforming to AuditedModel."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from tinyaudit.models import SklearnModel
from tinyaudit.models.base import AuditedModel


@pytest.fixture
def fitted_lr(binary_classification) -> tuple[SklearnModel, np.ndarray]:
    X, y, _sensitive = binary_classification
    est = LogisticRegression(max_iter=500).fit(X, y)
    return SklearnModel(est), X.to_numpy()


@pytest.fixture
def fitted_tree(binary_classification) -> tuple[SklearnModel, np.ndarray]:
    X, y, _sensitive = binary_classification
    est = DecisionTreeClassifier(random_state=0, max_depth=4).fit(X, y)
    return SklearnModel(est), X.to_numpy()


def test_satisfies_audited_model_protocol(fitted_lr) -> None:
    model, _X = fitted_lr
    assert isinstance(model, AuditedModel)


def test_framework_is_sklearn(fitted_lr, fitted_tree) -> None:
    assert fitted_lr[0].framework == "sklearn"
    assert fitted_tree[0].framework == "sklearn"


def test_logistic_regression_predict_shapes(fitted_lr) -> None:
    model, X = fitted_lr
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert isinstance(pred, np.ndarray)
    assert isinstance(proba, np.ndarray)
    assert pred.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_decision_tree_predict_shapes(fitted_tree) -> None:
    model, X = fitted_tree
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert pred.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], 2)


def test_predict_accepts_dataframe(binary_classification) -> None:
    X, y, _sensitive = binary_classification
    model = SklearnModel(LogisticRegression(max_iter=500).fit(X, y))
    pred_df = model.predict(X)
    pred_np = model.predict(X.to_numpy())
    assert isinstance(pred_df, np.ndarray)
    np.testing.assert_array_equal(pred_df, pred_np)
    assert isinstance(model.predict_proba(X), np.ndarray)


def test_logistic_regression_n_params(fitted_lr) -> None:
    model, _X = fitted_lr
    n = model.n_params
    assert isinstance(n, int)
    assert n > 0
    # Two features + intercept for binary LogisticRegression.
    est = model.estimator
    assert n == est.coef_.size + est.intercept_.size == 3


def test_decision_tree_n_params(fitted_tree) -> None:
    model, _X = fitted_tree
    n = model.n_params
    assert isinstance(n, int)
    assert n > 0
    assert n == model.estimator.tree_.node_count


def test_mlp_n_params(binary_classification) -> None:
    X, y, _sensitive = binary_classification
    est = MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=0)
    est.fit(X, y)
    model = SklearnModel(est)
    expected = sum(c.size for c in est.coefs_) + sum(b.size for b in est.intercepts_)
    assert model.n_params == expected
    assert model.n_params > 0


def test_generic_fallback_counts_known_attributes(binary_classification) -> None:
    X, y, _sensitive = binary_classification
    # GaussianNB exposes ``theta_``/``var_`` but none of the generic weight
    # attributes, so it must raise the informative TypeError.
    est = GaussianNB().fit(X, y)
    model = SklearnModel(est)
    with pytest.raises(TypeError, match="Cannot determine n_params"):
        _ = model.n_params


def test_generic_fallback_succeeds_for_linear_svc(binary_classification) -> None:
    from sklearn.svm import LinearSVC

    X, y, _sensitive = binary_classification
    est = LinearSVC().fit(X, y)
    model = SklearnModel(est)
    # LinearSVC has coef_ and intercept_ -> generic fallback sums them.
    n = model.n_params
    assert isinstance(n, int)
    assert n == est.coef_.size + est.intercept_.size > 0
