"""The constant-predictor baseline: protocol conformance and the parity claim.

The load-bearing test here is
:func:`test_constant_model_is_perfectly_fair_on_any_grouping`. The paper's
argument is that group-parity metrics are satisfiable by a model that carries
no information, so a constant predictor must score exactly 0.0 on demographic
parity difference and exactly 1.0 on disparate impact for *any* sensitive
attribute, binary or multi-valued. If that ever stops holding, either the
metrics or the baseline is wrong and the claim needs re-checking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.metrics import balanced_accuracy_score

from tinyaudit.fairness.parity import demographic_parity_difference, disparate_impact_ratio
from tinyaudit.models import ConstantModel
from tinyaudit.models.base import AuditedModel

# Labels whose majority class is 1 (7 ones, 3 zeros) so the selection rate is
# a non-degenerate 1.0 everywhere and DI is a real 1.0/1.0, not the 0/0 case.
_MAJORITY_ONE = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
# Labels whose majority class is 0 (Adult-like 76/24 skew).
_MAJORITY_ZERO = np.array([0] * 76 + [1] * 24)


@pytest.fixture
def X() -> np.ndarray:
    """Feature matrix the model must ignore entirely."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(10, 3))


# --------------------------------------------------------------------------- #
# Protocol conformance and basic shapes.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["majority", "prevalence"])
def test_satisfies_audited_model_protocol(mode: str) -> None:
    model = ConstantModel.from_labels(_MAJORITY_ONE, mode=mode)  # type: ignore[arg-type]
    assert isinstance(model, AuditedModel)


def test_framework_and_n_params() -> None:
    model = ConstantModel.from_labels(_MAJORITY_ONE)
    assert model.framework == "constant"
    assert model.n_params == 1
    assert isinstance(model.n_params, int)


@pytest.mark.parametrize("mode", ["majority", "prevalence"])
def test_predict_proba_shape_and_rows_sum_to_one(X: np.ndarray, mode: str) -> None:
    model = ConstantModel.from_labels(_MAJORITY_ZERO, mode=mode)  # type: ignore[arg-type]
    proba = model.predict_proba(X)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    # Every row identical: the input is genuinely ignored.
    np.testing.assert_array_equal(proba, np.tile(proba[0], (X.shape[0], 1)))


def test_predict_shape_and_constant_output(X: np.ndarray) -> None:
    model = ConstantModel.from_labels(_MAJORITY_ONE)
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (X.shape[0],)
    assert len(np.unique(pred)) == 1


def test_predict_accepts_dataframe_and_ignores_content() -> None:
    model = ConstantModel.from_labels(_MAJORITY_ONE)
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [-9.0, 0.0, 9.0]})
    np.testing.assert_array_equal(model.predict(frame), np.array([1, 1, 1]))
    assert model.predict_proba(frame).shape == (3, 2)
    # A completely different feature matrix of the same length -> same answer.
    other = pd.DataFrame({"a": [100.0, 200.0, 300.0], "b": [0.0, 0.0, 0.0]})
    np.testing.assert_array_equal(model.predict(frame), model.predict(other))


def test_predict_returns_original_labels_not_indices(X: np.ndarray) -> None:
    """Downstream metrics compare against ``y_true``, so labels must survive."""
    y = np.array([5, 5, 5, 7, 7], dtype=np.int64)
    model = ConstantModel.from_labels(y)
    pred = model.predict(X)
    # Majority label is 5, not its column index 0.
    assert set(np.unique(pred)) == {5}
    assert pred.dtype == y.dtype

    # String labels round-trip too.
    s_model = ConstantModel.from_labels(np.array(["no", "no", "no", "yes"]))
    assert set(np.unique(s_model.predict(X))) == {"no"}


def test_classes_are_sorted_like_sklearn() -> None:
    """Column order of ``predict_proba`` follows sorted ``classes_``."""
    model = ConstantModel.from_labels(np.array([2, 2, 2, 0, 0, 1]))
    np.testing.assert_array_equal(model.classes_, np.array([0, 1, 2]))
    # Prevalence is reordered with the classes: 2/6, 1/6, 3/6.
    np.testing.assert_allclose(model.prevalence, np.array([2 / 6, 1 / 6, 3 / 6]))
    # Even when constructed out of order.
    manual = ConstantModel([2, 0, 1], [0.5, 2 / 6, 1 / 6], mode="prevalence")
    np.testing.assert_array_equal(manual.classes_, np.array([0, 1, 2]))
    np.testing.assert_allclose(manual.predict_proba(np.zeros((1, 1)))[0], model.prevalence)


# --------------------------------------------------------------------------- #
# THE LOAD-BEARING PROPERTY: a useless model is perfectly "fair".
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["majority", "prevalence"])
@pytest.mark.parametrize("labels", [_MAJORITY_ONE, _MAJORITY_ZERO], ids=["pos_major", "neg_major"])
def test_constant_model_is_perfectly_fair_on_any_grouping(mode: str, labels: np.ndarray) -> None:
    """DP diff is exactly 0.0 and DI is exactly 1.0, for every grouping.

    Includes a four-valued sensitive attribute with wildly unequal group
    sizes, plus a degenerate single-group case: the selection rate is
    identical in every group because the prediction does not depend on the
    row at all.
    """
    model = ConstantModel.from_labels(labels, mode=mode)  # type: ignore[arg-type]
    n = 200
    rng = np.random.default_rng(20260816)
    X_any = rng.normal(size=(n, 4))
    y_pred = model.predict(X_any)

    groupings: list[np.ndarray] = [
        # Binary, balanced.
        np.tile([0, 1], n // 2),
        # Multi-valued (4 groups), string-labelled, wildly unbalanced:
        # 100 / 60 / 39 / 1.
        np.array(["a"] * 100 + ["b"] * 60 + ["c"] * 39 + ["d"] * 1),
        # Random assignment over 3 groups.
        rng.integers(0, 3, size=n),
        # Degenerate: one group only.
        np.zeros(n, dtype=int),
    ]

    for sensitive in groupings:
        assert demographic_parity_difference(y_pred, sensitive) == 0.0
        assert disparate_impact_ratio(y_pred, sensitive) == 1.0


@given(
    seed=st.integers(0, 100_000),
    n_groups=st.integers(1, 6),
    n=st.integers(2, 200),
    positive_major=st.booleans(),
)
def test_perfect_parity_holds_for_arbitrary_sensitive_arrays(
    seed: int, n_groups: int, n: int, positive_major: bool
) -> None:
    """The parity claim is a property, not a fixture artefact."""
    rng = np.random.default_rng(seed)
    labels = _MAJORITY_ONE if positive_major else _MAJORITY_ZERO
    model = ConstantModel.from_labels(labels)
    y_pred = model.predict(rng.normal(size=(n, 2)))
    sensitive = rng.integers(0, n_groups, size=n)

    assert demographic_parity_difference(y_pred, sensitive) == 0.0
    assert disparate_impact_ratio(y_pred, sensitive) == 1.0


def test_perfectly_fair_but_useless() -> None:
    """The other half of the argument: parity is perfect, skill is nil."""
    rng = np.random.default_rng(1)
    n = 400
    y_true = (rng.random(n) < 0.3).astype(int)
    model = ConstantModel.from_labels(y_true)
    y_pred = model.predict(rng.normal(size=(n, 2)))
    sensitive = rng.integers(0, 2, size=n)

    assert demographic_parity_difference(y_pred, sensitive) == 0.0
    assert disparate_impact_ratio(y_pred, sensitive) == 1.0
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(0.5)


def test_balanced_accuracy_is_one_half_on_two_class_problem() -> None:
    """One class always predicted -> TPR + TNR = 1, so balanced accuracy 0.5."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    model = ConstantModel.from_labels(y_true, mode="majority")
    y_pred = model.predict(np.zeros((y_true.shape[0], 1)))
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Mode semantics.
# --------------------------------------------------------------------------- #


def test_majority_mode_is_hard_one_hot(X: np.ndarray) -> None:
    model = ConstantModel.from_labels(_MAJORITY_ZERO, mode="majority")
    proba = model.predict_proba(X)
    # Majority class is 0 -> column 0 gets all the mass.
    np.testing.assert_array_equal(proba, np.tile([1.0, 0.0], (X.shape[0], 1)))
    assert set(np.unique(model.predict(X))) == {0}


def test_prevalence_mode_reproduces_training_base_rate(X: np.ndarray) -> None:
    """Adult-like 76/24 skew comes back verbatim in every row."""
    model = ConstantModel.from_labels(_MAJORITY_ZERO, mode="prevalence")
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba, np.tile([0.76, 0.24], (X.shape[0], 1)))
    np.testing.assert_allclose(model.prevalence, np.array([0.76, 0.24]))


def test_prevalence_mode_predict_is_argmax_of_proba(X: np.ndarray) -> None:
    for labels in (_MAJORITY_ZERO, _MAJORITY_ONE):
        model = ConstantModel.from_labels(labels, mode="prevalence")
        proba = model.predict_proba(X)
        expected = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(model.predict(X), expected)


def test_both_modes_agree_on_hard_predictions(X: np.ndarray) -> None:
    maj = ConstantModel.from_labels(_MAJORITY_ZERO, mode="majority")
    prev = ConstantModel.from_labels(_MAJORITY_ZERO, mode="prevalence")
    np.testing.assert_array_equal(maj.predict(X), prev.predict(X))
    # ...but not on the probabilities.
    assert not np.allclose(maj.predict_proba(X), prev.predict_proba(X))


def test_prevalence_mode_handles_multiclass(X: np.ndarray) -> None:
    y = np.array([0] * 5 + [1] * 3 + [2] * 2)
    model = ConstantModel.from_labels(y, mode="prevalence")
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    np.testing.assert_allclose(proba[0], np.array([0.5, 0.3, 0.2]))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    assert set(np.unique(model.predict(X))) == {0}


# --------------------------------------------------------------------------- #
# Determinism and validation.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["majority", "prevalence"])
def test_two_models_from_same_labels_are_identical(X: np.ndarray, mode: str) -> None:
    a = ConstantModel.from_labels(_MAJORITY_ZERO, mode=mode)  # type: ignore[arg-type]
    b = ConstantModel.from_labels(_MAJORITY_ZERO, mode=mode)  # type: ignore[arg-type]
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    np.testing.assert_array_equal(a.classes_, b.classes_)
    np.testing.assert_array_equal(a.prevalence, b.prevalence)
    # Repeated calls on the same instance are stable too.
    np.testing.assert_array_equal(a.predict_proba(X), a.predict_proba(X))


def test_label_order_does_not_change_the_model(X: np.ndarray) -> None:
    shuffled = np.random.default_rng(3).permutation(_MAJORITY_ZERO)
    a = ConstantModel.from_labels(_MAJORITY_ZERO, mode="prevalence")
    b = ConstantModel.from_labels(shuffled, mode="prevalence")
    np.testing.assert_array_equal(a.classes_, b.classes_)
    np.testing.assert_allclose(a.prevalence, b.prevalence)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_from_labels_accepts_pandas_series(X: np.ndarray) -> None:
    series = pd.Series(_MAJORITY_ZERO, name="label")
    model = ConstantModel.from_labels(series, mode="prevalence")
    np.testing.assert_allclose(model.prevalence, np.array([0.76, 0.24]))


def test_accessors_return_copies() -> None:
    model = ConstantModel.from_labels(_MAJORITY_ZERO, mode="prevalence")
    prev = model.prevalence
    prev[0] = 99.0
    np.testing.assert_allclose(model.prevalence, np.array([0.76, 0.24]))
    classes = model.classes_
    classes[0] = 42
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        ConstantModel.from_labels(_MAJORITY_ZERO, mode="mean")  # type: ignore[arg-type]


def test_empty_labels_raise() -> None:
    with pytest.raises(ValueError, match="at least one label"):
        ConstantModel.from_labels(np.array([], dtype=int))


def test_two_dimensional_labels_raise() -> None:
    with pytest.raises(ValueError, match="y must be 1-D"):
        ConstantModel.from_labels(np.zeros((4, 2), dtype=int))


def test_prevalence_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ConstantModel([0, 1], [0.5, 0.2])


def test_prevalence_length_must_match_classes() -> None:
    with pytest.raises(ValueError, match="same length as classes"):
        ConstantModel([0, 1, 2], [0.5, 0.5])


def test_duplicate_classes_raise() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        ConstantModel([1, 1], [0.5, 0.5])


def test_negative_prevalence_raises() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        ConstantModel([0, 1], [-0.5, 1.5])


# --------------------------------------------------------------------------- #
# Pipeline integration: duck-typed pass-through, no SklearnModel wrapper.
# --------------------------------------------------------------------------- #


def test_pipeline_coerce_model_passes_it_through_unwrapped() -> None:
    from tinyaudit.models import SklearnModel
    from tinyaudit.pipeline import _coerce_model

    model = ConstantModel.from_labels(_MAJORITY_ZERO)
    coerced = _coerce_model(model)
    assert coerced is model
    assert not isinstance(coerced, SklearnModel)
    assert coerced.framework == "constant"
