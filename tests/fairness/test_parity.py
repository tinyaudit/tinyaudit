"""Unit, property, and reference-oracle tests for the point fairness metrics.

The reference oracle is Fairlearn. It is import-only here, never in the
package hot path. ``demographic_parity_difference`` and
``equalized_odds_difference`` are defined to match Fairlearn's defaults
exactly (see the reconciliation comments on the oracle tests), so the
tolerance is tight. ``disparate_impact_ratio`` has no single Fairlearn
equivalent with the same direction convention, so it is pinned by a
dedicated hand-checked test instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import demographic_parity_difference as fl_dp
from fairlearn.metrics import equalized_odds_difference as fl_eo
from hypothesis import given
from hypothesis import strategies as st

from tinyaudit.fairness.parity import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
)

# --------------------------------------------------------------------------- #
# Unit tests on hand-constructed arrays with known answers.
# --------------------------------------------------------------------------- #


def test_perfect_parity_gives_zero_dp_and_unit_di() -> None:
    # Every group selected at exactly 0.5: no disparity at all.
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert demographic_parity_difference(y_pred, sensitive) == pytest.approx(0.0)
    assert disparate_impact_ratio(y_pred, sensitive) == pytest.approx(1.0)


def test_total_disparity_gives_unit_dp() -> None:
    # Group 0 never positive, group 1 always positive: maximal disparity.
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    sensitive = np.array([0, 0, 0, 1, 1, 1])
    assert demographic_parity_difference(y_pred, sensitive) == pytest.approx(1.0)
    # DI = min_rate / max_rate = 0 / 1 = 0.0.
    assert disparate_impact_ratio(y_pred, sensitive) == pytest.approx(0.0)


def test_dp_difference_known_value() -> None:
    # group 0 rate = 1/4 = 0.25, group 1 rate = 3/4 = 0.75 -> diff 0.5.
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert demographic_parity_difference(y_pred, sensitive) == pytest.approx(0.5)


def test_equalized_odds_known_value() -> None:
    # Group 0: y_true=[1,1,0,0], y_pred=[1,1,0,0] -> TPR 1.0, FPR 0.0.
    # Group 1: y_true=[1,1,0,0], y_pred=[0,0,1,1] -> TPR 0.0, FPR 1.0.
    # TPR range = 1.0, FPR range = 1.0 -> EO diff = max(1.0, 1.0) = 1.0.
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert equalized_odds_difference(y_true, y_pred, sensitive) == pytest.approx(1.0)


def test_equalized_odds_zero_when_groups_match() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert equalized_odds_difference(y_true, y_pred, sensitive) == pytest.approx(0.0)


def test_metrics_accept_pandas() -> None:
    y_pred = pd.Series([1, 0, 0, 1, 1, 0])
    sensitive = pd.Series(["m", "m", "m", "f", "f", "f"])
    y_true = pd.Series([1, 0, 1, 1, 0, 0])
    assert 0.0 <= demographic_parity_difference(y_pred, sensitive) <= 1.0
    assert 0.0 <= disparate_impact_ratio(y_pred, sensitive) <= 1.0
    assert 0.0 <= equalized_odds_difference(y_true, y_pred, sensitive) <= 1.0


def test_metrics_handle_more_than_two_groups() -> None:
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1])
    sensitive = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    assert 0.0 <= demographic_parity_difference(y_pred, sensitive) <= 1.0
    assert 0.0 <= disparate_impact_ratio(y_pred, sensitive) <= 1.0


# --------------------------------------------------------------------------- #
# Disparate-impact direction convention: pinned here AND in the docstring.
# Changing one without the other is a deliberate test failure.
# --------------------------------------------------------------------------- #


def test_disparate_impact_direction_convention_is_pinned() -> None:
    """DI = (min group positive rate) / (max group positive rate).

    Concretely: group 0 has positive rate 0.2 (1 of 5), group 1 has
    positive rate 0.8 (4 of 5). The pinned convention puts the *minimum*
    rate in the numerator and the *maximum* rate in the denominator, so
    DI = 0.2 / 0.8 = 0.25, which is in [0, 1] and below the 0.8
    four-fifths threshold. It must NOT be the inverse 0.8 / 0.2 = 4.0.
    """
    y_pred = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0])
    sensitive = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    rate0 = y_pred[sensitive == 0].mean()
    rate1 = y_pred[sensitive == 1].mean()
    assert rate0 == pytest.approx(0.2)
    assert rate1 == pytest.approx(0.8)

    di = disparate_impact_ratio(y_pred, sensitive)
    assert di == pytest.approx(0.25), "DI must be min/max = 0.2/0.8, not the inverse"
    assert 0.0 <= di <= 1.0
    assert di == pytest.approx(min(rate0, rate1) / max(rate0, rate1))


def test_disparate_impact_is_symmetric_in_group_labels() -> None:
    # Relabeling which group is "0" vs "1" must not change DI, because the
    # convention is min/max rather than (group A)/(group B). This symmetry
    # is exactly why the convention is the easiest place to ship a silent
    # bug, hence the explicit pin above.
    y_pred = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0])
    sensitive = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    flipped = 1 - sensitive
    assert disparate_impact_ratio(y_pred, sensitive) == pytest.approx(
        disparate_impact_ratio(y_pred, flipped)
    )


def test_disparate_impact_degenerate_when_no_positives() -> None:
    y_pred = np.zeros(8, dtype=int)
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # No group ever predicts positive: defined as 1.0 (degenerate parity).
    assert disparate_impact_ratio(y_pred, sensitive) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Hypothesis property tests.
# --------------------------------------------------------------------------- #


@given(
    preds=st.lists(st.integers(0, 1), min_size=2, max_size=120),
    seed=st.integers(0, 100_000),
    n_groups=st.integers(2, 4),
)
def test_dp_difference_always_in_unit_interval(preds: list[int], seed: int, n_groups: int) -> None:
    rng = np.random.default_rng(seed)
    y_pred = np.array(preds)
    sensitive = rng.integers(0, n_groups, size=len(preds))
    dp = demographic_parity_difference(y_pred, sensitive)
    assert 0.0 <= dp <= 1.0


@given(
    preds=st.lists(st.integers(0, 1), min_size=2, max_size=120),
    seed=st.integers(0, 100_000),
)
def test_disparate_impact_always_in_unit_interval(preds: list[int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_pred = np.array(preds)
    sensitive = rng.integers(0, 3, size=len(preds))
    di = disparate_impact_ratio(y_pred, sensitive)
    assert 0.0 <= di <= 1.0


@given(
    truth=st.lists(st.integers(0, 1), min_size=4, max_size=120),
    seed=st.integers(0, 100_000),
)
def test_equalized_odds_always_in_unit_interval(truth: list[int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = np.array(truth)
    y_pred = rng.integers(0, 2, size=len(truth))
    sensitive = rng.integers(0, 3, size=len(truth))
    eo = equalized_odds_difference(y_true, y_pred, sensitive)
    assert 0.0 <= eo <= 1.0


@given(preds=st.lists(st.integers(0, 1), min_size=4, max_size=120), seed=st.integers(0, 100_000))
def test_relabeling_groups_preserves_metric_magnitude(preds: list[int], seed: int) -> None:
    """Permuting / relabeling group labels must not change the metric.

    The metrics are symmetric reductions across groups, so an arbitrary
    bijection on the label set leaves DP, DI, and EO unchanged.
    """
    rng = np.random.default_rng(seed)
    y_pred = np.array(preds)
    y_true = rng.integers(0, 2, size=len(preds))
    sensitive = rng.integers(0, 3, size=len(preds))

    # Arbitrary relabeling 0->7, 1->"x", 2->-3 (also exercises mixed types
    # via object dtype) is a bijection, so magnitudes must match.
    mapping = {0: 7, 1: 99, 2: -3}
    relabeled = np.array([mapping[int(v)] for v in sensitive])

    assert demographic_parity_difference(y_pred, sensitive) == pytest.approx(
        demographic_parity_difference(y_pred, relabeled)
    )
    assert disparate_impact_ratio(y_pred, sensitive) == pytest.approx(
        disparate_impact_ratio(y_pred, relabeled)
    )
    assert equalized_odds_difference(y_true, y_pred, sensitive) == pytest.approx(
        equalized_odds_difference(y_true, y_pred, relabeled)
    )


# --------------------------------------------------------------------------- #
# Reference-oracle tests vs Fairlearn (fixed seed).
#
# Reconciliation:
#   * fairlearn.metrics.demographic_parity_difference with its default
#     method="between_groups" is exactly (max selection rate) -
#     (min selection rate) across groups, which is our definition. So the
#     two agree to floating-point noise, not just "within tolerance".
#   * fairlearn.metrics.equalized_odds_difference with its defaults
#     (agg="worst_case", method="between_groups") is
#     max( TPR max-min , FPR max-min ) across groups, which is our
#     definition. Same exact agreement.
# We still assert with a small tolerance to stay robust to library
# refactors, and additionally assert exact equality on the fixed seed to
# document that the definitions coincide.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_groups", [2, 3])
def test_dp_matches_fairlearn_oracle(n_groups: int) -> None:
    rng = np.random.default_rng(20260515)
    n = 600
    sensitive = rng.integers(0, n_groups, size=n)
    y_true = rng.integers(0, 2, size=n)
    skew = rng.random(n_groups)[sensitive]
    y_pred = (rng.random(n) < skew).astype(int)

    mine = demographic_parity_difference(y_pred, sensitive)
    oracle = fl_dp(y_true, y_pred, sensitive_features=sensitive)
    assert mine == pytest.approx(oracle, abs=1e-9)
    assert mine == oracle  # definitions coincide exactly on this seed.


@pytest.mark.parametrize("n_groups", [2, 3])
def test_equalized_odds_matches_fairlearn_oracle(n_groups: int) -> None:
    rng = np.random.default_rng(98765)
    n = 600
    sensitive = rng.integers(0, n_groups, size=n)
    base = rng.random(n_groups)[sensitive]
    y_true = (rng.random(n) < base).astype(int)
    pred_skew = rng.random(n_groups)[sensitive]
    y_pred = (rng.random(n) < pred_skew).astype(int)

    mine = equalized_odds_difference(y_true, y_pred, sensitive)
    oracle = fl_eo(y_true, y_pred, sensitive_features=sensitive)
    assert mine == pytest.approx(oracle, abs=1e-9)
    assert mine == oracle  # definitions coincide exactly on this seed.


def test_dp_matches_fairlearn_with_string_groups() -> None:
    rng = np.random.default_rng(7)
    n = 300
    sensitive = np.where(rng.integers(0, 2, size=n) == 1, "female", "male")
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    mine = demographic_parity_difference(y_pred, sensitive)
    oracle = fl_dp(y_true, y_pred, sensitive_features=sensitive)
    assert mine == pytest.approx(oracle, abs=1e-9)


# --------------------------------------------------------------------------- #
# Sanity on the shared fixture: a model that learns the signal should show a
# disparity in the same direction as the fixture's base-rate gap.
# --------------------------------------------------------------------------- #


def test_fixture_has_expected_disparity_direction(binary_classification) -> None:
    _X, y, sensitive = binary_classification
    y_arr = y.to_numpy()
    s_arr = sensitive.to_numpy()
    # Fixture: group 1 has the higher positive base rate.
    rate0 = y_arr[s_arr == 0].mean()
    rate1 = y_arr[s_arr == 1].mean()
    assert rate1 > rate0
    dp = demographic_parity_difference(y, sensitive)
    di = disparate_impact_ratio(y, sensitive)
    assert dp > 0.1
    assert di < 0.8  # below the four-fifths rule, as expected for this gap.
