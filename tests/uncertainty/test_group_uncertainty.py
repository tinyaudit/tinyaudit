"""Tests for the group-level *uncertainty* metrics.

These separate the uncertainty signal (predictive variance, mutual information)
from the calibration signal (ECE). They mirror ``group_predictive_entropy``:
a per-group mean over a per-sample array, empty groups skipped, nan-safe. The
``disparity`` helper reduces any per-group dict to a single max-min gap and is
the reduction the pipeline reports beside ``ece_disparity``.

Contracts covered per function: basic output, multi-valued (non-binary) groups,
empty/single-sample groups, nan propagation, and the disparity edge cases
(single group -> 0.0, no finite values -> nan). A Hypothesis property test
pins the disparity invariants over arbitrary group counts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tinyaudit.uncertainty.metrics import (
    disparity,
    group_mutual_information,
    group_predictive_variance,
)
from tinyaudit.uncertainty.types import UncertaintyOutput


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_output(rng: np.random.Generator, n: int = 100, n_classes: int = 2) -> UncertaintyOutput:
    """Build a deterministic UncertaintyOutput for testing."""
    proba = rng.dirichlet(np.ones(n_classes), size=n).astype(np.float64)
    entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
    variance = np.var(proba, axis=1)
    mi = entropy * rng.random(n) * 0.5
    return UncertaintyOutput(
        mean_proba=proba,
        predictive_entropy=entropy,
        predictive_variance=variance,
        mutual_information=mi,
    )


# --------------------------------------------------------------------------- #
# group_predictive_variance
# --------------------------------------------------------------------------- #


class TestGroupPredictiveVariance:
    def test_returns_dict_keyed_by_group(self) -> None:
        rng = np.random.default_rng(1)
        out = _make_output(rng, n=80)
        sensitive = rng.integers(0, 2, size=80).astype(np.int64)
        result = group_predictive_variance(out, sensitive)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"0", "1"}

    def test_matches_manual_group_mean(self) -> None:
        rng = np.random.default_rng(2)
        out = _make_output(rng, n=120)
        sensitive = rng.integers(0, 3, size=120).astype(np.int64)  # 3 groups
        result = group_predictive_variance(out, sensitive)
        for g in np.unique(sensitive):
            expected = float(np.mean(out.predictive_variance[sensitive == g]))
            assert result[str(g)] == pytest.approx(expected)

    def test_multi_valued_string_groups(self) -> None:
        rng = np.random.default_rng(3)
        out = _make_output(rng, n=60)
        sensitive = np.array((["a"] * 20) + (["b"] * 20) + (["c"] * 20), dtype=object)
        result = group_predictive_variance(out, sensitive)
        assert set(result.keys()) == {"a", "b", "c"}

    def test_single_sample_group_is_that_sample(self) -> None:
        rng = np.random.default_rng(4)
        out = _make_output(rng, n=10)
        sensitive = np.array([0] * 9 + [1], dtype=np.int64)  # group 1 has one sample
        result = group_predictive_variance(out, sensitive)
        assert result["1"] == pytest.approx(float(out.predictive_variance[-1]))

    def test_nan_propagates(self) -> None:
        rng = np.random.default_rng(5)
        out = _make_output(rng, n=20)
        nan_mask = np.arange(20) < 10
        out.predictive_variance[nan_mask] = np.nan
        sensitive = (~nan_mask).astype(np.int64)  # group 0 == the nan rows
        result = group_predictive_variance(out, sensitive)
        assert math.isnan(result["0"])
        assert not math.isnan(result["1"])


# --------------------------------------------------------------------------- #
# group_mutual_information
# --------------------------------------------------------------------------- #


class TestGroupMutualInformation:
    def test_returns_dict_keyed_by_group(self) -> None:
        rng = np.random.default_rng(6)
        out = _make_output(rng, n=80)
        sensitive = rng.integers(0, 2, size=80).astype(np.int64)
        result = group_mutual_information(out, sensitive)
        assert set(result.keys()) == {"0", "1"}

    def test_matches_manual_group_mean(self) -> None:
        rng = np.random.default_rng(7)
        out = _make_output(rng, n=90)
        sensitive = rng.integers(0, 2, size=90).astype(np.int64)
        result = group_mutual_information(out, sensitive)
        for g in np.unique(sensitive):
            expected = float(np.mean(out.mutual_information[sensitive == g]))
            assert result[str(g)] == pytest.approx(expected)

    def test_values_non_negative(self) -> None:
        rng = np.random.default_rng(8)
        out = _make_output(rng, n=150)
        sensitive = rng.integers(0, 2, size=150).astype(np.int64)
        result = group_mutual_information(out, sensitive)
        assert all(v >= 0.0 for v in result.values())


# --------------------------------------------------------------------------- #
# disparity
# --------------------------------------------------------------------------- #


class TestDisparity:
    def test_max_minus_min(self) -> None:
        assert disparity({"a": 0.1, "b": 0.5, "c": 0.3}) == pytest.approx(0.4)

    def test_single_group_is_zero_not_nan(self) -> None:
        # One group has no gap to any other, so the disparity is 0.0, not nan.
        assert disparity({"a": 0.7}) == 0.0

    def test_empty_is_nan(self) -> None:
        assert math.isnan(disparity({}))

    def test_all_nan_is_nan(self) -> None:
        assert math.isnan(disparity({"a": float("nan"), "b": float("nan")}))

    def test_ignores_nan_values(self) -> None:
        # Only the two finite values contribute to the gap.
        assert disparity({"a": 0.2, "b": float("nan"), "c": 0.9}) == pytest.approx(0.7)

    def test_one_finite_among_nan_is_zero(self) -> None:
        assert disparity({"a": float("nan"), "b": 0.4}) == 0.0

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=3),
            values=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
            min_size=2,
            max_size=8,
        )
    )
    def test_property_nonnegative_and_bounded(self, groups: dict[str, float]) -> None:
        d = disparity(groups)
        vals = list(groups.values())
        assert d == pytest.approx(max(vals) - min(vals))
        assert d >= 0.0
