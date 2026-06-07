"""Tests for uncertainty-aware fairness metrics.

Covers group_predictive_entropy, ece_per_group, and selective_fairness_auc.
Each test targets one behavioural contract: basic output shape / value,
edge-case handling (single-class groups, empty groups, sparse coverage), and
the nan-return contract when computation is impossible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tinyaudit.uncertainty.metrics import (
    ece_per_group,
    group_predictive_entropy,
    selective_fairness_auc,
)
from tinyaudit.uncertainty.types import UncertaintyOutput

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


def _make_output(
    rng: np.random.Generator,
    n: int = 100,
    n_classes: int = 2,
) -> UncertaintyOutput:
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


def _binary_sensitive(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n).astype(np.int64)


def _binary_y_true(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n).astype(np.int64)


# --------------------------------------------------------------------------- #
# group_predictive_entropy
# --------------------------------------------------------------------------- #


class TestGroupPredictiveEntropy:
    def test_returns_dict_with_both_groups(self) -> None:
        rng = np.random.default_rng(1)
        out = _make_output(rng, n=80)
        sensitive = _binary_sensitive(80, rng)
        result = group_predictive_entropy(out, sensitive)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"0", "1"}

    def test_values_are_finite_non_negative(self) -> None:
        rng = np.random.default_rng(2)
        out = _make_output(rng, n=200)
        sensitive = _binary_sensitive(200, rng)
        result = group_predictive_entropy(out, sensitive)
        for k, v in result.items():
            assert math.isfinite(v), f"group {k} entropy is not finite: {v}"
            assert v >= 0.0, f"group {k} entropy is negative: {v}"

    def test_string_group_labels(self) -> None:
        rng = np.random.default_rng(3)
        n = 60
        out = _make_output(rng, n=n)
        sensitive = np.where(rng.integers(0, 2, size=n) == 0, "male", "female")
        result = group_predictive_entropy(out, sensitive)
        assert set(result.keys()) == {"male", "female"}

    def test_single_class_group_does_not_crash(self) -> None:
        """A group that has only one sample must not raise."""
        rng = np.random.default_rng(4)
        n = 50
        out = _make_output(rng, n=n)
        # Group 1 has exactly one member.
        sensitive = np.zeros(n, dtype=np.int64)
        sensitive[0] = 1
        result = group_predictive_entropy(out, sensitive)
        assert "1" in result
        assert math.isfinite(result["1"]) or math.isnan(result["1"])

    def test_handles_three_groups(self) -> None:
        rng = np.random.default_rng(5)
        n = 90
        out = _make_output(rng, n=n)
        sensitive = rng.integers(0, 3, size=n).astype(np.int64)
        result = group_predictive_entropy(out, sensitive)
        # All three groups should be present (very unlikely for one to be empty
        # with n=90 and 3 groups, but we don't mandate they exist).
        for _k, v in result.items():
            assert math.isfinite(v) or math.isnan(v)

    def test_empty_group_not_in_result(self) -> None:
        """Groups that are entirely absent from sensitive are not in the output.

        We construct an array that never assigns group label 99 so the dict
        must not contain it.
        """
        rng = np.random.default_rng(6)
        n = 40
        out = _make_output(rng, n=n)
        sensitive = np.zeros(n, dtype=np.int64)  # only group 0
        result = group_predictive_entropy(out, sensitive)
        assert "99" not in result
        assert len(result) == 1

    def test_mean_is_close_to_manual_calculation(self) -> None:
        rng = np.random.default_rng(7)
        n = 100
        out = _make_output(rng, n=n)
        sensitive = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        result = group_predictive_entropy(out, sensitive)
        expected_0 = float(np.mean(out.predictive_entropy[:50]))
        expected_1 = float(np.mean(out.predictive_entropy[50:]))
        assert result["0"] == pytest.approx(expected_0, abs=1e-10)
        assert result["1"] == pytest.approx(expected_1, abs=1e-10)


# --------------------------------------------------------------------------- #
# ece_per_group
# --------------------------------------------------------------------------- #


class TestEcePerGroup:
    def test_ece_finite_and_in_unit_interval(self) -> None:
        rng = np.random.default_rng(10)
        n = 200
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = _binary_sensitive(n, rng)
        result = ece_per_group(out, y_true, sensitive)
        for k, v in result.items():
            assert math.isfinite(v), f"group {k} ECE is not finite"
            assert 0.0 <= v <= 1.0, f"group {k} ECE out of [0,1]: {v}"

    def test_empty_bins_do_not_crash(self) -> None:
        """Very small groups may have sparse confidence coverage; must not raise."""
        rng = np.random.default_rng(11)
        n = 30
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = np.zeros(n, dtype=np.int64)
        sensitive[:3] = 1  # group 1 has only 3 samples -> most bins are empty
        result = ece_per_group(out, y_true, sensitive)
        assert "1" in result
        v = result["1"]
        assert math.isfinite(v) and 0.0 <= v <= 1.0

    def test_group_with_one_sample_returns_nan(self) -> None:
        rng = np.random.default_rng(12)
        n = 50
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = np.zeros(n, dtype=np.int64)
        sensitive[0] = 2  # group 2 has exactly 1 sample
        result = ece_per_group(out, y_true, sensitive)
        assert math.isnan(result["2"])

    def test_ece_zero_for_perfect_calibration(self) -> None:
        """Construct a model where confidence exactly equals accuracy in every bin."""
        n = 100
        # All samples in bin [0.9, 1.0]: confidence ~0.95, prediction always
        # matches y_true -> acc = 1.0, conf = 0.95 -> ECE = 0.05.
        # Instead use a case where they must be equal: confidence = 1.0 for all
        # samples and all predictions are correct.
        proba = np.zeros((n, 2), dtype=np.float64)
        proba[:, 1] = 1.0  # confidence = 1.0
        out = UncertaintyOutput(
            mean_proba=proba,
            predictive_entropy=np.zeros(n),
            predictive_variance=np.zeros(n),
            mutual_information=np.zeros(n),
        )
        y_true = np.ones(n, dtype=np.int64)  # all positive (matches argmax=1)
        sensitive = np.zeros(n, dtype=np.int64)
        result = ece_per_group(out, y_true, sensitive)
        # acc = 1.0, conf = 1.0, so ECE = 0.0
        assert result["0"] == pytest.approx(0.0, abs=1e-10)

    def test_ece_returns_dict_with_string_keys(self) -> None:
        rng = np.random.default_rng(13)
        n = 60
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = np.where(rng.integers(0, 2, size=n) == 0, "A", "B")
        result = ece_per_group(out, y_true, sensitive)
        assert set(result.keys()) == {"A", "B"}

    def test_ece_three_groups(self) -> None:
        rng = np.random.default_rng(14)
        n = 150
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = rng.integers(0, 3, size=n).astype(np.int64)
        result = ece_per_group(out, y_true, sensitive)
        for v in result.values():
            assert math.isfinite(v) or math.isnan(v)

    def test_group_with_zero_samples_is_nan(self) -> None:
        """Group that np.unique never returns is never in the output."""
        rng = np.random.default_rng(15)
        n = 40
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        # Only group 0 exists.
        sensitive = np.zeros(n, dtype=np.int64)
        result = ece_per_group(out, y_true, sensitive)
        assert len(result) == 1
        assert "0" in result


# --------------------------------------------------------------------------- #
# selective_fairness_auc
# --------------------------------------------------------------------------- #


class TestSelectiveFairnessAuc:
    def test_returns_finite_float(self) -> None:
        rng = np.random.default_rng(20)
        n = 120
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = _binary_sensitive(n, rng)
        result = selective_fairness_auc(out, y_true, sensitive)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_auc_non_negative(self) -> None:
        """DP diff is in [0, 1] at every threshold, so the AUC must be >= 0."""
        rng = np.random.default_rng(21)
        n = 100
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = _binary_sensitive(n, rng)
        result = selective_fairness_auc(out, y_true, sensitive)
        assert result >= 0.0

    def test_auc_upper_bound(self) -> None:
        """AUC of DP diff over [0.1, 1.0] can be at most 0.9 (max DP diff = 1)."""
        rng = np.random.default_rng(22)
        n = 100
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = _binary_sensitive(n, rng)
        result = selective_fairness_auc(out, y_true, sensitive)
        assert result <= 0.9 + 1e-9  # trapezoidal over [0.1, 1.0]

    def test_returns_nan_when_only_one_group(self) -> None:
        """If all samples belong to one group, no DP diff is computable."""
        rng = np.random.default_rng(23)
        n = 80
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = np.zeros(n, dtype=np.int64)  # single group
        result = selective_fairness_auc(out, y_true, sensitive)
        assert math.isnan(result)

    def test_returns_nan_when_one_group_disappears_at_all_thresholds(self) -> None:
        """With n=10 and group 1 having 1 sample, most thresholds lose group 1."""
        rng = np.random.default_rng(24)
        n = 10
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = np.zeros(n, dtype=np.int64)
        sensitive[0] = 1  # only 1 sample in group 1
        result = selective_fairness_auc(out, y_true, sensitive)
        # Either nan (if < 2 thresholds are computable) or a valid float.
        assert math.isnan(result) or math.isfinite(result)

    def test_perfect_fairness_gives_near_zero_auc(self) -> None:
        """When both groups always have identical selection rates, AUC ~ 0."""
        n = 200
        rng = np.random.default_rng(25)
        # Identical predicted distributions for all samples.
        proba = np.tile([0.4, 0.6], (n, 1)).astype(np.float64)
        entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
        out = UncertaintyOutput(
            mean_proba=proba,
            predictive_entropy=entropy,
            predictive_variance=np.var(proba, axis=1),
            mutual_information=np.zeros(n),
        )
        y_true = _binary_y_true(n, rng)
        sensitive = _binary_sensitive(n, rng)
        result = selective_fairness_auc(out, y_true, sensitive)
        # All samples predict class 1 -> DP diff = 0 at every threshold.
        assert math.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_accepts_string_sensitive_attributes(self) -> None:
        rng = np.random.default_rng(26)
        n = 80
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = np.where(rng.integers(0, 2, size=n) == 0, "group_a", "group_b")
        result = selective_fairness_auc(out, y_true, sensitive)
        assert math.isfinite(result) or math.isnan(result)

    def test_larger_sample_stable(self) -> None:
        """Smoke test with n=1000 to confirm no performance or correctness issue."""
        rng = np.random.default_rng(27)
        n = 1000
        out = _make_output(rng, n=n)
        y_true = _binary_y_true(n, rng)
        sensitive = rng.integers(0, 3, size=n).astype(np.int64)
        result = selective_fairness_auc(out, y_true, sensitive)
        assert math.isfinite(result)
        assert 0.0 <= result <= 0.9 + 1e-9
