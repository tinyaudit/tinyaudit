"""Bin placement and bin count in ``ece_per_group``.

The default is 10 equal-width bins, which is what the pipeline and every
existing result use, so the first thing pinned here is that the default has not
moved. The rest covers the equal-mass path, whose reason for existing is the
degenerate case: a collapsed model emits nearly one repeated confidence, and
equal-width binning then leaves almost every bin empty.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyaudit.uncertainty.metrics import ece_per_group
from tinyaudit.uncertainty.types import UncertaintyOutput


def _output(proba: np.ndarray) -> UncertaintyOutput:
    """An UncertaintyOutput carrying ``proba``; only ``mean_proba`` is read."""
    n = len(proba)
    return UncertaintyOutput(
        mean_proba=proba,
        predictive_entropy=np.zeros(n),
        predictive_variance=np.zeros(n),
        mutual_information=np.zeros(n),
    )


@pytest.fixture
def spread_case():
    """Confidences spread across [0.5, 1.0] with two sensitive groups."""
    rng = np.random.default_rng(0)
    n = 400
    p1 = rng.uniform(0.5, 1.0, size=n)
    proba = np.column_stack([1.0 - p1, p1])
    y_true = (rng.random(n) < p1).astype(int)
    sensitive = np.where(np.arange(n) < n // 2, "a", "b")
    return _output(proba), y_true, sensitive


def test_default_is_unchanged(spread_case):
    """The default call must equal an explicit 10-bin equal-width call."""
    out, y, s = spread_case
    assert ece_per_group(out, y, s) == ece_per_group(out, y, s, n_bins=10, binning="equal_width")


def test_equal_mass_is_a_real_alternative(spread_case):
    """Different placement, same groups, finite values, and not identical."""
    out, y, s = spread_case
    width = ece_per_group(out, y, s, binning="equal_width")
    mass = ece_per_group(out, y, s, binning="equal_mass")

    assert set(width) == set(mass)
    assert all(np.isfinite(v) for v in mass.values())
    assert width != mass


def test_bin_count_matters_less_than_placement(spread_case):
    """The originally proposed 10-vs-15 check is the weaker axis.

    Both counts are equal-width, so they mostly agree by construction. This
    test states that quantitatively rather than as an opinion: changing the
    count moves ECE less than changing the placement does.
    """
    out, y, s = spread_case
    w10 = ece_per_group(out, y, s, n_bins=10, binning="equal_width")
    w15 = ece_per_group(out, y, s, n_bins=15, binning="equal_width")
    m10 = ece_per_group(out, y, s, n_bins=10, binning="equal_mass")

    count_shift = max(abs(w10[g] - w15[g]) for g in w10)
    placement_shift = max(abs(w10[g] - m10[g]) for g in w10)
    assert placement_shift > count_shift


class TestDegenerateModel:
    """A constant predictor is the case that motivates equal-mass binning."""

    @staticmethod
    def _constant_case(confidence: float = 0.76, n: int = 200):
        proba = np.tile([1.0 - confidence, confidence], (n, 1))
        rng = np.random.default_rng(1)
        y_true = (rng.random(n) < confidence).astype(int)
        sensitive = np.where(np.arange(n) < n // 2, "a", "b")
        return _output(proba), y_true, sensitive

    def test_equal_width_survives_a_single_populated_bin(self):
        out, y, s = self._constant_case()
        result = ece_per_group(out, y, s, binning="equal_width")
        assert all(np.isfinite(v) for v in result.values())

    def test_equal_mass_collapses_to_one_bin_without_crashing(self):
        """Every quantile edge is identical here, so the deduplicated edges
        leave a single bin. The result must still be the plain gap between
        mean confidence and accuracy, not NaN and not a crash."""
        out, y, s = self._constant_case()
        result = ece_per_group(out, y, s, binning="equal_mass")

        assert set(result) == {"a", "b"}
        for group in ("a", "b"):
            assert np.isfinite(result[group])
            mask = s == group
            accuracy = float(np.mean(y[mask] == 1))
            assert result[group] == pytest.approx(abs(accuracy - 0.76))

    def test_both_binnings_agree_when_confidence_is_constant(self):
        """With one distinct confidence there is nothing for placement to
        change, so the two schemes must coincide."""
        out, y, s = self._constant_case()
        assert ece_per_group(out, y, s, binning="equal_width") == pytest.approx(
            ece_per_group(out, y, s, binning="equal_mass")
        )


class TestValidation:
    def test_unknown_binning_is_rejected(self, spread_case):
        out, y, s = spread_case
        with pytest.raises(ValueError, match="equal_width.*equal_mass"):
            ece_per_group(out, y, s, binning="quantile")

    def test_non_positive_bin_count_is_rejected(self, spread_case):
        out, y, s = spread_case
        with pytest.raises(ValueError, match="at least 1"):
            ece_per_group(out, y, s, n_bins=0)

    @pytest.mark.parametrize("binning", ["equal_width", "equal_mass"])
    def test_tiny_group_is_nan_not_a_crash(self, binning):
        proba = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        y = np.array([1, 0, 1])
        s = np.array(["a", "a", "b"])  # group "b" has a single sample

        result = ece_per_group(_output(proba), y, s, binning=binning)
        assert np.isnan(result["b"])
        assert np.isfinite(result["a"])
