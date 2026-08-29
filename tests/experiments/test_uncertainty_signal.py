"""Unit tests for the pure logic in ``experiments/run_uncertainty_signal.py``.

The experiment script itself is validated by running, but its correctness rests
on two pure helpers -- the Spearman rank correlation that drives the
complementarity verdict, and the ``cell_rows`` reshape that pairs the three
lenses per group. Both are tested here in isolation, with no models fitted.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from tinyaudit.card.schema import (
    AuditCard,
    FairnessBlock,
    MetricValue,
    UncertaintyBlock,
)
from tinyaudit.profile.footprint import Footprint

# Load the experiment module by path (experiments/ is not an installed package).
_MOD_PATH = Path(__file__).resolve().parents[2] / "experiments" / "run_uncertainty_signal.py"
_spec = importlib.util.spec_from_file_location("run_uncertainty_signal", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
uncsig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uncsig)


class TestSpearman:
    def test_perfect_positive(self) -> None:
        assert uncsig.spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)

    def test_perfect_negative(self) -> None:
        assert uncsig.spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(
            -1.0
        )

    def test_monotone_but_nonlinear_is_rank_one(self) -> None:
        # Spearman is on ranks, so a monotone nonlinear map is still +1.
        assert uncsig.spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)

    def test_fewer_than_three_points_is_nan(self) -> None:
        # A 2-group attribute makes rank correlation trivial; not reported.
        assert math.isnan(uncsig.spearman([1.0, 2.0], [2.0, 1.0]))

    def test_nan_pairs_dropped(self) -> None:
        # Dropping the nan pair leaves a perfectly-ordered triple.
        r = uncsig.spearman([1.0, float("nan"), 2.0, 3.0], [1.0, 9.0, 2.0, 3.0])
        assert r == pytest.approx(1.0)

    def test_zero_variance_is_nan(self) -> None:
        assert math.isnan(uncsig.spearman([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]))


class TestDisparity:
    def test_max_minus_min(self) -> None:
        assert uncsig._disparity([0.1, 0.5, 0.3]) == pytest.approx(0.4)

    def test_single_finite_is_zero(self) -> None:
        assert uncsig._disparity([float("nan"), 0.4]) == 0.0

    def test_all_nan_is_nan(self) -> None:
        assert math.isnan(uncsig._disparity([float("nan"), float("nan")]))


def _card_with(groups: dict[str, dict[str, float]], unc: dict[str, dict[str, float]]) -> AuditCard:
    """Build a minimal AuditCard carrying per-group fairness and uncertainty."""
    return AuditCard(
        dataset="synthetic",
        model="LogisticRegression",
        compression=None,
        footprint=Footprint(n_params=1, peak_ram_bytes=1, flops=1, wall_clock_s_per_sample=0.0),
        fairness=FairnessBlock(
            metrics=[MetricValue(name="demographic_parity_difference", value=0.1, band="red")],
            per_group=groups,
        ),
        uncertainty=UncertaintyBlock(metrics=[], per_group=unc),
        manifest_path="/tmp/none.json",
    )


class TestCellRows:
    def test_pairs_three_lenses_per_group(self) -> None:
        card = _card_with(
            groups={
                "A": {"selection_rate": 0.2, "n": 100},
                "B": {"selection_rate": 0.5, "n": 80},
                "C": {"selection_rate": 0.1, "n": 60},
            },
            unc={
                "A": {
                    "mean_entropy": 0.3,
                    "mean_variance": 0.01,
                    "mutual_information": 0.02,
                    "ece": 0.05,
                },
                "B": {
                    "mean_entropy": 0.4,
                    "mean_variance": 0.02,
                    "mutual_information": 0.03,
                    "ece": 0.01,
                },
                "C": {
                    "mean_entropy": 0.2,
                    "mean_variance": 0.00,
                    "mutual_information": 0.01,
                    "ece": 0.09,
                },
            },
        )
        rows = uncsig.cell_rows(card, "adult", "logreg", "race", "none", 0)
        assert len(rows) == 3
        by_group = {r["group"]: r for r in rows}
        assert by_group["B"]["selection_rate"] == pytest.approx(0.5)
        assert by_group["B"]["mean_entropy"] == pytest.approx(0.4)
        assert by_group["C"]["ece"] == pytest.approx(0.09)
        # Cell-level disparity is repeated onto every group row.
        assert by_group["A"]["entropy_disparity"] == pytest.approx(0.2)  # 0.4 - 0.2
        assert by_group["A"]["ece_disparity"] == pytest.approx(0.08)  # 0.09 - 0.01

    def test_missing_uncertainty_group_is_nan(self) -> None:
        card = _card_with(
            groups={"A": {"selection_rate": 0.2, "n": 10}, "B": {"selection_rate": 0.3, "n": 10}},
            unc={},  # uncertainty stage skipped
        )
        rows = uncsig.cell_rows(card, "adult", "logreg", "sex", "none", 0)
        assert all(math.isnan(r["mean_entropy"]) for r in rows)
