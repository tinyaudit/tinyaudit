"""The performance stage and the calibration-disparity metric.

These cover the pipeline change that makes the compression finding
falsifiable. A fairness score read on its own cannot distinguish an equitable
model from a collapsed one, because a model that answers a single class for
every input scores a demographic-parity difference of exactly 0.0 and a
disparate-impact ratio of exactly 1.0. The performance block is what tells
those two apart, so the tests below pin the collapse signature rather than
just checking the fields exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tinyaudit import audit
from tinyaudit.models import ConstantModel
from tinyaudit.pipeline import _perf_band

_PERF_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "positive_prediction_rate",
    "mean_predicted_prob",
    "std_predicted_prob",
}


def _fit(X, y) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X.to_numpy(), y.to_numpy())
    return clf


def test_performance_block_present_and_complete(binary_classification):
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive, dataset="synthetic")

    assert card.performance is not None
    assert {m.name for m in card.performance.metrics} == _PERF_METRICS
    for m in card.performance.metrics:
        assert np.isfinite(m.value)
        assert m.band in {"green", "amber", "red"}


def test_majority_class_rate_matches_the_data(binary_classification):
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive)

    counts = np.bincount(y.to_numpy())
    assert card.performance is not None
    assert card.performance.majority_class_rate == pytest.approx(counts.max() / len(y))


def test_performance_describes_the_compressed_model(binary_classification):
    """Under compression the block must report the compressed model, not the
    original. If it reported the original, the collapse would be invisible."""
    X, y, sensitive = binary_classification
    clf = _fit(X, y)

    full = audit(clf, data=(X, y), sensitive=sensitive)
    pruned = audit(clf, data=(X, y), sensitive=sensitive, compression="prune:0.9")

    assert full.performance is not None and pruned.performance is not None
    acc_full = next(m.value for m in full.performance.metrics if m.name == "accuracy")
    acc_pruned = next(m.value for m in pruned.performance.metrics if m.name == "accuracy")
    assert acc_pruned < acc_full


def test_constant_predictor_is_perfectly_fair_and_visibly_useless(binary_classification):
    """The paper's central claim, as one assertion.

    Every fairness metric reports parity while every performance metric
    reports collapse. This is the regression test for the failure mode the
    whole tool exists to catch: if a future change lets the fairness block
    look like this without the performance block contradicting it, the audit
    card has started lying.
    """
    X, y, sensitive = binary_classification
    model = ConstantModel.from_labels(y.to_numpy(), mode="majority")

    card = audit(model, data=(X, y), sensitive=sensitive, dataset="synthetic")

    fairness = {m.name: m for m in card.fairness.metrics}
    assert fairness["demographic_parity_difference"].value == 0.0
    assert fairness["disparate_impact_ratio"].value == 1.0
    assert fairness["demographic_parity_difference"].band == "green"
    assert fairness["disparate_impact_ratio"].band == "green"

    assert card.performance is not None
    perf = {m.name: m for m in card.performance.metrics}
    assert perf["balanced_accuracy"].value == pytest.approx(0.5)
    assert perf["std_predicted_prob"].value == pytest.approx(0.0)
    assert perf["accuracy"].value == pytest.approx(card.performance.majority_class_rate)
    for name in ("accuracy", "balanced_accuracy", "std_predicted_prob"):
        assert perf[name].band == "red", f"{name} should flag the collapse"


def test_performance_stage_in_manifest(binary_classification):
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive)

    manifest = json.loads(Path(card.manifest_path).read_text())
    assert "performance" in manifest["methods_run"]
    stage = manifest["stages"]["performance"]
    assert set(stage["metrics"]) == _PERF_METRICS
    assert "majority_class_rate" in stage


class TestPerfBands:
    """Accuracy is banded against the majority-class rate, not an absolute
    threshold, because an absolute one is meaningless on imbalanced data."""

    def test_accuracy_no_better_than_guessing_is_red(self):
        assert _perf_band("accuracy", 0.76, majority_rate=0.76) == "red"
        assert _perf_band("accuracy", 0.70, majority_rate=0.76) == "red"

    def test_accuracy_clearly_above_the_trivial_baseline_is_green(self):
        assert _perf_band("accuracy", 0.85, majority_rate=0.76) == "green"

    def test_accuracy_barely_above_the_trivial_baseline_is_amber(self):
        assert _perf_band("accuracy", 0.78, majority_rate=0.76) == "amber"

    def test_chance_balanced_accuracy_is_red(self):
        assert _perf_band("balanced_accuracy", 0.50, majority_rate=0.76) == "red"

    def test_degenerate_positive_rate_is_red(self):
        assert _perf_band("positive_prediction_rate", 0.0, majority_rate=0.5) == "red"
        assert _perf_band("positive_prediction_rate", 1.0, majority_rate=0.5) == "red"
        assert _perf_band("positive_prediction_rate", 0.4, majority_rate=0.5) == "green"

    def test_zero_probability_spread_is_red(self):
        assert _perf_band("std_predicted_prob", 0.0, majority_rate=0.5) == "red"
        assert _perf_band("std_predicted_prob", 0.2, majority_rate=0.5) == "green"


class TestEceDisparity:
    """Calibration *quality* and calibration *disparity* answer different
    questions, and a single averaged ECE cannot separate them."""

    def test_reported_alongside_the_mean(self, binary_classification):
        X, y, sensitive = binary_classification
        card = audit(_fit(X, y), data=(X, y), sensitive=sensitive)

        assert card.uncertainty is not None
        names = {m.name for m in card.uncertainty.metrics}
        assert {"mean_ece_per_group", "ece_disparity"} <= names

    def test_equals_the_spread_of_the_per_group_values(self, binary_classification):
        X, y, sensitive = binary_classification
        card = audit(_fit(X, y), data=(X, y), sensitive=sensitive)

        assert card.uncertainty is not None
        manifest = json.loads(Path(card.manifest_path).read_text())
        per_group = manifest["stages"]["uncertainty"]["ece_per_group"]
        finite = [v for v in per_group.values() if v is not None and np.isfinite(v)]

        reported = next(m.value for m in card.uncertainty.metrics if m.name == "ece_disparity")
        assert reported == pytest.approx(max(finite) - min(finite))

    def test_single_group_has_no_disparity(self, binary_classification):
        """One group means no gap, which is 0.0 and not NaN."""
        X, y, _ = binary_classification
        only_group = np.zeros(len(y), dtype=int)

        card = audit(_fit(X, y), data=(X, y), sensitive=only_group)

        assert card.uncertainty is not None
        reported = next(m.value for m in card.uncertainty.metrics if m.name == "ece_disparity")
        assert reported == 0.0
