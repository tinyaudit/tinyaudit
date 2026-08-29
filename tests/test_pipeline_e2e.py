"""End-to-end pipeline tests on synthetic data (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tinyaudit import AuditCard, audit
from tinyaudit.models.sklearn import SklearnModel


def _fit(X, y) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X.to_numpy(), y.to_numpy())
    return clf


def test_audit_end_to_end_bare_estimator(binary_classification):
    X, y, sensitive = binary_classification
    clf = _fit(X, y)

    card = audit(clf, data=(X, y), sensitive=sensitive, dataset="synthetic")

    assert isinstance(card, AuditCard)
    assert card.dataset == "synthetic"
    assert card.model == "LogisticRegression"
    assert card.compression is None
    names = {m.name for m in card.fairness.metrics}
    assert names == {
        "demographic_parity_difference",
        "equalized_odds_difference",
        "disparate_impact_ratio",
    }
    for m in card.fairness.metrics:
        assert m.band in {"green", "amber", "red"}
        assert np.isfinite(m.value)
    assert card.footprint.n_params > 0
    # uncertainty and xai are now implemented
    assert card.uncertainty is not None
    assert card.explainability is not None


def test_audit_accepts_wrapped_model(binary_classification):
    X, y, sensitive = binary_classification
    card = audit(SklearnModel(_fit(X, y)), data=(X, y), sensitive=sensitive)
    assert isinstance(card, AuditCard)


def test_manifest_written_and_valid(binary_classification):
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive, dataset="synthetic")

    mpath = Path(card.manifest_path)
    assert mpath.exists()
    manifest = json.loads(mpath.read_text())
    assert manifest["config"]["model"] == "LogisticRegression"
    assert "fairness" in manifest["methods_run"]
    assert "uncertainty" in manifest["methods_run"]
    assert "xai" in manifest["methods_run"]
    assert len(manifest["methods_skipped"]) == 0
    assert "footprint" in manifest["stages"]["profile"]
    assert manifest["config_hash"] in str(mpath)


def test_uncertainty_block_structure(binary_classification):
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive)

    assert card.uncertainty is not None
    names = {m.name for m in card.uncertainty.metrics}
    assert "mean_group_predictive_entropy" in names
    assert "mean_ece_per_group" in names
    assert "selective_fairness_auc" in names
    for m in card.uncertainty.metrics:
        assert m.band in {"green", "amber", "red"}
        assert np.isfinite(m.value)


def test_uncertainty_block_includes_uncertainty_signal(binary_classification):
    """The uncertainty stage surfaces variance / MI (not just calibration).

    The reviewer's ask: distinguish *uncertainty* from *calibration*. ECE is a
    calibration quantity; predictive variance and mutual information are the
    pure-uncertainty ones. Both their group means and their group disparities
    must appear on the card and in the manifest.
    """
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive, dataset="synthetic")

    assert card.uncertainty is not None
    names = {m.name for m in card.uncertainty.metrics}
    assert "mean_group_predictive_variance" in names
    assert "mean_group_mutual_information" in names
    assert "entropy_disparity" in names
    assert "variance_disparity" in names
    assert "mi_disparity" in names

    # Per-group breakdown carries variance and mutual information beside entropy.
    for _group, vals in card.uncertainty.per_group.items():
        assert "mean_variance" in vals
        assert "mutual_information" in vals

    # Manifest records the disparities so the sweep runner can read them.
    manifest = json.loads(Path(card.manifest_path).read_text())
    unc = manifest["stages"]["uncertainty"]
    for key in ("entropy_disparity", "variance_disparity", "mi_disparity"):
        assert key in unc


def test_xai_block_structure(binary_classification):
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive)

    assert card.explainability is not None
    assert len(card.explainability.top_features) > 0
    assert isinstance(card.explainability.importance_flips, list)


def test_html_renders(binary_classification):
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive, dataset="synthetic")
    html = card.to_html()
    assert isinstance(html, str)
    assert "synthetic" in html
    assert "demographic_parity_difference" in html


def test_prune_compression_runs_through_pipeline(binary_classification):
    """Pruning is the dependency-light compression path; verify the full stack."""
    X, y, sensitive = binary_classification
    card = audit(
        _fit(X, y),
        data=(X, y),
        sensitive=sensitive,
        compression="prune:0.5",
        dataset="synthetic",
    )

    assert card.compression == "prune:0.5"
    # Card stays well-formed: fairness still runs, downstream blocks still populate.
    names = {m.name for m in card.fairness.metrics}
    assert "demographic_parity_difference" in names
    assert card.uncertainty is not None
    assert card.explainability is not None

    manifest = json.loads(Path(card.manifest_path).read_text())
    assert manifest["config"]["compression"] == "prune:0.5"
    assert "compression" in manifest["methods_run"]
    cstage = manifest["stages"]["compression"]
    assert cstage["method"] == "prune"
    assert cstage["param"] == 0.5
    assert cstage["n_params_before"] == cstage["n_params_after"]
    assert cstage["framework_before"] == "sklearn"
    assert cstage["framework_after"] == "sklearn"


def test_pruned_model_metrics_diverge_from_baseline(binary_classification):
    """Downstream stages must actually see the compressed weights."""
    X, y, sensitive = binary_classification
    base = audit(_fit(X, y), data=(X, y), sensitive=sensitive)
    pruned = audit(_fit(X, y), data=(X, y), sensitive=sensitive, compression="prune:0.9")

    base_dp = next(
        m.value for m in base.fairness.metrics if m.name == "demographic_parity_difference"
    )
    pruned_dp = next(
        m.value for m in pruned.fairness.metrics if m.name == "demographic_parity_difference"
    )
    # 90% sparsity on a 2-feature logreg should shift behavior, not silently noop.
    assert base_dp != pruned_dp


def test_unknown_compression_raises(binary_classification):
    X, y, sensitive = binary_classification
    with pytest.raises(ValueError, match="unknown compression"):
        audit(_fit(X, y), data=(X, y), sensitive=sensitive, compression="bogus")


def test_prune_with_non_numeric_sparsity_raises(binary_classification):
    X, y, sensitive = binary_classification
    with pytest.raises(ValueError, match="numeric sparsity"):
        audit(_fit(X, y), data=(X, y), sensitive=sensitive, compression="prune:abc")


def test_int8_compression_runs_through_pipeline(binary_classification):
    """int8 quantization swaps the model framework to onnx and keeps the card valid."""
    pytest.importorskip("skl2onnx")
    X, y, sensitive = binary_classification
    card = audit(
        _fit(X, y),
        data=(X, y),
        sensitive=sensitive,
        compression="int8",
        dataset="synthetic",
    )

    assert card.compression == "int8"
    # Fairness, profile, xai still work on the quantized model. Uncertainty may
    # fall through to None because DeepEnsemble cannot retrain a QuantizedOnnxModel
    # member; that is a documented limitation, not a regression.
    assert card.fairness is not None
    for m in card.fairness.metrics:
        assert np.isfinite(m.value)
    assert card.footprint.n_params > 0

    manifest = json.loads(Path(card.manifest_path).read_text())
    cstage = manifest["stages"]["compression"]
    assert cstage["method"] == "int8"
    assert cstage["param"] is None
    assert cstage["framework_before"] == "sklearn"
    assert cstage["framework_after"] == "onnx"


def test_length_mismatch_rejected(binary_classification):
    X, y, sensitive = binary_classification
    with pytest.raises(ValueError):
        audit(_fit(X, y), data=(X, y), sensitive=sensitive.iloc[:-5])


def test_fairness_only_methods(binary_classification):
    """Requesting only fairness skips uncertainty and xai stages."""
    X, y, sensitive = binary_classification
    card = audit(_fit(X, y), data=(X, y), sensitive=sensitive, methods=["fairness"])

    assert card.uncertainty is None
    assert card.explainability is None
    mpath = Path(card.manifest_path)
    manifest = json.loads(mpath.read_text())
    assert "fairness" in manifest["methods_run"]
    assert "uncertainty" not in manifest["methods_run"]
    assert "xai" not in manifest["methods_run"]
