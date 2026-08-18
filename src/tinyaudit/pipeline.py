"""Stage orchestration.

``audit()`` runs the pipeline stages, writes one JSON manifest, and returns
an ``AuditCard``. The CLI calls this same function so a run reproduces from
a YAML config.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tinyaudit._seed import seed_everything
from tinyaudit.card.schema import (
    AuditCard,
    Band,
    FairnessBlock,
    MetricValue,
    PerformanceBlock,
    UncertaintyBlock,
    XaiBlock,
)
from tinyaudit.compress import magnitude_prune, quantize_int8
from tinyaudit.fairness.parity import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
)
from tinyaudit.models.base import AuditedModel
from tinyaudit.models.sklearn import SklearnModel
from tinyaudit.profile.footprint import profile_model

DEFAULT_METHODS = ["fairness", "uncertainty", "xai"]
_IMPLEMENTED = {"fairness", "uncertainty", "xai"}
RESULTS_ROOT = Path("experiments/results")


def _apply_compression(
    audited: AuditedModel,
    compression: str,
    manifest: dict[str, Any],
) -> AuditedModel:
    """Apply ``compression`` to ``audited`` and return the new model.

    ``compression`` selects one of two paths:

    * ``"int8"`` — :func:`tinyaudit.compress.quantize_int8`.
    * ``"prune:<sparsity>"`` — :func:`tinyaudit.compress.magnitude_prune` at
      the given target fraction (``0.0 <= sparsity < 1.0``).

    The stage is timed and logged to ``manifest["stages"]["compression"]``,
    and ``"compression"`` is appended to ``manifest["methods_run"]``.
    """
    spec = compression.strip()
    if spec == "int8":
        method: str = "int8"
        param: float | None = None
    elif spec.startswith("prune:"):
        method = "prune"
        try:
            param = float(spec.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"compression={compression!r} must be 'prune:<sparsity>' "
                f"with a numeric sparsity"
            ) from exc
    else:
        raise ValueError(
            f"unknown compression {compression!r}; supported: " "'int8', 'prune:<sparsity>'"
        )

    t0 = time.perf_counter()
    if method == "int8":
        compressed = quantize_int8(audited)
    else:
        assert param is not None
        compressed = magnitude_prune(audited, param)

    manifest["stages"]["compression"] = {
        "seconds": time.perf_counter() - t0,
        "method": method,
        "param": param,
        "framework_before": getattr(audited, "framework", "unknown"),
        "framework_after": getattr(compressed, "framework", "unknown"),
        "n_params_before": int(audited.n_params),
        "n_params_after": int(compressed.n_params),
    }
    manifest["methods_run"].append("compression")
    return compressed


def _coerce_model(model: Any) -> AuditedModel:
    """Accept an AuditedModel or auto-wrap a fitted scikit-learn estimator."""
    if all(hasattr(model, a) for a in ("predict", "predict_proba", "n_params", "framework")):
        return model
    if hasattr(model, "predict") and hasattr(model, "predict_proba"):
        return SklearnModel(model)
    raise TypeError("model must be an AuditedModel or a fitted scikit-learn estimator")


def _sensitive_array(
    sensitive: str | pd.Series | pd.DataFrame | np.ndarray, X: pd.DataFrame
) -> np.ndarray:
    """Resolve the protected attribute to a 1-D array aligned with ``X``."""
    if isinstance(sensitive, str):
        return np.asarray(X[sensitive])
    if isinstance(sensitive, pd.DataFrame):
        if sensitive.shape[1] != 1:
            raise ValueError(
                "sensitive DataFrame must have exactly one column; "
                "pass one attribute at a time (e.g. sensitive=S['sex'])"
            )
        return sensitive.iloc[:, 0].to_numpy()
    if isinstance(sensitive, pd.Series):
        return sensitive.to_numpy()
    return np.asarray(sensitive)


def _band(metric: str, value: float) -> Band:
    """Traffic-light band for the card.

    DP and EO are differences (0.0 is parity, larger is worse). DI is a
    ratio (1.0 is parity); the four-fifths rule flags below 0.8.
    """
    if metric == "disparate_impact_ratio":
        if value >= 0.80:
            return "green"
        if value >= 0.60:
            return "amber"
        return "red"
    if value < 0.05:
        return "green"
    if value < 0.10:
        return "amber"
    return "red"


def _perf_band(metric: str, value: float, majority_rate: float) -> Band:
    """Traffic-light band for the performance metrics.

    These bands flag *degeneracy*, not model quality. A heavily compressed
    model drifts toward predicting a single class for every input, which
    scores perfectly on demographic parity while being useless; the bands
    below are tuned to make that collapse visible on the card rather than to
    judge whether an accuracy is good for its dataset.

    ``accuracy`` is judged against the majority-class rate on the same
    evaluation set, because an absolute threshold is meaningless on
    imbalanced data (always answering "no" scores ~0.76 on Adult).
    ``balanced_accuracy`` uses absolute thresholds, since 0.5 is chance
    regardless of class balance. The remaining three are output-distribution
    statistics: a positive rate pinned at 0 or 1, or a predicted-probability
    spread near zero, means the model has stopped distinguishing inputs.
    """
    if metric == "accuracy":
        if value >= majority_rate + 0.05:
            return "green"
        if value > majority_rate:
            return "amber"
        return "red"
    if metric == "balanced_accuracy":
        if value >= 0.70:
            return "green"
        if value >= 0.60:
            return "amber"
        return "red"
    if metric == "std_predicted_prob":
        if value >= 0.05:
            return "green"
        if value >= 0.01:
            return "amber"
        return "red"
    # positive_prediction_rate and mean_predicted_prob: distance from a
    # degenerate all-one-class output.
    if 0.05 <= value <= 0.95:
        return "green"
    if 0.01 <= value <= 0.99:
        return "amber"
    return "red"


def _unc_band(metric: str, value: float) -> Band:
    """Traffic-light band for uncertainty-aware metrics."""
    if metric == "selective_fairness_auc":
        if value < 0.05:
            return "green"
        if value < 0.10:
            return "amber"
        return "red"
    # group entropy and ECE: lower is more uniform / better calibrated
    if value < 0.05:
        return "green"
    if value < 0.15:
        return "amber"
    return "red"


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "tinyaudit"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def _positive_scores(audited: AuditedModel, feats: np.ndarray) -> np.ndarray | None:
    """Per-row score used for the output-distribution statistics.

    Binary models give the positive-class column; multi-class models give the
    max class probability, which is still the right collapse signal (it pins
    at 1.0 when the model stops distinguishing inputs). Returns ``None`` if
    the model exposes no usable ``predict_proba``.
    """
    try:
        proba = np.asarray(audited.predict_proba(feats), dtype=float)
    except Exception:  # noqa: BLE001 - probabilities are optional for the protocol
        return None
    if proba.ndim != 2 or proba.shape[0] != feats.shape[0] or proba.shape[1] == 0:
        return None
    if proba.shape[1] == 2:
        return np.asarray(proba[:, 1])
    return np.asarray(proba.max(axis=1))


def _run_performance(
    audited: AuditedModel,
    feats: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    manifest: dict[str, Any],
) -> PerformanceBlock:
    """Accuracy and output-distribution statistics for the audited model.

    This stage exists because a fairness number read on its own is
    unfalsifiable: a model that answers one class for every input has a
    demographic-parity difference of exactly 0.0 and a disparate-impact ratio
    of exactly 1.0. Recording accuracy, balanced accuracy, the positive
    prediction rate, and the spread of the predicted probabilities alongside
    the fairness block is what lets a reader tell an equitable model from a
    collapsed one. Under compression these describe the *compressed* model,
    since ``audited`` has already been replaced by that point.

    Balanced accuracy is the mean of the per-class recalls, so it reads 0.5
    for a constant predictor no matter how imbalanced the labels are.
    """
    t0 = time.perf_counter()

    classes = np.unique(y_true)
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) else float("nan")

    recalls: list[float] = []
    for c in classes:
        mask = y_true == c
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == c)))
    balanced_accuracy = float(np.mean(recalls)) if recalls else float("nan")

    majority_rate = (
        float(max(np.mean(y_true == c) for c in classes)) if len(classes) else float("nan")
    )

    # "Positive" is the largest class label, which is 1 for the 0/1 datasets
    # in this project and keeps the rate well defined for any ordered labels.
    positive_label = classes.max() if len(classes) else None
    positive_rate = (
        float(np.mean(y_pred == positive_label)) if positive_label is not None else float("nan")
    )

    scores = _positive_scores(audited, feats)
    if scores is None or scores.size == 0:
        mean_prob = float("nan")
        std_prob = float("nan")
    else:
        mean_prob = float(np.mean(scores))
        std_prob = float(np.std(scores))

    values = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "positive_prediction_rate": positive_rate,
        "mean_predicted_prob": mean_prob,
        "std_predicted_prob": std_prob,
    }
    metrics = [
        MetricValue(name=name, value=value, band=_perf_band(name, value, majority_rate))
        for name, value in values.items()
        if not np.isnan(value)
    ]

    manifest["stages"]["performance"] = {
        "seconds": time.perf_counter() - t0,
        "metrics": values,
        "majority_class_rate": majority_rate,
    }
    manifest["methods_run"].append("performance")
    return PerformanceBlock(metrics=metrics, majority_class_rate=majority_rate)


def _run_uncertainty(
    audited: AuditedModel,
    feats: np.ndarray,
    y_true: np.ndarray,
    s: np.ndarray,
    seed: int,
    manifest: dict[str, Any],
) -> UncertaintyBlock | None:
    """Run the uncertainty stage; return None on any failure."""
    try:
        from tinyaudit.uncertainty.ensemble import DeepEnsemble
        from tinyaudit.uncertainty.metrics import (
            ece_per_group,
            group_predictive_entropy,
            selective_fairness_auc,
        )

        # Perturb construction derives members from the given (possibly
        # compressed) model's fitted weights, so uncertainty tracks
        # compression (issue #5). Retrain mode would clone+refit from scratch
        # and be blind to it. If the model's float weights are not reachable
        # (int8 / ONNX), fit raises PerturbNotSupportedError and the except
        # below records the uncertainty stage as skipped.
        ens = DeepEnsemble(n_members=5, seed=seed, construction="perturb")
        ens.fit(audited, feats, y_true)
        out = ens.predict_dist(feats)

        group_entropy = group_predictive_entropy(out, s)
        ece = ece_per_group(out, y_true, s)
        sel_auc = selective_fairness_auc(out, y_true, s)

        # Scalar summaries for the card metrics list.
        mean_entropy = float(np.nanmean(list(group_entropy.values())))
        mean_ece = float(np.nanmean(list(ece.values())))

        # Calibration *quality* (mean_ece_per_group, above) and calibration
        # *disparity* (below) answer different questions and a single averaged
        # number cannot separate them: "every group got less trustworthy" and
        # "one group got much less trustworthy" move the mean identically, but
        # only the second is a fairness problem. Groups whose ECE is undefined
        # (empty or single-class) are excluded; a lone group has no gap and so
        # scores 0.0 rather than NaN.
        finite_ece = [v for v in ece.values() if not np.isnan(v)]
        if len(finite_ece) >= 2:
            ece_disparity = float(max(finite_ece) - min(finite_ece))
        elif len(finite_ece) == 1:
            ece_disparity = 0.0
        else:
            ece_disparity = float("nan")

        metrics: list[MetricValue] = [
            MetricValue(
                name="mean_group_predictive_entropy",
                value=mean_entropy,
                band=_unc_band("mean_group_predictive_entropy", mean_entropy),
            ),
            MetricValue(
                name="mean_ece_per_group",
                value=mean_ece,
                band=_unc_band("mean_ece_per_group", mean_ece),
            ),
            MetricValue(
                name="ece_disparity",
                value=ece_disparity if not np.isnan(ece_disparity) else 0.0,
                band=_unc_band(
                    "ece_disparity", ece_disparity if not np.isnan(ece_disparity) else 0.0
                ),
            ),
            MetricValue(
                name="selective_fairness_auc",
                value=sel_auc if not np.isnan(sel_auc) else 0.0,
                band=_unc_band("selective_fairness_auc", sel_auc if not np.isnan(sel_auc) else 0.0),
            ),
        ]
        per_group: dict[str, dict[str, float]] = {}
        for g in group_entropy:
            per_group[g] = {
                "mean_entropy": group_entropy[g],
                "ece": ece.get(g, float("nan")),
            }

        manifest["stages"]["uncertainty"] = {
            "group_entropy": group_entropy,
            "ece_per_group": {k: v for k, v in ece.items()},
            "ece_disparity": ece_disparity,
            "selective_fairness_auc": sel_auc,
            "metrics": {m.name: m.value for m in metrics},
        }
        manifest["methods_run"].append("uncertainty")
        return UncertaintyBlock(metrics=metrics, per_group=per_group)

    except Exception as exc:  # noqa: BLE001
        manifest["stages"]["uncertainty"] = {"error": str(exc)}
        manifest["methods_skipped"].append("uncertainty")
        return None


def _run_xai(
    audited: AuditedModel,
    feats: np.ndarray,
    feature_names: list[str],
    s: np.ndarray,
    manifest: dict[str, Any],
    top_k: int = 10,
) -> XaiBlock | None:
    """Run the XAI stage using occlusion; return None on failure.

    ``top_k`` is the per-group importance shortlist size used to detect
    importance flips across sensitive groups (a feature in some group's top-k
    but not another's).
    """
    try:
        from tinyaudit.xai.occlusion import (
            importance_flips,
            occlusion_attributions,
            per_group_importance,
        )

        attr = occlusion_attributions(audited, feats)
        pg = per_group_importance(attr, s)
        flips = importance_flips(pg, top_k=top_k)

        # Global top-5 features by mean |attribution|.
        global_importance = np.abs(attr).mean(axis=0)
        top_idx = np.argsort(-global_importance)[:5]
        top_features = [feature_names[i] for i in top_idx]

        # Convert per-group ndarray importances to dicts keyed by feature name.
        pg_dict: dict[str, dict[str, float]] = {
            group: {feature_names[i]: float(imp[i]) for i in range(len(feature_names))}
            for group, imp in pg.items()
        }
        flip_names = [feature_names[i] for i in flips]

        manifest["stages"]["xai"] = {
            "explainer": "occlusion",
            "top_k": top_k,
            "top_features": top_features,
            "importance_flips": flip_names,
        }
        manifest["methods_run"].append("xai")
        return XaiBlock(
            top_features=top_features,
            per_group_importance=pg_dict,
            importance_flips=flip_names,
        )

    except Exception as exc:  # noqa: BLE001
        manifest["stages"]["xai"] = {"error": str(exc)}
        manifest["methods_skipped"].append("xai")
        return None


def audit(
    model: Any,
    data: tuple[pd.DataFrame, pd.Series],
    sensitive: str | pd.Series | pd.DataFrame | np.ndarray,
    methods: list[str] | None = None,
    compression: str | None = None,
    seed: int = 0,
    *,
    dataset: str = "dataset",
    xai_top_k: int = 10,
) -> AuditCard:
    """Audit ``model`` on ``data`` grouped by ``sensitive``.

    ``data`` is ``(X, y)``. ``sensitive`` is a column name in ``X`` or an
    aligned Series/array of the protected attribute. ``methods`` defaults to
    all stages; only implemented stages run, the rest are logged as skipped.
    ``compression`` is applied before profiling so every downstream stage sees
    the compressed model. Pass ``"int8"`` for dynamic int8 quantization, or
    ``"prune:<sparsity>"`` (e.g. ``"prune:0.5"``) for magnitude pruning at the
    given target sparsity. ``xai_top_k`` sets the per-group importance shortlist
    size used to flag explainability flips across sensitive groups.
    """
    # Seed every global RNG (random / numpy / torch) and PYTHONHASHSEED from the
    # single ``seed`` flag before any stage runs, so the whole run reproduces.
    # Estimators that take an explicit ``seed=`` (e.g. DeepEnsemble) still get it
    # threaded below; this covers everything reading the global RNG state.
    seed_everything(seed)

    methods = list(DEFAULT_METHODS if methods is None else methods)

    X_raw, y = data
    X = X_raw if isinstance(X_raw, pd.DataFrame) else pd.DataFrame(np.asarray(X_raw))
    y_true = np.asarray(y)
    s = _sensitive_array(sensitive, X)
    if not (len(X) == len(y_true) == len(s)):
        raise ValueError("features, labels, and sensitive must have equal length")

    audited = _coerce_model(model)
    underlying = getattr(audited, "estimator", audited)
    model_name = type(underlying).__name__
    feature_names = list(X.columns.astype(str))

    manifest: dict[str, Any] = {
        "config": {
            "dataset": dataset,
            "model": model_name,
            "sensitive": sensitive if isinstance(sensitive, str) else "<array>",
            "compression": compression,
            "seed": seed,
            "methods_requested": methods,
            "xai_top_k": xai_top_k,
        },
        "library_versions": _library_versions(),
        "stages": {},
        "methods_run": [],
        "methods_skipped": [],
    }

    if compression is not None:
        audited = _apply_compression(audited, compression, manifest)

    feats = X.to_numpy()

    t0 = time.perf_counter()
    footprint = profile_model(audited, feats)
    manifest["stages"]["profile"] = {
        "seconds": time.perf_counter() - t0,
        "footprint": footprint.model_dump(),
    }

    # Predict once and share the labels across the performance and fairness
    # stages: they are two readings of the same forward pass, and computing
    # them separately would let them drift apart on a nondeterministic model.
    y_pred = np.asarray(audited.predict(feats))

    # Performance is unconditional, like profiling. It is not a "method" the
    # caller opts into, because a fairness block without it is exactly the
    # failure mode this tool is meant to catch.
    performance_block = _run_performance(audited, feats, y_true, y_pred, manifest)

    fairness_block: FairnessBlock | None = None
    if "fairness" in methods:
        t0 = time.perf_counter()
        dp = float(demographic_parity_difference(y_pred, s))
        eo = float(equalized_odds_difference(y_true, y_pred, s))
        di = float(disparate_impact_ratio(y_pred, s))
        metrics_list: list[MetricValue] = [
            MetricValue(
                name="demographic_parity_difference",
                value=dp,
                band=_band("demographic_parity_difference", dp),
            ),
            MetricValue(
                name="equalized_odds_difference",
                value=eo,
                band=_band("equalized_odds_difference", eo),
            ),
            MetricValue(
                name="disparate_impact_ratio",
                value=di,
                band=_band("disparate_impact_ratio", di),
            ),
        ]
        per_group = {
            str(g): {
                "selection_rate": float(np.mean(y_pred[s == g])),
                "n": float(np.sum(s == g)),
            }
            for g in np.unique(s)
        }
        fairness_block = FairnessBlock(metrics=metrics_list, per_group=per_group)
        manifest["stages"]["fairness"] = {
            "seconds": time.perf_counter() - t0,
            "metrics": {m.name: m.value for m in metrics_list},
            "per_group": per_group,
        }
        manifest["methods_run"].append("fairness")

    uncertainty_block: UncertaintyBlock | None = None
    if "uncertainty" in methods and "uncertainty" in _IMPLEMENTED:
        t0 = time.perf_counter()
        uncertainty_block = _run_uncertainty(audited, feats, y_true, s, seed, manifest)
        if "uncertainty" in manifest["methods_run"]:
            manifest["stages"]["uncertainty"]["seconds"] = time.perf_counter() - t0

    xai_block: XaiBlock | None = None
    if "xai" in methods and "xai" in _IMPLEMENTED:
        t0 = time.perf_counter()
        xai_block = _run_xai(audited, feats, feature_names, s, manifest, top_k=xai_top_k)
        if "xai" in manifest["methods_run"]:
            manifest["stages"]["xai"]["seconds"] = time.perf_counter() - t0

    for method in methods:
        if method not in _IMPLEMENTED and method not in manifest["methods_skipped"]:
            manifest["methods_skipped"].append(method)
            manifest["stages"][method] = {"skipped": "not yet implemented"}

    if fairness_block is None:
        raise ValueError("methods must include 'fairness' in the current build")

    config_hash = hashlib.sha256(
        json.dumps(manifest["config"], sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    manifest["config_hash"] = config_hash
    run_dir = RESULTS_ROOT / config_hash
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return AuditCard(
        dataset=dataset,
        model=model_name,
        compression=compression,
        footprint=footprint,
        performance=performance_block,
        fairness=fairness_block,
        uncertainty=uncertainty_block,
        explainability=xai_block,
        manifest_path=str(manifest_path),
    )
