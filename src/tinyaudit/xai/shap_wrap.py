"""SHAP adapter.

The explainer is chosen by the model's framework and underlying estimator:

* ``LinearExplainer`` for logistic regression (closed form, exact, fast).
* ``TreeExplainer`` for a decision tree (exact tree path attributions).
* ``KernelExplainer`` otherwise (model-agnostic; background samples are
  capped at ``background_size`` to keep runtimes bounded).

All branches return a per-sample, per-feature attribution array for the
positive class so the card renderer is uniform across explainers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from tinyaudit.models.base import AuditedModel

ArrayLike = np.ndarray | pd.DataFrame


def _to_array(X: ArrayLike) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {arr.shape}")
    return arr


def _estimator_name(model: AuditedModel) -> str:
    """Underlying estimator class name, or '' when not a sklearn wrapper."""
    est = getattr(model, "estimator", None)
    return "" if est is None else type(est).__name__


def _positive_class_slice(values: object, n_features: int) -> np.ndarray:
    """Reduce a SHAP return value to ``(n_samples, n_features)`` for class +.

    SHAP varies its output container across explainers and versions: a list
    (one array per class), a 3-D array ``(n_samples, n_features, n_classes)``,
    or a plain 2-D array for a single output. This normalises all of them.
    """
    if isinstance(values, list):
        # One array per class; take the positive (last) class.
        arr = np.asarray(values[-1], dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes) -> positive class.
            arr = arr[:, :, -1]

    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(
            f"Unexpected SHAP value shape {arr.shape}; expected (n_samples, {n_features})"
        )
    return arr


def shap_attributions(model: AuditedModel, X: ArrayLike, background_size: int = 1000) -> np.ndarray:
    """SHAP attributions for the positive class.

    Parameters
    ----------
    model:
        Any :class:`AuditedModel`. The SHAP explainer is selected from
        ``model.framework`` and the underlying estimator class.
    X:
        Inputs, ``(n_samples, n_features)``.
    background_size:
        Maximum number of background samples for ``KernelExplainer``
        (summarised with ``shap.sample``). Ignored by the linear/tree paths.

    Returns
    -------
    np.ndarray
        Attributions, shape ``(n_samples, n_features)``.
    """
    arr = _to_array(X)
    n_features = arr.shape[1]
    est_name = _estimator_name(model)

    if model.framework == "sklearn" and est_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model.estimator, arr)
        values = explainer.shap_values(arr)
        return _positive_class_slice(values, n_features)

    if model.framework == "sklearn" and est_name == "DecisionTreeClassifier":
        explainer = shap.TreeExplainer(model.estimator)
        values = explainer.shap_values(arr, check_additivity=False)
        return _positive_class_slice(values, n_features)

    # Model-agnostic fallback: KernelExplainer on predict_proba, with a
    # background set capped at ``background_size``.
    n_bg = min(background_size, arr.shape[0])
    background = shap.sample(arr, n_bg, random_state=0)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    values = explainer.shap_values(arr, silent=True, l1_reg="num_features(10)")
    return _positive_class_slice(values, n_features)
