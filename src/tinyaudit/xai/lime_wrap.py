"""LIME adapter (model-agnostic).

LIME has a fragile build (it pulls an old ``scikit-image`` that conflicts
with the ``numpy<2`` pin). It is imported lazily inside the function so the
rest of the package keeps working if the import breaks: a clear, actionable
:class:`RuntimeError` is raised instead of an opaque ``ImportError`` at
package import time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tinyaudit.models.base import AuditedModel

ArrayLike = np.ndarray | pd.DataFrame

_LIME_IMPORT_HINT = (
    "LIME is unavailable (failed to import 'lime.lime_tabular'). LIME has a "
    "fragile build under the numpy<2 pin; reinstall it with "
    "SETUPTOOLS_USE_DISTUTILS=stdlib and a scikit-image<0.23, or use "
    "occlusion_attributions / shap_attributions instead."
)


def _to_array(X: ArrayLike) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {arr.shape}")
    return arr


def lime_attributions(model: AuditedModel, X: ArrayLike) -> np.ndarray:
    """Per-sample, per-feature LIME attributions for the positive class.

    Parameters
    ----------
    model:
        Any :class:`AuditedModel`.
    X:
        Inputs, ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Attributions, shape ``(n_samples, n_features)``.

    Raises
    ------
    RuntimeError
        If ``lime.lime_tabular`` cannot be imported. The message names the
        likely cause and the lightweight alternatives.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(_LIME_IMPORT_HINT) from exc

    arr = _to_array(X)
    n_samples, n_features = arr.shape

    explainer = LimeTabularExplainer(
        training_data=arr,
        mode="classification",
        discretize_continuous=False,
        random_state=0,
    )

    attributions = np.zeros((n_samples, n_features), dtype=float)
    for i in range(n_samples):
        explanation = explainer.explain_instance(
            arr[i],
            model.predict_proba,
            num_features=n_features,
            num_samples=200,
        )
        # explain_instance returns (feature_index, weight) pairs for the
        # explained label; map them back onto the feature axis.
        for feature_index, weight in explanation.as_map()[1]:
            attributions[i, feature_index] = weight

    return attributions
