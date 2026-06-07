"""Feature-occlusion explainer: the microcontroller-feasible alternative.

Pure NumPy, no heavy dependencies. For each feature, the attribution is the
change in predicted positive-class probability when that feature is replaced
by a baseline value (the column mean by default). This is cheap enough to run
inside the same footprint the audited model runs in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tinyaudit.models.base import AuditedModel

ArrayLike = np.ndarray | pd.DataFrame


def _to_array(X: ArrayLike) -> np.ndarray:
    """Accept an ndarray or a DataFrame and return a float 2-D ndarray."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {arr.shape}")
    return arr


def _positive_proba(model: AuditedModel, X: np.ndarray) -> np.ndarray:
    """Probability of the positive (last) class as a 1-D array."""
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"predict_proba must return a 2-D array, got shape {proba.shape}")
    return proba[:, -1]


def occlusion_attributions(model: AuditedModel, X: ArrayLike, baseline: str = "mean") -> np.ndarray:
    """Per-sample, per-feature occlusion attributions.

    For each feature ``j`` the attribution of sample ``i`` is

        ``p(+ | x_i) - p(+ | x_i with feature j set to baseline_j)``

    so a feature the model ignores gets ~0 and an informative feature gets a
    non-zero signed value. Deterministic given the model and ``X``.

    Parameters
    ----------
    model:
        Any :class:`AuditedModel`.
    X:
        Inputs, ``(n_samples, n_features)``.
    baseline:
        ``"mean"`` (default) or ``"median"`` of ``X``, computed per column.

    Returns
    -------
    np.ndarray
        Signed attributions, shape ``(n_samples, n_features)``.
    """
    arr = _to_array(X)
    n_samples, n_features = arr.shape

    if baseline == "mean":
        base = arr.mean(axis=0)
    elif baseline == "median":
        base = np.median(arr, axis=0)
    else:
        raise ValueError(f"baseline must be 'mean' or 'median', got {baseline!r}")

    base_proba = _positive_proba(model, arr)

    attributions = np.empty((n_samples, n_features), dtype=float)
    for j in range(n_features):
        occluded = arr.copy()
        occluded[:, j] = base[j]
        attributions[:, j] = base_proba - _positive_proba(model, occluded)

    return attributions


def per_group_importance(
    attr: np.ndarray, sensitive: ArrayLike | pd.Series
) -> dict[str, np.ndarray]:
    """Mean absolute attribution per feature, split by sensitive group.

    Parameters
    ----------
    attr:
        Attribution matrix, ``(n_samples, n_features)`` (from any explainer).
    sensitive:
        Group label per sample. Any hashable values; need not be binary.

    Returns
    -------
    dict[str, np.ndarray]
        Group label (stringified) -> mean ``|attribution|`` per feature,
        each value shaped ``(n_features,)``.
    """
    attr = np.asarray(attr, dtype=float)
    if attr.ndim != 2:
        raise ValueError(f"attr must be 2-D, got shape {attr.shape}")

    groups = np.asarray(sensitive).ravel()
    if groups.shape[0] != attr.shape[0]:
        raise ValueError(
            f"sensitive and attr length mismatch: {groups.shape[0]} vs {attr.shape[0]}"
        )

    out: dict[str, np.ndarray] = {}
    for g in pd.unique(groups):
        mask = groups == g
        out[str(g)] = np.abs(attr[mask]).mean(axis=0)
    return out


def importance_flips(per_group: dict[str, np.ndarray]) -> list[int]:
    """Feature indices whose importance rank ordering flips across groups.

    A feature flips when its rank (by mean ``|attribution|``, most important =
    rank 0) is not identical in every group. With only one group there is
    nothing to compare, so the result is empty.

    Parameters
    ----------
    per_group:
        Output of :func:`per_group_importance`.

    Returns
    -------
    list[int]
        Sorted feature indices that change rank between at least two groups.
    """
    if len(per_group) < 2:
        return []

    ranks: list[np.ndarray] = []
    n_features: int | None = None
    for importances in per_group.values():
        imp = np.asarray(importances, dtype=float)
        if n_features is None:
            n_features = imp.shape[0]
        elif imp.shape[0] != n_features:
            raise ValueError("per-group importance vectors have inconsistent length")
        # Descending importance -> rank 0 is the most important feature.
        order = np.argsort(-imp, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.shape[0])
        ranks.append(rank)

    stacked = np.vstack(ranks)
    flipped = np.where((stacked != stacked[0]).any(axis=0))[0]
    return sorted(int(i) for i in flipped)
