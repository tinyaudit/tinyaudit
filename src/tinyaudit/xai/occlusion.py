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


def importance_flips(per_group: dict[str, np.ndarray], top_k: int = 10) -> list[int]:
    """Feature indices whose top-``k`` importance membership flips across groups.

    A feature flips when it is among the ``top_k`` most important features (by
    mean ``|attribution|``) for at least one group but not for every group. In
    other words, the model leans on it for some groups' predictions and not
    others'. This is the meaningful, noise-robust notion of a flip: a one-slot
    wobble deep in the ranking is not a flip, but a feature that is decisive for
    one group and irrelevant for another is.

    Comparing the full rank ordering instead (any exact rank change) is not
    used: with continuous attributions no two groups ever share an identical
    ordering, so it flags almost every feature and means nothing.

    With only one group there is nothing to compare, so the result is empty.

    Parameters
    ----------
    per_group:
        Output of :func:`per_group_importance`.
    top_k:
        Size of the per-group importance shortlist. A feature flips when it is
        in some group's top-``k`` but not in another's. Capped at the feature
        count, so ``top_k`` larger than ``n_features`` yields no flips.

    Returns
    -------
    list[int]
        Sorted feature indices whose top-``k`` membership differs across groups.
    """
    if len(per_group) < 2:
        return []
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    memberships: list[set[int]] = []
    n_features: int | None = None
    for importances in per_group.values():
        imp = np.asarray(importances, dtype=float)
        if n_features is None:
            n_features = imp.shape[0]
        elif imp.shape[0] != n_features:
            raise ValueError("per-group importance vectors have inconsistent length")
        k = min(top_k, imp.shape[0])
        # Descending importance -> the first k indices are this group's top-k.
        top = np.argsort(-imp, kind="stable")[:k]
        memberships.append(set(int(i) for i in top))

    in_all = set.intersection(*memberships)
    in_any = set.union(*memberships)
    # Flipped == top-k for some group but not shared by all groups.
    flipped = in_any - in_all
    return sorted(flipped)
