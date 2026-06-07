"""Magnitude pruning.

``magnitude_prune`` zeroes out the smallest-magnitude weights of a fitted
model until a target ``sparsity`` fraction of its prunable weights is exactly
zero, and returns a *new* :class:`AuditedModel` wrapping a deep-copied,
pruned estimator. The original model is never mutated, so a sweep over
several sparsities can reuse the same input model.

Supported families:

* scikit-learn ``LogisticRegression`` (the ``coef_`` matrix).
* scikit-learn ``MLPClassifier`` (every weight matrix in ``coefs_``).
* any torch model exposed through the protocol: every ``nn.Linear``
  ``weight`` tensor of the wrapped module.

Sparsity sweep values used by the experiments: 0.30, 0.50, 0.70, 0.90.

Torch handling is duck-typed on purpose. The torch adapter is built in a
parallel wave and is not imported here: a model is treated as torch when
``model.framework == "torch"`` and its module is reached via
``getattr(model, "module", None)``. A pruned model is rebuilt with
``type(model)(pruned_module, device=...)`` so this module never depends on
the concrete ``TorchModel`` class.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from tinyaudit.models.base import AuditedModel
from tinyaudit.models.sklearn import SklearnModel


def _prune_array(weights: np.ndarray, sparsity: float) -> np.ndarray:
    """Return a copy of ``weights`` with the smallest ``|w|`` set to zero.

    The number zeroed is ``floor(sparsity * size)`` so ``sparsity`` is a
    lower bound on the resulting exact-zero fraction (ceiling ensures at
    least ``sparsity`` fraction of weights are zeroed). ``sparsity == 0.0``
    returns an unmodified copy.
    """
    arr = np.array(weights, copy=True)
    size = arr.size
    n_prune = int(np.ceil(sparsity * size))
    if n_prune <= 0:
        return arr

    flat = arr.reshape(-1)
    magnitudes = np.abs(flat)
    # Indices of the ``n_prune`` smallest magnitudes (unordered, O(n)).
    cut = np.argpartition(magnitudes, n_prune - 1)[:n_prune]
    flat[cut] = 0.0
    return flat.reshape(arr.shape)


def _validate_sparsity(sparsity: float) -> None:
    if not isinstance(sparsity, (int, float)) or isinstance(sparsity, bool):
        raise ValueError(f"sparsity must be a float in [0.0, 1.0); got {sparsity!r}")
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must satisfy 0.0 <= sparsity < 1.0; got {sparsity}")


def _prune_sklearn(model: SklearnModel, sparsity: float) -> SklearnModel:
    """Deep-copy the estimator, prune its weight arrays, re-wrap it."""
    est = copy.deepcopy(model.estimator)
    cls_name = type(est).__name__

    if cls_name == "LogisticRegression":
        est.coef_ = _prune_array(np.asarray(est.coef_), sparsity)
    elif cls_name == "MLPClassifier":
        est.coefs_ = [_prune_array(np.asarray(c), sparsity) for c in est.coefs_]
    else:
        raise RuntimeError(
            "magnitude_prune supports the sklearn weight-array families "
            "LogisticRegression (coef_) and MLPClassifier (coefs_); got "
            f"{cls_name!r}, which has no prunable dense weight matrix."
        )
    return SklearnModel(est)


def _prune_torch(model: AuditedModel, sparsity: float) -> AuditedModel:
    """Deep-copy the wrapped module, prune every Linear weight, re-wrap it.

    Duck-typed: the concrete torch adapter is not imported. The module is
    reached via ``model.module`` and the wrapper is reconstructed with
    ``type(model)(module, device=...)``.
    """
    import torch
    from torch import nn

    module = getattr(model, "module", None)
    if not isinstance(module, nn.Module):
        raise RuntimeError(
            "magnitude_prune received a model with framework=='torch' but "
            "no torch.nn.Module reachable via a 'module' attribute; cannot "
            "prune it."
        )

    pruned_module = copy.deepcopy(module)
    with torch.no_grad():
        for sub in pruned_module.modules():
            if isinstance(sub, nn.Linear):
                w = sub.weight.detach().cpu().numpy()
                pruned = _prune_array(w, sparsity)
                sub.weight.copy_(torch.as_tensor(pruned, dtype=sub.weight.dtype))

    device = getattr(model, "device", "cpu")
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    try:
        return type(model)(pruned_module, **kwargs)
    except TypeError:
        # Fall back to the positional-only constructor if the adapter does
        # not accept a ``device`` keyword.
        return type(model)(pruned_module)


def magnitude_prune(model: AuditedModel, sparsity: float) -> AuditedModel:
    """Zero the smallest-magnitude weights of ``model`` to reach ``sparsity``.

    ``sparsity`` is the target fraction of prunable weights set exactly to
    zero and must satisfy ``0.0 <= sparsity < 1.0`` (``ValueError`` otherwise;
    ``1.0`` would erase the model). ``sparsity == 0.0`` is an effective
    passthrough (a pruned *copy* identical to the original).

    The input model is never mutated: a deep copy of the underlying
    estimator/module is pruned and re-wrapped. The return value satisfies the
    :class:`AuditedModel` protocol (same ``framework``, working ``predict`` /
    ``predict_proba`` / ``n_params``) so every downstream stage runs
    unchanged.

    Supported: sklearn ``LogisticRegression`` and ``MLPClassifier``, and any
    ``framework == "torch"`` model whose module exposes ``nn.Linear`` layers.
    Any other family raises a clear ``RuntimeError`` rather than silently
    returning an unpruned model.
    """
    _validate_sparsity(sparsity)

    framework = getattr(model, "framework", None)

    if isinstance(model, SklearnModel) or framework == "sklearn":
        if not isinstance(model, SklearnModel):
            raise RuntimeError(
                "magnitude_prune got framework=='sklearn' but the model is "
                f"a {type(model).__name__}, not a SklearnModel; cannot reach "
                "its weight arrays."
            )
        return _prune_sklearn(model, sparsity)

    if framework == "torch":
        return _prune_torch(model, sparsity)

    raise RuntimeError(
        "magnitude_prune supports sklearn (LogisticRegression, MLPClassifier) "
        f"and torch models; got framework={framework!r} "
        f"({type(model).__name__}), which is not prunable."
    )
