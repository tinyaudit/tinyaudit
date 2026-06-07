"""PyTorch adapter conforming to the :class:`AuditedModel` protocol.

Wraps a fitted ``torch.nn.Module`` classifier (binary or multiclass) so the
rest of the pipeline depends only on the protocol, never on PyTorch directly.

Beyond the protocol, the wrapper exposes the two hooks the uncertainty
estimators need on an MLP:

* ``predict_proba(X, enable_dropout=True)`` (or the ``mc_dropout`` context
  manager) runs a stochastic forward pass with every ``nn.Dropout`` /
  ``nn.Dropout*d`` / ``nn.AlphaDropout`` module switched to ``train`` mode
  while the rest of the network stays in ``eval`` mode (so e.g. BatchNorm
  running statistics are *not* perturbed). This is exactly the regime MC
  Dropout samples from.
* ``early_exit_logits(X)`` returns the per-branch logits of any module that
  advertises intermediate early-exit heads (see ``EARLY_EXIT_ATTR``), used by
  the QUTE-style early-exit ensemble. ``None`` when the module has no heads.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

# Dropout module types that MC Dropout reactivates at inference time.
_DROPOUT_TYPES: tuple[type[nn.Module], ...] = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
)

# A module exposes early-exit heads either by defining a method with this name
# returning a list of per-branch logit tensors, or by carrying an attribute of
# this name holding the most recent such list (populated during ``forward``).
EARLY_EXIT_ATTR = "early_exit_logits"


def _to_tensor(X: np.ndarray | pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Accept an ndarray or a DataFrame and return a 2-D float32 tensor."""
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
    arr = np.atleast_2d(np.asarray(arr, dtype=np.float32))
    return torch.from_numpy(arr).to(device)


def _softmax_np(logits: torch.Tensor) -> np.ndarray:
    """Stable softmax over the last axis, returned as float64 ndarray.

    A 1-D logit vector (the binary-with-one-output edge case) is promoted to
    two columns ``[1 - sigmoid, sigmoid]`` so ``predict_proba`` always has
    shape ``(n_samples, n_classes)``.
    """
    if logits.ndim == 1 or logits.shape[-1] == 1:
        p1 = torch.sigmoid(logits.reshape(-1))
        proba = torch.stack([1.0 - p1, p1], dim=1)
    else:
        proba = torch.softmax(logits, dim=-1)
    return proba.detach().cpu().numpy().astype(np.float64)


class TorchModel:
    """An :class:`AuditedModel` backed by a fitted ``torch.nn.Module``.

    The module must already be trained and must map a feature batch of shape
    ``(n_samples, n_features)`` to class logits of shape
    ``(n_samples, n_classes)`` (or a length-``n_samples`` vector / a single
    output column for the binary case). It is moved to ``device`` and held in
    ``eval`` mode; the wrapper never trains it.
    """

    def __init__(self, module: nn.Module, *, device: str | torch.device = "cpu") -> None:
        self._device = torch.device(device)
        self._module = module.to(self._device)
        self._module.eval()

    @property
    def module(self) -> nn.Module:
        """The wrapped fitted ``torch.nn.Module``."""
        return self._module

    @property
    def device(self) -> torch.device:
        """The device the module and inputs live on."""
        return self._device

    @property
    def framework(self) -> str:
        return "torch"

    @property
    def n_params(self) -> int:
        """Total number of learnable parameters: ``sum(p.numel())``.

        Counts every entry of every ``Parameter`` returned by
        ``module.parameters()`` (frozen parameters included; they are still
        part of the model's footprint).
        """
        return int(sum(p.numel() for p in self._module.parameters()))

    # --------------------------------------------------------------------- #
    # MC-Dropout support: dropout modules train, everything else eval.
    # --------------------------------------------------------------------- #
    @contextmanager
    def mc_dropout(self) -> Iterator[None]:
        """Context manager: dropout layers ``train``, rest ``eval``.

        Restores every touched module's original training flag on exit, so
        nesting and exceptions cannot leave the network in a stochastic state.
        """
        toggled: list[tuple[nn.Module, bool]] = []
        try:
            for sub in self._module.modules():
                if isinstance(sub, _DROPOUT_TYPES):
                    toggled.append((sub, sub.training))
                    sub.train(True)
            yield
        finally:
            for sub, was_training in toggled:
                sub.train(was_training)

    def has_dropout(self) -> bool:
        """True if the wrapped module contains at least one dropout layer."""
        return any(isinstance(m, _DROPOUT_TYPES) for m in self._module.modules())

    @torch.no_grad()
    def _logits(self, X: np.ndarray | pd.DataFrame) -> torch.Tensor:
        out = self._module(_to_tensor(X, self._device))
        if isinstance(out, (tuple, list)):
            # Modules that also return early-exit heads put the final logits
            # first by convention.
            out = out[0]
        return torch.as_tensor(out)

    def predict_proba(
        self, X: np.ndarray | pd.DataFrame, *, enable_dropout: bool = False
    ) -> np.ndarray:
        """Per-class probabilities, shape ``(n_samples, n_classes)``.

        With ``enable_dropout=True`` a single stochastic forward pass is run
        under :meth:`mc_dropout` (one Monte-Carlo sample); otherwise the
        network is fully deterministic.
        """
        if enable_dropout:
            with self.mc_dropout():
                return _softmax_np(self._logits(X))
        return _softmax_np(self._logits(X))

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Hard class labels (deterministic ``argmax`` of the softmax)."""
        return np.asarray(np.argmax(self.predict_proba(X), axis=1))

    # --------------------------------------------------------------------- #
    # Early-exit support for the QUTE-style estimator.
    # --------------------------------------------------------------------- #
    def early_exit_logits(self, X: np.ndarray | pd.DataFrame) -> list[np.ndarray] | None:
        """Per-branch early-exit logits, or ``None`` if the module has none.

        The module advertises heads by either (a) defining a callable
        ``early_exit_logits(x) -> list[Tensor]`` or (b) populating an attribute
        ``early_exit_logits`` (a list of per-branch logit tensors) during its
        ``forward``. Each returned array has shape ``(n_samples, n_classes)``.
        """
        attr = getattr(self._module, EARLY_EXIT_ATTR, None)

        with torch.no_grad():
            tensors: Any
            if callable(attr):
                tensors = attr(_to_tensor(X, self._device))
            else:
                # Trigger a forward so a forward-populated attribute is fresh.
                _ = self._logits(X)
                tensors = getattr(self._module, EARLY_EXIT_ATTR, None)

        if tensors is None:
            return None
        return [torch.as_tensor(t).detach().cpu().numpy().astype(np.float64) for t in tensors]
