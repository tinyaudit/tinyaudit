"""QUTE-style early-exit-assisted ensemble (torch MLP).

QUTE (Quantization-aware Uncertainty via early-exiTs and Ensembles) gets an
ensemble for almost free by reading the *intermediate* classifier heads a
network already computes on its way to the final prediction. Each early-exit
branch is a weak, cheap ensemble member; together with the final head they
form an ensemble whose disagreement is the uncertainty signal, at the cost of
a single forward pass.

Architecture assumption (documented and asserted)
-------------------------------------------------
The wrapped :class:`TorchModel`'s module must expose early-exit heads through
the contract :class:`TorchModel` already defines: either a callable
``early_exit_logits(x) -> list[Tensor]`` or an attribute of that name
populated during ``forward``. Each entry is a logit tensor of shape
``(n_samples, n_classes)`` for one branch. The final-head logits are obtained
from the ordinary forward pass and appended as the last member, so an
``E``-branch network yields an ``E + 1``-member ensemble. A module without any
early-exit heads is rejected in ``fit`` (there is no ensemble to form).

Branches at different depths are weighted equally; QUTE's refinement of
learnt per-exit weights is out of scope for this small-model audit and would
not change the interface. Aggregation reuses the shared
:func:`tinyaudit.uncertainty.mc_dropout.aggregate_samples`.
"""

from __future__ import annotations

import numpy as np

from tinyaudit.models.base import AuditedModel
from tinyaudit.models.torch import TorchModel
from tinyaudit.uncertainty.mc_dropout import aggregate_samples
from tinyaudit.uncertainty.types import NDArrayAny, UncertaintyOutput

_EPS = 1e-12


def _softmax(logits: NDArrayAny) -> NDArrayAny:
    """Row-wise stable softmax; a single-logit column becomes 2 columns."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.shape[1] == 1:
        # Binary one-logit head: [1 - sigmoid, sigmoid].
        s = 1.0 / (1.0 + np.exp(-z[:, 0]))
        return np.asarray(np.stack([1.0 - s, s], axis=1))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / np.clip(e.sum(axis=1, keepdims=True), _EPS, None))


class EarlyExitEnsemble:
    """QUTE-style early-exit ensemble. Applies to a torch MLP only.

    ``fit`` validates that the module advertises early-exit heads and stores
    the model; no extra training happens (the branches are already trained as
    part of the host network). ``predict_dist`` runs one forward pass, reads
    every branch plus the final head, softmaxes each, and aggregates.
    """

    def __init__(self, *, include_final: bool = True) -> None:
        self.include_final = include_final
        self._model: TorchModel | None = None

    def fit(self, model: AuditedModel, X: NDArrayAny, y: NDArrayAny) -> None:
        if not isinstance(model, TorchModel):
            raise TypeError(
                "EarlyExitEnsemble applies to the torch MLP only; " f"got a {type(model).__name__}"
            )
        # A length-1 probe is enough to confirm the heads exist.
        probe = np.asarray(X, dtype=np.float64)[:1]
        if probe.shape[0] == 0:
            raise ValueError("fit needs at least one sample to probe early-exit heads")
        heads = model.early_exit_logits(probe)
        if heads is None or len(heads) == 0:
            raise ValueError(
                "EarlyExitEnsemble requires the module to expose early-exit heads "
                "(a callable or attribute 'early_exit_logits' yielding per-branch "
                "logits); the wrapped module exposes none."
            )
        self._model = model

    def predict_dist(self, X: NDArrayAny) -> UncertaintyOutput:
        if self._model is None:
            raise RuntimeError("EarlyExitEnsemble.predict_dist called before fit")

        heads = self._model.early_exit_logits(X)
        if heads is None or len(heads) == 0:
            raise RuntimeError(
                "module stopped exposing early-exit heads between fit and predict_dist"
            )

        members = [_softmax(h) for h in heads]
        if self.include_final:
            members.append(np.asarray(self._model.predict_proba(X), dtype=np.float64))

        n_classes = members[0].shape[1]
        for m in members:
            if m.shape[1] != n_classes:
                raise ValueError(
                    "early-exit branches disagree on the number of classes "
                    f"({m.shape[1]} vs {n_classes})"
                )
        samples = np.stack(members, axis=0)
        return aggregate_samples(samples)
