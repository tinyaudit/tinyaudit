"""Deep-ensemble uncertainty estimator (every model family).

A deep ensemble (Lakshminarayanan et al., 2017) trains the same architecture
under several random seeds and averages the resulting predictive
distributions; the disagreement between members is the epistemic signal.

Design
------
The ensemble supports two construction modes, both documented and tested:

* **Pre-built members.** Pass ``members=[m1, ..., mk]`` where each ``mi`` is
  an already-fitted :class:`AuditedModel`. ``fit`` then only validates them
  (it does not retrain), so callers who trained members elsewhere keep full
  control. ``X``/``y`` are accepted for interface compatibility and used only
  for a shape sanity check.

* **Retrain from a template.** Pass nothing; ``fit(model, X, y)`` clones the
  given model ``n_members`` times and refits each clone on ``(X, y)`` under a
  distinct seed. scikit-learn estimators are cloned with
  ``sklearn.base.clone`` and seeded through whatever ``random_state``-like
  parameter they expose. Torch modules are deep-copied, re-initialised, and
  retrained with a small built-in training loop (the modules here are tiny
  MLPs; the loop is deliberately minimal, not a general trainer).

Either way the members must agree on the class set; aggregation reuses the
shared :func:`tinyaudit.uncertainty.mc_dropout.aggregate_samples` so all three
estimators define entropy / variance / mutual information identically.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from tinyaudit.models.base import AuditedModel
from tinyaudit.models.sklearn import SklearnModel
from tinyaudit.models.torch import TorchModel
from tinyaudit.uncertainty.mc_dropout import aggregate_samples
from tinyaudit.uncertainty.types import NDArrayAny, UncertaintyOutput

_DEFAULT_N_MEMBERS = 5


def _seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def _reinit_module(module: Any) -> None:
    """Reset the parameters of every child that knows how to."""
    import torch

    def _reset(m: Any) -> None:
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    with torch.no_grad():
        module.apply(_reset)


def _train_torch_member(
    module: Any, X: NDArrayAny, y: NDArrayAny, *, seed: int, epochs: int, lr: float
) -> TorchModel:
    """Tiny SGD/Adam loop for one ensemble member. MLP-scale only."""
    import torch
    from torch import nn

    _seed_everything(seed)
    _reinit_module(module)

    xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(y).astype(np.int64))

    module.train()
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        out = module(xt)
        if isinstance(out, (tuple, list)):
            out = out[0]
        logits = torch.as_tensor(out)
        if logits.ndim == 1 or logits.shape[-1] == 1:
            # Binary single-logit head -> two-column logits for CE.
            p1 = logits.reshape(-1)
            logits = torch.stack([torch.zeros_like(p1), p1], dim=1)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    module.eval()
    return TorchModel(module)


class DeepEnsemble:
    """Deep-ensemble estimator across the sklearn and torch model families.

    Parameters
    ----------
    n_members:
        Ensemble size when retraining from a template (default 5, the value
        the spec fixes). Ignored when ``members`` is given.
    members:
        Optional pre-fitted :class:`AuditedModel` instances to use directly.
    seed:
        Base seed; member ``i`` is trained with ``seed + i``.
    epochs, lr:
        Hyperparameters for the built-in torch training loop (template mode
        only). Small by default to keep the audit's own cost bounded.
    """

    def __init__(
        self,
        n_members: int = _DEFAULT_N_MEMBERS,
        *,
        members: Sequence[AuditedModel] | None = None,
        seed: int = 0,
        epochs: int = 60,
        lr: float = 0.05,
    ) -> None:
        if members is not None and len(members) < 2:
            raise ValueError("a deep ensemble needs at least 2 members")
        if members is None and n_members < 2:
            raise ValueError("a deep ensemble needs at least 2 members")
        self.n_members = n_members
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self._members: list[AuditedModel] = list(members) if members is not None else []
        self._prebuilt = members is not None

    @property
    def members(self) -> list[AuditedModel]:
        return self._members

    def _clone_and_fit_sklearn(
        self, template: SklearnModel, X: NDArrayAny, y: NDArrayAny
    ) -> list[AuditedModel]:
        from sklearn.base import clone

        out: list[AuditedModel] = []
        for i in range(self.n_members):
            est = clone(template.estimator)
            params = est.get_params()
            if "random_state" in params:
                est.set_params(random_state=self.seed + i)
            est.fit(np.asarray(X), np.asarray(y))
            out.append(SklearnModel(est))
        return out

    def _clone_and_fit_torch(
        self, template: TorchModel, X: NDArrayAny, y: NDArrayAny
    ) -> list[AuditedModel]:
        out: list[AuditedModel] = []
        for i in range(self.n_members):
            module = copy.deepcopy(template.module)
            member = _train_torch_member(
                module, X, y, seed=self.seed + i, epochs=self.epochs, lr=self.lr
            )
            out.append(member)
        return out

    def fit(self, model: AuditedModel, X: NDArrayAny, y: NDArrayAny) -> None:
        if self._prebuilt:
            n = np.asarray(X).shape[0] if X is not None else None
            for m in self._members:
                if n is not None:
                    proba = m.predict_proba(X)
                    if proba.shape[0] != n:
                        raise ValueError(
                            "pre-built member returned "
                            f"{proba.shape[0]} rows for {n} input samples"
                        )
            return

        if isinstance(model, SklearnModel):
            self._members = self._clone_and_fit_sklearn(model, X, y)
        elif isinstance(model, TorchModel):
            self._members = self._clone_and_fit_torch(model, X, y)
        else:
            raise TypeError(
                "DeepEnsemble template mode supports SklearnModel and TorchModel; "
                f"got {type(model).__name__}. Pass pre-built members instead."
            )

    def predict_dist(self, X: NDArrayAny) -> UncertaintyOutput:
        if not self._members:
            raise RuntimeError("DeepEnsemble.predict_dist called before fit")

        probas = [np.asarray(m.predict_proba(X), dtype=np.float64) for m in self._members]
        n_classes = probas[0].shape[1]
        for p in probas:
            if p.shape[1] != n_classes:
                raise ValueError(
                    "ensemble members disagree on the number of classes "
                    f"({p.shape[1]} vs {n_classes}); they must share a class set"
                )
        samples = np.stack(probas, axis=0)
        return aggregate_samples(samples)
