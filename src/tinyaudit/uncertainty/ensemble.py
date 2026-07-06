"""Deep-ensemble uncertainty estimator (every model family).

A deep ensemble (Lakshminarayanan et al., 2017) trains the same architecture
under several random seeds and averages the resulting predictive
distributions; the disagreement between members is the epistemic signal.

Design
------
The ensemble supports three construction modes, all documented and tested:

* **Pre-built members.** Pass ``members=[m1, ..., mk]`` where each ``mi`` is
  an already-fitted :class:`AuditedModel`. ``fit`` then only validates them
  (it does not retrain), so callers who trained members elsewhere keep full
  control. ``X``/``y`` are accepted for interface compatibility and used only
  for a shape sanity check.

* **Retrain from a template** (``construction="retrain"``, the default).
  Pass nothing; ``fit(model, X, y)`` clones the given model ``n_members``
  times and refits each clone on ``(X, y)`` under a distinct seed.
  scikit-learn estimators are cloned with ``sklearn.base.clone`` and seeded
  through whatever ``random_state``-like parameter they expose. Torch modules
  are deep-copied, re-initialised, and retrained with a small built-in
  training loop (the modules here are tiny MLPs; the loop is deliberately
  minimal, not a general trainer).

* **Perturb the fitted solution** (``construction="perturb"``). ``fit(model,
  X, y)`` does *not* retrain. Instead it builds ``n_members`` copies of the
  *given fitted* model and adds small seeded Gaussian noise to each copy's
  weights: for member ``i`` the noise is drawn from
  ``np.random.default_rng(seed + i)`` with standard deviation
  ``perturb_scale * std(W_nonzero)`` per weight matrix ``W``. Pruning masks
  are preserved -- wherever ``W == 0`` the perturbed weight stays exactly 0
  (noise is multiplied by ``W != 0``), so pruned-away connections are never
  revived. Biases/intercepts are perturbed without a mask.

  This is an *anchored / weight-perturbation ensemble* around the fitted
  solution, not the classic Lakshminarayanan independent-retrain deep
  ensemble. The tradeoff is deliberate: because the members are perturbations
  of *the model passed in*, the ensemble's disagreement (and therefore the
  uncertainty it reports) tracks whatever compression that model carries.
  Retrain mode discards the fitted (e.g. compressed) weights by cloning and
  refitting from scratch, so its uncertainty is blind to compression; perturb
  mode fixes that (issue #5). The cost is that the members are less diverse
  than fully independent retrains, so absolute epistemic-uncertainty
  magnitudes are not directly comparable to a textbook deep ensemble.

Either way the members must agree on the class set; aggregation reuses the
shared :func:`tinyaudit.uncertainty.mc_dropout.aggregate_samples` so all three
estimators define entropy / variance / mutual information identically.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from tinyaudit.models.base import AuditedModel
from tinyaudit.models.sklearn import SklearnModel
from tinyaudit.models.torch import TorchModel
from tinyaudit.uncertainty.mc_dropout import aggregate_samples
from tinyaudit.uncertainty.types import NDArrayAny, UncertaintyOutput

_DEFAULT_N_MEMBERS = 5


class PerturbNotSupportedError(RuntimeError):
    """Raised when ``construction="perturb"`` cannot reach a model's weights.

    Perturb mode derives each ensemble member from the *given* model's fitted
    float weights. Models whose float weights are not reachable (e.g. int8 /
    ONNX-backed :class:`~tinyaudit.compress.quantize.QuantizedOnnxModel`, or an
    sklearn family with no dense weight matrix) cannot be perturbed. This is
    raised rather than silently falling back to an uncompressed retrain, so the
    caller can record an explicit skip instead of reporting misleading
    uncertainty.
    """


def _perturb_array(
    weights: NDArrayAny, rng: np.random.Generator, scale: float, *, mask: bool
) -> NDArrayAny:
    """Return ``weights`` plus seeded Gaussian noise.

    The noise standard deviation is ``scale * std(nonzero weights)``, so the
    perturbation is proportional to the weight matrix's own magnitude. When
    ``mask`` is true the noise is zeroed wherever ``weights == 0``, preserving a
    pruning mask exactly (pruned-away connections are never revived).
    """
    arr = np.array(weights, dtype=np.float64, copy=True)
    nonzero = arr[arr != 0.0]
    std = float(np.std(nonzero)) if nonzero.size > 0 else 0.0
    noise = rng.normal(loc=0.0, scale=scale * std, size=arr.shape)
    if mask:
        noise = noise * (arr != 0.0)
    return arr + noise


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
    construction:
        Template-mode strategy (ignored when ``members`` is given).
        ``"retrain"`` (default) clones and refits each member from scratch --
        this discards the given model's fitted weights, so the ensemble is
        blind to any compression carried by that model. ``"perturb"`` instead
        builds each member as a seeded Gaussian perturbation of the given
        fitted model's weights, so the ensemble's uncertainty tracks the
        compressed solution (issue #5). See the module docstring for the
        anchored-ensemble tradeoff.
    perturb_scale:
        Perturb-mode noise scale: member weights are ``W + N(0, perturb_scale
        * std(W_nonzero))`` per weight matrix. Ignored in retrain mode.
    """

    def __init__(
        self,
        n_members: int = _DEFAULT_N_MEMBERS,
        *,
        members: Sequence[AuditedModel] | None = None,
        seed: int = 0,
        epochs: int = 60,
        lr: float = 0.05,
        construction: Literal["retrain", "perturb"] = "retrain",
        perturb_scale: float = 0.1,
    ) -> None:
        if members is not None and len(members) < 2:
            raise ValueError("a deep ensemble needs at least 2 members")
        if members is None and n_members < 2:
            raise ValueError("a deep ensemble needs at least 2 members")
        if construction not in ("retrain", "perturb"):
            raise ValueError(f"construction must be 'retrain' or 'perturb'; got {construction!r}")
        self.n_members = n_members
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.construction = construction
        self.perturb_scale = perturb_scale
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

    def _perturb_sklearn(self, model: SklearnModel) -> list[AuditedModel]:
        """Build members by perturbing a fitted sklearn estimator's weights.

        Supports ``LogisticRegression`` (``coef_`` masked, ``intercept_``
        unmasked) and ``MLPClassifier`` (each ``coefs_`` matrix masked, each
        ``intercepts_`` vector unmasked). Any other family raises
        :class:`PerturbNotSupportedError` (no dense weight matrix to perturb).
        """
        cls_name = type(model.estimator).__name__
        if cls_name not in ("LogisticRegression", "MLPClassifier"):
            raise PerturbNotSupportedError(
                "DeepEnsemble perturb mode supports the sklearn weight-array "
                "families LogisticRegression (coef_) and MLPClassifier "
                f"(coefs_); got {cls_name!r}, which has no prunable dense "
                "weight matrix to perturb."
            )

        out: list[AuditedModel] = []
        for i in range(self.n_members):
            rng = np.random.default_rng(self.seed + i)
            est = copy.deepcopy(model.estimator)
            if cls_name == "LogisticRegression":
                est.coef_ = _perturb_array(
                    np.asarray(est.coef_), rng, self.perturb_scale, mask=True
                )
                est.intercept_ = _perturb_array(
                    np.asarray(est.intercept_), rng, self.perturb_scale, mask=False
                )
            else:  # MLPClassifier
                est.coefs_ = [
                    _perturb_array(np.asarray(c), rng, self.perturb_scale, mask=True)
                    for c in est.coefs_
                ]
                est.intercepts_ = [
                    _perturb_array(np.asarray(b), rng, self.perturb_scale, mask=False)
                    for b in est.intercepts_
                ]
            out.append(SklearnModel(est))
        return out

    def _perturb_torch(self, model: AuditedModel) -> list[AuditedModel]:
        """Build members by perturbing every ``nn.Linear`` of a torch module.

        The module is reached duck-typed via ``model.module`` and re-wrapped
        with ``type(model)(module, device=...)`` exactly as ``compress.prune``
        does. ``weight`` is perturbed with a pruning mask, ``bias`` without.
        """
        import torch
        from torch import nn

        module = getattr(model, "module", None)
        if not isinstance(module, nn.Module):
            raise PerturbNotSupportedError(
                "DeepEnsemble perturb mode received a model with "
                "framework=='torch' but no torch.nn.Module reachable via a "
                "'module' attribute; cannot perturb its weights."
            )

        device: Any = getattr(model, "device", "cpu")
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        # ``type(model)`` is a concrete adapter (TorchModel) whose constructor
        # takes a module; the AuditedModel protocol advertises no constructor,
        # so bind it through ``Any`` (mirrors compress/prune._prune_torch).
        ctor: Any = type(model)

        out: list[AuditedModel] = []
        for i in range(self.n_members):
            rng = np.random.default_rng(self.seed + i)
            member_module = copy.deepcopy(module)
            with torch.no_grad():
                for sub in member_module.modules():
                    if isinstance(sub, nn.Linear):
                        w = sub.weight.detach().cpu().numpy()
                        sub.weight.copy_(
                            torch.as_tensor(
                                _perturb_array(w, rng, self.perturb_scale, mask=True),
                                dtype=sub.weight.dtype,
                            )
                        )
                        if sub.bias is not None:
                            b = sub.bias.detach().cpu().numpy()
                            sub.bias.copy_(
                                torch.as_tensor(
                                    _perturb_array(b, rng, self.perturb_scale, mask=False),
                                    dtype=sub.bias.dtype,
                                )
                            )
            try:
                member = ctor(member_module, **kwargs)
            except TypeError:
                member = ctor(member_module)
            out.append(member)
        return out

    def _build_perturb_members(self, model: AuditedModel) -> list[AuditedModel]:
        if isinstance(model, SklearnModel):
            return self._perturb_sklearn(model)
        if getattr(model, "framework", None) == "torch":
            return self._perturb_torch(model)
        raise PerturbNotSupportedError(
            "DeepEnsemble perturb mode supports sklearn (LogisticRegression, "
            "MLPClassifier) and torch models with reachable float weights; got "
            f"framework={getattr(model, 'framework', None)!r} "
            f"({type(model).__name__}). Its float weights are not reachable "
            "(e.g. int8 / ONNX-backed), so perturb mode cannot derive members "
            "from it."
        )

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

        if self.construction == "perturb":
            self._members = self._build_perturb_members(model)
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
