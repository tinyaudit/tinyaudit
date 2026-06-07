"""MC Dropout uncertainty estimator (torch MLP only).

Monte-Carlo Dropout (Gal & Ghahramani, 2016) treats dropout left on at
inference as a Bernoulli variational approximation to a Bayesian network.
``T`` stochastic forward passes give ``T`` per-sample probability vectors;
their mean is the predictive distribution and their spread is the epistemic
uncertainty.

This module also owns the shared softmax-sample aggregation used by every
estimator in this package (deep ensemble, early-exit ensemble): given the
per-member ``(T, n_samples, n_classes)`` probability tensor it produces the
:class:`UncertaintyOutput`. Keeping that math in one place keeps the three
estimators consistent by construction.
"""

from __future__ import annotations

import numpy as np

from tinyaudit.models.base import AuditedModel
from tinyaudit.models.torch import TorchModel
from tinyaudit.uncertainty.types import NDArrayAny, UncertaintyOutput

# Probabilities are clipped away from 0 before the log so entropy is finite.
_EPS = 1e-12


def _entropy(proba: NDArrayAny) -> NDArrayAny:
    """Shannon entropy (nats) along the last axis.

    ``proba`` has shape ``(..., n_classes)``; the result drops that axis.
    Probabilities are clipped to ``[_EPS, 1]`` so ``0 * log 0`` is ``0``.
    """
    p = np.clip(np.asarray(proba, dtype=np.float64), _EPS, 1.0)
    return np.asarray(-np.sum(p * np.log(p), axis=-1))


def aggregate_samples(samples: NDArrayAny) -> UncertaintyOutput:
    """Reduce a sample tensor to an :class:`UncertaintyOutput`.

    Parameters
    ----------
    samples:
        Array of shape ``(n_members, n_samples, n_classes)`` holding one
        probability vector per (member, sample). ``n_members`` is ``T`` for
        MC Dropout, the ensemble size for a deep ensemble, and the number of
        exits for the early-exit ensemble.

    Returns
    -------
    UncertaintyOutput
        ``mean_proba`` is the member-averaged probability. ``predictive_entropy``
        is the entropy of that mean (total uncertainty). ``mutual_information``
        is ``H(mean) - mean_members H`` (the BALD epistemic term); it is
        clipped at ``0`` to absorb floating-point underflow. ``predictive_variance``
        is the mean over classes of the per-class variance across members
        (a single scalar spread per sample).
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"samples must be (n_members, n_samples, n_classes); got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("samples must contain at least one member")

    mean_proba = arr.mean(axis=0)
    predictive_entropy = _entropy(mean_proba)
    expected_entropy = _entropy(arr).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    predictive_variance = arr.var(axis=0).mean(axis=-1)

    return UncertaintyOutput(
        mean_proba=mean_proba,
        predictive_entropy=predictive_entropy,
        predictive_variance=predictive_variance,
        mutual_information=mutual_information,
    )


class MCDropout:
    """MC Dropout estimator. Applies to a torch MLP only.

    ``fit`` only validates and stores the model (no extra training: the
    dropout-on forward passes happen at ``predict_dist`` time). The wrapped
    module must contain at least one ``nn.Dropout`` layer, otherwise every
    pass is identical and the estimator is meaningless; that is rejected
    eagerly in ``fit``.
    """

    def __init__(self, n_passes: int = 30, *, seed: int = 0) -> None:
        if n_passes < 2:
            raise ValueError(f"n_passes must be >= 2 for a variance estimate; got {n_passes}")
        self.n_passes = n_passes
        self.seed = seed
        self._model: TorchModel | None = None

    def fit(self, model: AuditedModel, X: NDArrayAny, y: NDArrayAny) -> None:
        if not isinstance(model, TorchModel):
            raise TypeError(
                "MCDropout applies to the torch MLP only; "
                f"got a {type(model).__name__} (framework={getattr(model, 'framework', '?')!r})"
            )
        if not model.has_dropout():
            raise ValueError(
                "MCDropout requires the module to contain at least one nn.Dropout "
                "layer; the wrapped module has none, so every pass would be identical."
            )
        self._model = model

    def predict_dist(self, X: NDArrayAny) -> UncertaintyOutput:
        if self._model is None:
            raise RuntimeError("MCDropout.predict_dist called before fit")

        import torch

        # Seed locally so repeated calls (and the test suite) are reproducible
        # without perturbing the global RNG stream.
        gen_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.seed)
            passes = [
                self._model.predict_proba(X, enable_dropout=True) for _ in range(self.n_passes)
            ]
        finally:
            torch.random.set_rng_state(gen_state)

        samples = np.stack(passes, axis=0)
        return aggregate_samples(samples)
