"""The model contract every adapter conforms to.

Metrics, estimators, explainers, and the profiler depend only on this
protocol, never on a concrete framework.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AuditedModel(Protocol):
    """A trained classifier the pipeline can audit.

    Implementations wrap a fitted estimator. ``predict`` returns hard class
    labels; ``predict_proba`` returns per-class probabilities with shape
    ``(n_samples, n_classes)``.
    """

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def n_params(self) -> int:
        """Total number of learnable parameters."""

    @property
    def framework(self) -> str:
        """One of ``"sklearn"``, ``"torch"``, or ``"onnx"``."""
