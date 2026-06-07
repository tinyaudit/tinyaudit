"""Frozen interface for the uncertainty layer.

The estimators (wave 2) and the uncertainty-aware metrics build against
these types. Freezing this early lets the metrics, the card schema, and the
renderer be built in parallel with the estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np

from tinyaudit.models.base import AuditedModel

NDArrayAny: TypeAlias = np.ndarray[Any, Any]


@dataclass
class UncertaintyOutput:
    """Per-sample predictive distribution summary.

    Every array has length ``n_samples``. ``mean_proba`` has shape
    ``(n_samples, n_classes)``.
    """

    mean_proba: NDArrayAny
    predictive_entropy: NDArrayAny
    predictive_variance: NDArrayAny
    mutual_information: NDArrayAny


class UncertaintyEstimator(Protocol):
    """One interface, three implementations (MC dropout, ensemble, early exit)."""

    def fit(self, model: AuditedModel, X: NDArrayAny, y: NDArrayAny) -> None: ...

    def predict_dist(self, X: NDArrayAny) -> UncertaintyOutput: ...
