"""Per-group split-and-aggregate helper.

:class:`MetricFrame` evaluates a metric independently on each sensitive
group and exposes scalar disparity reductions. The parity metrics build on
this so they stay short.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

NDArrayAny: TypeAlias = np.ndarray[Any, Any]
ArrayLike: TypeAlias = NDArrayAny | pd.Series | list[object]

# A metric is called as ``metric(y_true, y_pred)``; ``y_true`` may be None for
# metrics that do not need ground truth (e.g. selection rate).
MetricFn = Callable[[NDArrayAny | None, NDArrayAny], float]


def _to_1d(values: ArrayLike) -> NDArrayAny:
    arr = np.asarray(values)
    return arr.ravel()


class MetricFrame:
    """Split arrays by sensitive group and aggregate a metric per group.

    Parameters
    ----------
    metric:
        Callable ``metric(y_true, y_pred) -> float``. ``y_true`` is passed as
        ``None`` when ``y_true`` was not supplied.
    y_pred:
        Predicted labels, one per sample.
    sensitive:
        Group label per sample. Any hashable values; need not be binary.
    y_true:
        Optional ground-truth labels, one per sample.
    """

    def __init__(
        self,
        metric: MetricFn,
        *,
        y_pred: ArrayLike,
        sensitive: ArrayLike,
        y_true: ArrayLike | None = None,
    ) -> None:
        pred = _to_1d(y_pred)
        groups = _to_1d(sensitive)
        truth = None if y_true is None else _to_1d(y_true)

        if pred.shape[0] != groups.shape[0]:
            raise ValueError(
                f"y_pred and sensitive length mismatch: {pred.shape[0]} vs {groups.shape[0]}"
            )
        if truth is not None and truth.shape[0] != pred.shape[0]:
            raise ValueError(
                f"y_true and y_pred length mismatch: {truth.shape[0]} vs {pred.shape[0]}"
            )

        by_group: dict[Hashable, float] = {}
        for g in pd.unique(groups):
            mask = groups == g
            g_true = None if truth is None else truth[mask]
            by_group[g] = float(metric(g_true, pred[mask]))

        self._by_group = by_group

    @property
    def by_group(self) -> dict[Hashable, float]:
        """Mapping of group label to that group's metric value."""
        return self._by_group

    def _values(self) -> list[float]:
        if not self._by_group:
            raise ValueError("MetricFrame has no groups to aggregate over")
        return list(self._by_group.values())

    def difference(self) -> float:
        """Max minus min of the per-group metric. Zero means parity."""
        values = self._values()
        return float(max(values) - min(values))

    def ratio(self) -> float:
        """Min divided by max of the per-group metric, in ``[0, 1]``.

        Returns ``1.0`` when the maximum is zero (all groups equal at zero,
        i.e. perfect parity) so the result stays well defined.
        """
        values = self._values()
        hi = max(values)
        lo = min(values)
        if hi == 0.0:
            return 1.0
        return float(lo / hi)
