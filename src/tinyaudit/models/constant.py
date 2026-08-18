"""The constant-predictor baseline.

A :class:`ConstantModel` ignores its input entirely: every row gets the same
prediction and the same probability vector. That makes it useless as a
classifier and *perfect* on any group-parity metric, since the selection rate
is identical in every sensitive group by construction. It is the reference
point the fairness argument leans on -- a model can score 0.0 demographic
parity difference and 1.0 disparate impact while carrying no information at
all -- so it is a first-class adapter, not a test double.

Two modes:

``"majority"``
    ``predict_proba`` is a hard one-hot on the most frequent training class.

``"prevalence"``
    ``predict_proba`` is the training base rate, repeated for every row.
    ``predict`` is the argmax of that, i.e. still the majority class.

Both modes therefore produce identical hard predictions; they differ only in
how confident the probabilities are, which is what makes the pair useful for
calibration and uncertainty experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

NDArrayAny: TypeAlias = np.ndarray[Any, Any]

Mode = Literal["majority", "prevalence"]

_MODES: tuple[Mode, ...] = ("majority", "prevalence")


def _n_rows(X: NDArrayAny | pd.DataFrame | Sequence[Any]) -> int:
    """Number of samples in ``X``, which is otherwise ignored."""
    if isinstance(X, pd.DataFrame):
        return int(X.shape[0])
    arr = np.asarray(X)
    if arr.ndim == 0:
        raise ValueError("X must be at least 1-D; got a scalar")
    return int(arr.shape[0])


class ConstantModel:
    """An :class:`AuditedModel` whose output does not depend on ``X``.

    Parameters
    ----------
    classes:
        The class labels, in the column order of :meth:`predict_proba`. They
        are sorted on construction so the order matches scikit-learn's
        ``classes_`` convention, and ``prevalence`` is reordered with them.
        Labels keep their original dtype, so ``predict`` returns e.g. numpy
        ``int64`` 0/1 rather than positional indices.
    prevalence:
        Per-class training base rate, aligned with ``classes`` *before*
        sorting. Must be non-negative and sum to 1.
    mode:
        ``"majority"`` for one-hot probabilities on the most frequent class,
        ``"prevalence"`` for the base rate itself.

    Prefer :meth:`from_labels` to derive both from a label array.
    """

    def __init__(
        self,
        classes: Sequence[Any] | NDArrayAny,
        prevalence: Sequence[float] | NDArrayAny,
        mode: Mode = "majority",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")

        class_arr = np.asarray(classes)
        prev_arr = np.asarray(prevalence, dtype=float)

        if class_arr.ndim != 1 or class_arr.shape[0] == 0:
            raise ValueError("classes must be a non-empty 1-D array of labels")
        if prev_arr.shape != class_arr.shape:
            raise ValueError(
                f"prevalence must have the same length as classes; "
                f"got {prev_arr.shape} vs {class_arr.shape}"
            )
        if len(np.unique(class_arr)) != class_arr.shape[0]:
            raise ValueError("classes must not contain duplicates")
        if not np.all(np.isfinite(prev_arr)) or np.any(prev_arr < 0.0):
            raise ValueError("prevalence must be finite and non-negative")
        total = float(prev_arr.sum())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"prevalence must sum to 1.0; got {total!r}")

        # Sort for a deterministic, sklearn-compatible column order.
        order = np.argsort(class_arr, kind="stable")
        self._classes: NDArrayAny = class_arr[order]
        self._prevalence: NDArrayAny = prev_arr[order]
        self._mode: Mode = mode
        # Ties go to the lowest-sorted label, which argmax already does.
        self._majority_index = int(np.argmax(self._prevalence))

    @classmethod
    def from_labels(
        cls,
        y: Sequence[Any] | NDArrayAny | pd.Series,
        mode: Mode = "majority",
    ) -> ConstantModel:
        """Build a constant predictor from a 1-D array of training labels.

        The class list is the sorted unique labels and the prevalence is their
        empirical frequency, so ``from_labels(y).predict_proba(X)[0]`` is the
        training base rate (e.g. ``[0.76, 0.24]`` on Adult).
        """
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y_arr.shape}")
        if y_arr.shape[0] == 0:
            raise ValueError("y must contain at least one label")

        labels, counts = np.unique(y_arr, return_counts=True)
        prevalence = counts.astype(float) / float(counts.sum())
        return cls(labels, prevalence, mode=mode)

    # --- introspection -------------------------------------------------- #

    @property
    def classes_(self) -> NDArrayAny:
        """Sorted class labels, matching the ``predict_proba`` column order."""
        return self._classes.copy()

    @property
    def prevalence(self) -> NDArrayAny:
        """Training base rate per class, aligned with :attr:`classes_`."""
        return self._prevalence.copy()

    @property
    def mode(self) -> Mode:
        """``"majority"`` or ``"prevalence"``."""
        return self._mode

    @property
    def constant_class(self) -> Any:
        """The single label this model always predicts, in its original dtype."""
        return self._classes[self._majority_index]

    # --- AuditedModel protocol ------------------------------------------ #

    def predict(self, X: NDArrayAny | pd.DataFrame) -> NDArrayAny:
        """The majority class, repeated once per row of ``X``.

        Returns the *original* label (dtype preserved), not a positional
        index, so it can be compared directly against ``y_true``.
        """
        n = _n_rows(X)
        return np.full(n, self.constant_class, dtype=self._classes.dtype)

    def predict_proba(self, X: NDArrayAny | pd.DataFrame) -> NDArrayAny:
        """Shape ``(n_samples, n_classes)``; every row identical.

        One-hot on the majority class in ``"majority"`` mode, the training
        base rate in ``"prevalence"`` mode.
        """
        n = _n_rows(X)
        if self._mode == "majority":
            row = np.zeros(self._classes.shape[0], dtype=float)
            row[self._majority_index] = 1.0
        else:
            row = self._prevalence
        return np.tile(row, (n, 1))

    @property
    def n_params(self) -> int:
        """One: the stored constant."""
        return 1

    @property
    def framework(self) -> str:
        return "constant"

    def __repr__(self) -> str:
        prev = ", ".join(f"{p:.4g}" for p in self._prevalence)
        return (
            f"ConstantModel(mode={self._mode!r}, "
            f"classes={self._classes.tolist()!r}, prevalence=[{prev}])"
        )
