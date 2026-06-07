"""scikit-learn adapter conforming to the :class:`AuditedModel` protocol.

Wraps a fitted estimator so the rest of the pipeline depends only on the
protocol, never on scikit-learn directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Attributes scanned by the generic ``n_params`` fallback, in priority order.
_GENERIC_PARAM_ATTRS = ("coef_", "intercept_", "coefs_", "intercepts_", "feature_log_prob_")


def _to_array(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Accept an ndarray or a DataFrame and return a 2-D ndarray."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return np.asarray(X)


class SklearnModel:
    """An :class:`AuditedModel` backed by a fitted scikit-learn estimator.

    The estimator must already be fitted; ``predict`` and ``predict_proba``
    delegate straight to it and always return ``np.ndarray``.
    """

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator

    @property
    def estimator(self) -> Any:
        """The wrapped fitted scikit-learn estimator."""
        return self._estimator

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.asarray(self._estimator.predict(_to_array(X)))

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.asarray(self._estimator.predict_proba(_to_array(X)))

    @property
    def framework(self) -> str:
        return "sklearn"

    @property
    def n_params(self) -> int:
        """Total number of learnable parameters.

        Exact per estimator family:

        * ``LogisticRegression``: ``coef_.size + intercept_.size``.
        * ``DecisionTreeClassifier``: ``tree_.node_count``.
        * ``MLPClassifier``: summed sizes of ``coefs_`` and ``intercepts_``.

        Generic fallback for any other estimator: the summed size of whichever
        of the common fitted weight attributes
        (``coef_``, ``intercept_``, ``coefs_``, ``intercepts_``,
        ``feature_log_prob_``) are present. If none are present a
        ``TypeError`` is raised naming the estimator class, since a parameter
        count is undefined for it.
        """
        est = self._estimator
        cls_name = type(est).__name__

        if cls_name == "LogisticRegression":
            return int(np.asarray(est.coef_).size + np.asarray(est.intercept_).size)

        if cls_name == "DecisionTreeClassifier":
            return int(est.tree_.node_count)

        if cls_name == "MLPClassifier":
            coefs = sum(int(np.asarray(c).size) for c in est.coefs_)
            intercepts = sum(int(np.asarray(b).size) for b in est.intercepts_)
            return coefs + intercepts

        total = 0
        found = False
        for attr in _GENERIC_PARAM_ATTRS:
            value = getattr(est, attr, None)
            if value is None:
                continue
            found = True
            if isinstance(value, (list, tuple)):
                total += sum(int(np.asarray(v).size) for v in value)
            else:
                total += int(np.asarray(value).size)

        if not found:
            raise TypeError(
                f"Cannot determine n_params for estimator of type {cls_name!r}: "
                "none of the known fitted weight attributes "
                f"{_GENERIC_PARAM_ATTRS} are present. Provide an explicit "
                "adapter for this estimator family."
            )
        return total
