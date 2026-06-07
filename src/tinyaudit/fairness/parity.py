"""Point-prediction fairness metrics, implemented from scratch.

No Fairlearn in this path; it is used only as a test oracle. Every metric
builds on :class:`MetricFrame` so the per-group split lives in one place.
"""

from __future__ import annotations

import numpy as np

from tinyaudit.fairness._frames import ArrayLike, MetricFrame, NDArrayAny


def _selection_rate(_y_true: NDArrayAny | None, y_pred: NDArrayAny) -> float:
    """Positive-prediction rate. Empty group is treated as rate 0.0."""
    if y_pred.shape[0] == 0:
        return 0.0
    return float(np.mean(y_pred == 1))


def _true_positive_rate(y_true: NDArrayAny | None, y_pred: NDArrayAny) -> float:
    """TPR = P(y_pred == 1 | y_true == 1). No positives => 0.0."""
    assert y_true is not None
    positives = y_true == 1
    n_pos = int(np.count_nonzero(positives))
    if n_pos == 0:
        return 0.0
    return float(np.count_nonzero((y_pred == 1) & positives) / n_pos)


def _false_positive_rate(y_true: NDArrayAny | None, y_pred: NDArrayAny) -> float:
    """FPR = P(y_pred == 1 | y_true == 0). No negatives => 0.0."""
    assert y_true is not None
    negatives = y_true == 0
    n_neg = int(np.count_nonzero(negatives))
    if n_neg == 0:
        return 0.0
    return float(np.count_nonzero((y_pred == 1) & negatives) / n_neg)


def demographic_parity_difference(y_pred: ArrayLike, sensitive: ArrayLike) -> float:
    """Range of the positive-prediction rate across sensitive groups.

    Defined as ``max_a P(y_pred == 1 | A = a) - min_a P(y_pred == 1 | A = a)``.
    Lies in ``[0, 1]``; ``0.0`` means every group has the same selection rate
    (perfect demographic parity). Ground truth is not used.
    """
    frame = MetricFrame(_selection_rate, y_pred=y_pred, sensitive=sensitive)
    return frame.difference()


def equalized_odds_difference(y_true: ArrayLike, y_pred: ArrayLike, sensitive: ArrayLike) -> float:
    """Worst-case equalized-odds violation across sensitive groups.

    Computes the across-group range of the true-positive rate and the
    across-group range of the false-positive rate, and returns the larger of
    the two::

        max( range_a TPR_a , range_a FPR_a )

    Lies in ``[0, 1]``; ``0.0`` means all groups share both TPR and FPR. This
    matches ``fairlearn.metrics.equalized_odds_difference`` with its defaults
    (``agg="worst_case"``, ``method="between_groups"``).
    """
    tpr = MetricFrame(
        _true_positive_rate, y_pred=y_pred, sensitive=sensitive, y_true=y_true
    ).difference()
    fpr = MetricFrame(
        _false_positive_rate, y_pred=y_pred, sensitive=sensitive, y_true=y_true
    ).difference()
    return float(max(tpr, fpr))


def disparate_impact_ratio(y_pred: ArrayLike, sensitive: ArrayLike) -> float:
    """Disparate-impact ratio of the positive-prediction rate.

    DIRECTION CONVENTION (pinned here and in ``test_parity.py``; do not change
    one without the other):

    * Favored outcome: the positive class, ``y_pred == 1``.
    * Numerator: the *minimum* group selection rate.
    * Denominator: the *maximum* group selection rate.

    So ``DI = min_a P(y_pred == 1 | A = a) / max_a P(y_pred == 1 | A = a)``,
    which always lies in ``[0, 1]``; ``1.0`` means perfect parity and smaller
    values mean a larger disparity (the classic 0.8 "four-fifths rule"
    threshold applies directly). This min/max framing is symmetric in the
    group labels, so it is unaffected by which group is called "0" or "1" --
    that symmetry is exactly why it is the easiest place in the system to ship
    a silent bug, hence the pin.

    When the maximum selection rate is ``0`` (no group ever predicts
    positive), DI is defined as ``1.0`` (degenerate parity) rather than
    ``0/0``.
    """
    frame = MetricFrame(_selection_rate, y_pred=y_pred, sensitive=sensitive)
    return frame.ratio()
