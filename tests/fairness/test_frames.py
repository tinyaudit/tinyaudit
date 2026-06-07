"""Unit and property tests for the per-group MetricFrame helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tinyaudit.fairness._frames import MetricFrame


def _mean(_y_true: np.ndarray | None, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred))


def test_by_group_splits_correctly() -> None:
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    sensitive = np.array(["a", "a", "a", "b", "b", "b"])
    mf = MetricFrame(_mean, y_pred=y_pred, sensitive=sensitive)
    assert mf.by_group == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


def test_difference_is_max_minus_min() -> None:
    mf = MetricFrame(
        _mean,
        y_pred=np.array([1, 1, 0, 0]),
        sensitive=np.array([0, 0, 1, 1]),
    )
    assert mf.difference() == pytest.approx(1.0)


def test_ratio_is_min_over_max() -> None:
    mf = MetricFrame(
        _mean,
        y_pred=np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        sensitive=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0]),
    )
    # group 0: 5/6, group 1: 0/4 -> ratio 0.0
    assert mf.ratio() == pytest.approx(0.0)


def test_ratio_guards_divide_by_zero() -> None:
    mf = MetricFrame(
        _mean,
        y_pred=np.zeros(6, dtype=int),
        sensitive=np.array([0, 0, 0, 1, 1, 1]),
    )
    assert mf.ratio() == 1.0


def test_y_true_passed_through_when_given() -> None:
    seen: list[np.ndarray | None] = []

    def capture(y_true: np.ndarray | None, y_pred: np.ndarray) -> float:
        seen.append(y_true)
        return 0.0

    MetricFrame(
        capture,
        y_pred=np.array([1, 0, 1, 0]),
        sensitive=np.array([0, 0, 1, 1]),
        y_true=np.array([1, 1, 0, 0]),
    )
    assert all(s is not None for s in seen)
    assert len(seen) == 2


def test_y_true_is_none_when_omitted() -> None:
    seen: list[np.ndarray | None] = []

    def capture(y_true: np.ndarray | None, y_pred: np.ndarray) -> float:
        seen.append(y_true)
        return 0.0

    MetricFrame(capture, y_pred=np.array([1, 0]), sensitive=np.array([0, 1]))
    assert seen == [None, None]


def test_accepts_pandas_inputs() -> None:
    mf = MetricFrame(
        _mean,
        y_pred=pd.Series([1, 0, 1, 0]),
        sensitive=pd.Series(["g0", "g0", "g1", "g1"]),
    )
    assert set(mf.by_group) == {"g0", "g1"}


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        MetricFrame(_mean, y_pred=np.array([1, 0, 1]), sensitive=np.array([0, 1]))


def test_handles_more_than_two_groups() -> None:
    mf = MetricFrame(
        _mean,
        y_pred=np.array([1, 0, 1, 0, 1, 1]),
        sensitive=np.array(["a", "b", "c", "a", "b", "c"]),
    )
    assert set(mf.by_group) == {"a", "b", "c"}
    assert 0.0 <= mf.ratio() <= 1.0


@given(
    preds=st.lists(st.integers(0, 1), min_size=2, max_size=80),
    seed=st.integers(0, 10_000),
)
def test_ratio_always_in_unit_interval(preds: list[int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_pred = np.array(preds)
    sensitive = rng.integers(0, 3, size=len(preds))
    mf = MetricFrame(_mean, y_pred=y_pred, sensitive=sensitive)
    r = mf.ratio()
    assert 0.0 <= r <= 1.0


@given(
    preds=st.lists(st.integers(0, 1), min_size=2, max_size=80),
    seed=st.integers(0, 10_000),
)
def test_difference_nonnegative(preds: list[int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    sensitive = rng.integers(0, 3, size=len(preds))
    mf = MetricFrame(_mean, y_pred=np.array(preds), sensitive=sensitive)
    assert mf.difference() >= 0.0
