"""Tests for the UCI Adult loader.

These exercise the real dataset and may use the network on a cold cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

from tinyaudit.data import load_adult
from tinyaudit.data.adult import _split_indices


@pytest.fixture(scope="module")
def test_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    return load_adult(split="test", seed=0)


def test_shapes_and_columns(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features, label, sensitive = test_split

    assert len(features) == len(label) == len(sensitive)
    assert len(features) > 0
    assert features.shape[1] > 0

    assert all(ptypes.is_numeric_dtype(features[c]) for c in features.columns)
    assert "sex" not in features.columns
    assert "race" not in features.columns

    assert label.dtype == "int64"
    assert set(label.unique().tolist()) == {0, 1}

    assert list(sensitive.columns) == ["sex", "race"]
    assert set(sensitive["sex"].unique()) == {"Female", "Male"}
    assert sensitive["race"].nunique() > 2


def test_no_nans(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features, label, sensitive = test_split
    assert not features.isna().any().any()
    assert not label.isna().any()
    assert not sensitive.isna().any().any()


def test_determinism(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features_a, label_a, sensitive_a = test_split
    features_b, label_b, sensitive_b = load_adult(split="test", seed=0)

    pd.testing.assert_frame_equal(features_a, features_b)
    pd.testing.assert_series_equal(label_a, label_b)
    pd.testing.assert_frame_equal(sensitive_a, sensitive_b)


def test_split_indices_are_an_exact_disjoint_partition() -> None:
    # Direct, network-free check of the split contract: positions are
    # partitioned, never shared, and the partition is seed-deterministic.
    label = pd.Series([0, 1] * 500, name="income")
    train_idx, test_idx = _split_indices(label, seed=0)

    assert set(train_idx).isdisjoint(set(test_idx))
    assert sorted(np.concatenate([train_idx, test_idx])) == list(range(len(label)))

    train_idx_b, test_idx_b = _split_indices(label, seed=0)
    np.testing.assert_array_equal(train_idx, train_idx_b)
    np.testing.assert_array_equal(test_idx, test_idx_b)

    train_idx_c, _ = _split_indices(label, seed=1)
    assert not np.array_equal(train_idx, train_idx_c)


def test_train_test_disjoint_and_partition() -> None:
    f_train, _, s_train = load_adult(split="train", seed=0)
    f_test, _, s_test = load_adult(split="test", seed=0)

    # Adult has 52 fully-duplicated source rows, so a raw content-overlap
    # check would see the same tuple on both sides even though the split is
    # index-disjoint. Any shared unique tuple must be a known source
    # duplicate, never a leaked split position.
    full = pd.concat([f_train, s_train], axis=1)
    other = pd.concat([f_test, s_test], axis=1)
    full_unique = {tuple(r) for r in full.drop_duplicates().to_numpy()}
    other_unique = {tuple(r) for r in other.drop_duplicates().to_numpy()}
    shared = full_unique & other_unique
    combined = pd.concat([full, other], ignore_index=True)
    assert len(shared) <= int(combined.duplicated(keep=False).sum())

    assert len(f_train) > len(f_test)
    assert set(f_train.columns) == set(f_test.columns)
    assert len(f_train) + len(f_test) > 40000
    assert abs(len(f_test) / (len(f_train) + len(f_test)) - 0.25) < 0.01


def test_seed_changes_split() -> None:
    f0, _, _ = load_adult(split="test", seed=0)
    f1, _, _ = load_adult(split="test", seed=1)

    assert f0.shape == f1.shape
    assert not f0.equals(f1)


def test_invalid_split_raises() -> None:
    with pytest.raises(ValueError, match="split must be"):
        load_adult(split="validation", seed=0)
