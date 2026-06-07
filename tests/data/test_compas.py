"""Tests for the ProPublica COMPAS loader.

These tests pass even when the real COMPAS data is unavailable; the loader
falls back to a deterministic synthetic dataset in that case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

from tinyaudit.data import load_compas
from tinyaudit.data.compas import _split_indices


@pytest.fixture(scope="module")
def test_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    return load_compas(split="test", seed=0)


@pytest.fixture(scope="module")
def train_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    return load_compas(split="train", seed=0)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


def test_return_types(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features, label, sensitive = test_split
    assert isinstance(features, pd.DataFrame)
    assert isinstance(label, pd.Series)
    assert isinstance(sensitive, pd.DataFrame)


def test_lengths_match(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features, label, sensitive = test_split
    assert len(features) == len(label) == len(sensitive)
    assert len(features) > 0


def test_features_are_numeric(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features, _, _ = test_split
    assert features.shape[1] > 0
    non_numeric = [c for c in features.columns if not ptypes.is_numeric_dtype(features[c])]
    assert non_numeric == [], f"Non-numeric feature columns: {non_numeric}"


def test_sensitive_columns_not_in_features(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features, _, _ = test_split
    assert "sex" not in features.columns
    assert "race" not in features.columns


def test_label_is_binary(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    _, label, _ = test_split
    assert label.dtype == "int64"
    assert set(label.unique().tolist()) <= {0, 1}
    assert len(set(label.unique().tolist())) == 2


def test_sensitive_columns(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    _, _, sensitive = test_split
    assert list(sensitive.columns) == ["sex", "race"]


def test_sensitive_sex_values(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    _, _, sensitive = test_split
    sex_values = set(sensitive["sex"].unique())
    assert sex_values <= {"Male", "Female"}
    assert len(sex_values) == 2, f"Expected both Male and Female, got {sex_values}"


def test_compas_race_spot_check(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    """Per CLAUDE.md: African-American and Caucasian must both be present."""
    _, _, sensitive = test_split
    race_values = sensitive["race"].unique()
    assert (
        "African-American" in race_values
    ), f"'African-American' not found in race column; got {sorted(race_values)}"
    assert (
        "Caucasian" in race_values
    ), f"'Caucasian' not found in race column; got {sorted(race_values)}"


def test_sensitive_race_is_multi_valued(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    _, _, sensitive = test_split
    assert sensitive["race"].nunique() >= 2


def test_no_nans(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features, label, sensitive = test_split
    assert not features.isna().any().any()
    assert not label.isna().any()
    assert not sensitive.isna().any().any()


def test_expected_feature_columns(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features, _, _ = test_split
    expected = {"age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"}
    assert set(features.columns) == expected


# ---------------------------------------------------------------------------
# Determinism / reproducibility tests
# ---------------------------------------------------------------------------


def test_determinism(
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    features_a, label_a, sensitive_a = test_split
    features_b, label_b, sensitive_b = load_compas(split="test", seed=0)

    pd.testing.assert_frame_equal(features_a, features_b)
    pd.testing.assert_series_equal(label_a, label_b)
    pd.testing.assert_frame_equal(sensitive_a, sensitive_b)


def test_seed_changes_split() -> None:
    f0, _, _ = load_compas(split="test", seed=0)
    f1, _, _ = load_compas(split="test", seed=1)

    assert f0.shape == f1.shape
    assert not f0.equals(f1)


# ---------------------------------------------------------------------------
# Split integrity tests
# ---------------------------------------------------------------------------


def test_train_test_sizes(
    train_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    f_train, _, _ = train_split
    f_test, _, _ = test_split

    total = len(f_train) + len(f_test)
    assert total > 0
    assert len(f_train) > len(f_test)
    assert abs(len(f_test) / total - 0.25) < 0.02


def test_train_test_same_columns(
    train_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    f_train, _, _ = train_split
    f_test, _, _ = test_split
    assert set(f_train.columns) == set(f_test.columns)


def test_split_indices_partition() -> None:
    label = pd.Series([0, 1] * 500, name="two_year_recid")
    train_idx, test_idx = _split_indices(label, seed=0)

    assert set(train_idx).isdisjoint(set(test_idx))
    assert sorted(np.concatenate([train_idx, test_idx])) == list(range(len(label)))

    # Reproducible.
    train_idx_b, test_idx_b = _split_indices(label, seed=0)
    np.testing.assert_array_equal(train_idx, train_idx_b)
    np.testing.assert_array_equal(test_idx, test_idx_b)

    # Different seed → different partition.
    train_idx_c, _ = _split_indices(label, seed=1)
    assert not np.array_equal(train_idx, train_idx_c)


def test_invalid_split_raises() -> None:
    with pytest.raises(ValueError, match="split must be"):
        load_compas(split="validation", seed=0)
