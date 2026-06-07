"""UCI Adult Income loader.

Deterministic preprocessing for the UCI Adult Income benchmark. This module is
the single source of truth for how Adult is prepared: missing-value handling,
categorical encoding, the train/test split, and schema validation all live
here, in code.

Acquisition order:

1. ``sklearn.datasets.fetch_openml("adult", version=2, as_frame=True)`` -- the
   canonical OpenML copy (48842 rows, 14 features + target). sklearn caches it
   under ``~/scikit_learn_data``.
2. Fallback: ``fetch_openml(data_id=1590, as_frame=True)``, the same dataset
   addressed by its OpenML data id (used when name resolution is unavailable).
3. Fallback: the canonical UCI ``adult-all.csv`` mirror on GitHub
   (``jbrownlee/Datasets``). It carries the identical 48842 rows in the
   original UCI column order with ``?`` for missing values, so it yields the
   same preprocessed frame as the OpenML copies. Used only when OpenML is
   unreachable (e.g. restricted-network CI).

Whichever source succeeds, the rows are normalized to one schema before
preprocessing, so the returned frames do not depend on the source.
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

_RAW_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

_NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

_CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "native-country",
]

_SENSITIVE_COLUMNS = ["sex", "race"]

_MISSING_TOKEN = "?"
_NA_FILL = "Missing"

_GITHUB_MIRROR = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"


def _normalize_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce any accepted source into the canonical raw schema.

    Handles the two OpenML quirks: the target column may be named ``class``,
    and string cells may carry surrounding whitespace or the literal ``?``
    missing token.
    """
    frame = frame.copy()
    if "class" in frame.columns and "income" not in frame.columns:
        frame = frame.rename(columns={"class": "income"})
    missing = [c for c in _RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Adult source is missing expected columns: {missing}")
    frame = frame[_RAW_COLUMNS]

    obj_cols = _CATEGORICAL_COLUMNS + _SENSITIVE_COLUMNS + ["income"]
    for col in obj_cols:
        series = frame[col].astype("string").str.strip()
        series = series.replace({_MISSING_TOKEN: pd.NA})
        frame[col] = series
    for col in _NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _fetch_raw() -> pd.DataFrame:
    """Return the raw Adult frame from the first source that succeeds."""
    try:
        bunch = fetch_openml("adult", version=2, as_frame=True)
        return _normalize_raw(bunch.frame)
    except Exception:
        pass
    try:
        bunch = fetch_openml(data_id=1590, as_frame=True)
        return _normalize_raw(bunch.frame)
    except Exception:
        pass
    with urllib.request.urlopen(_GITHUB_MIRROR, timeout=60) as response:
        payload = response.read()
    frame = pd.read_csv(
        io.BytesIO(payload),
        header=None,
        names=_RAW_COLUMNS,
        skipinitialspace=True,
    )
    return _normalize_raw(frame)


def _preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Turn the raw frame into (features, label, sensitive).

    Deterministic and order-stable: missing categoricals become an explicit
    ``"Missing"`` level, numeric columns have no missing values in Adult,
    categoricals are one-hot encoded with sorted column names, and the raw
    ``sex``/``race`` columns are held out of the features.
    """
    raw = raw.dropna(subset=["income"]).reset_index(drop=True)

    label = (
        raw["income"]
        .str.replace(".", "", regex=False)
        .map({"<=50K": 0, ">50K": 1})
        .astype("int64")
        .rename("income")
    )
    if label.isna().any():
        raise ValueError("Adult income column has values outside {<=50K, >50K}.")

    sensitive = raw[_SENSITIVE_COLUMNS].copy()
    sensitive["sex"] = sensitive["sex"].astype("object")
    sensitive["race"] = sensitive["race"].astype("object")
    if sensitive.isna().any().any():
        raise ValueError("Adult sensitive columns contain missing values.")

    numeric = raw[_NUMERIC_COLUMNS].astype("int64")
    if numeric.isna().any().any():
        raise ValueError("Adult numeric columns contain missing values.")

    categorical = raw[_CATEGORICAL_COLUMNS].astype("object").fillna(_NA_FILL)
    encoded = pd.get_dummies(categorical, columns=_CATEGORICAL_COLUMNS, dtype="int64")
    encoded = encoded.reindex(sorted(encoded.columns), axis=1)

    features = pd.concat([numeric, encoded], axis=1)
    features.index = raw.index
    label.index = raw.index
    sensitive.index = raw.index
    return features, label, sensitive


def _validate(features: pd.DataFrame, label: pd.Series, sensitive: pd.DataFrame) -> None:
    """Assert the contract: numeric features, binary label, clean sensitive."""
    if features.isna().any().any():
        raise ValueError("Adult features contain NaNs after preprocessing.")
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(f"Adult features have non-numeric columns: {non_numeric}")

    if label.dtype != "int64":
        raise ValueError(f"Adult label dtype must be int64, got {label.dtype}.")
    label_values = set(label.unique().tolist())
    if not label_values <= {0, 1} or not label_values:
        raise ValueError(f"Adult label is not binary 0/1: {sorted(label_values)}.")
    if len(label_values) != 2:
        raise ValueError("Adult label must contain both classes.")

    if list(sensitive.columns) != _SENSITIVE_COLUMNS:
        raise ValueError(
            f"sensitive columns must be {_SENSITIVE_COLUMNS}, " f"got {list(sensitive.columns)}."
        )
    if sensitive.isna().any().any():
        raise ValueError("Adult sensitive columns contain NaNs.")
    if sensitive["race"].nunique() < 2:
        raise ValueError("Adult race column is not multi-valued.")


def _split_indices(label: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, label-stratified 75/25 positional split.

    Returns ``(train_idx, test_idx)`` over ``label``'s 0..n-1 positions. Same
    seed yields the same partition; the two arrays are disjoint and together
    cover every row.
    """
    idx = np.arange(len(label))
    return train_test_split(idx, test_size=0.25, random_state=seed, stratify=label.to_numpy())


def load_adult(split: str = "test", seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (features, label, sensitive) for UCI Adult Income.

    features: model input columns, sensitive columns excluded.
    label: binary int64 Series, 1 == income > 50K.
    sensitive: DataFrame with exactly columns ['sex', 'race'], raw values.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    raw = _fetch_raw()
    features, label, sensitive = _preprocess(raw)
    _validate(features, label, sensitive)

    train_idx, test_idx = _split_indices(label, seed)
    keep = train_idx if split == "train" else test_idx

    features = features.iloc[keep].reset_index(drop=True)
    label = label.iloc[keep].reset_index(drop=True)
    sensitive = sensitive.iloc[keep].reset_index(drop=True)
    return features, label, sensitive
