"""ACSIncome (Folktables) loader.

Deterministic preprocessing for the ACS Income benchmark from the Folktables
library. This module is the single source of truth for how ACSIncome is
prepared: acquisition, missing-value handling, feature selection, the
train/test split, and schema validation all live here, in code.

Acquisition order:

1. ``folktables`` package (``ACSDataSource`` + ``ACSIncome``) -- the canonical
   source. Downloads 2018 1-year ACS data for California (state FIPS 06) and
   caches locally.
2. Fallback: GitHub CSV mirror at the folktables repository's data directory.
   Used when the ``folktables`` package is not installed and the mirror is
   reachable.
3. Synthetic fallback: a deterministic synthetic dataset that mirrors the
   ACSIncome schema (same column names, dtypes, ~1000 rows). Used only when
   neither real source is available (e.g. restricted-network CI). Prints a
   warning when activated.
"""

from __future__ import annotations

import io
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ACSIncome feature columns as defined in the Folktables paper.
_ACSINCOME_FEATURES = [
    "AGEP",  # age
    "COW",  # class of worker
    "SCHL",  # educational attainment
    "MAR",  # marital status
    "OCCP",  # occupation
    "POBP",  # place of birth
    "RELP",  # relationship
    "WKHP",  # usual hours worked per week
    "SEX",
    "RAC1P",
]
_ACSINCOME_TARGET = "PINCP"  # personal income

_SENSITIVE_COLUMNS = ["sex", "race"]

_SEX_MAP: dict[int, str] = {1: "Male", 2: "Female"}
_RACE_MAP: dict[int, str] = {
    1: "White",
    2: "Black",
    3: "AIAN",
    4: "Alaska Native",
    5: "AIAN+",
    6: "Asian",
    7: "NHOPI",
    8: "Other",
    9: "Mixed",
}

# Features that go into the model (SEX and RAC1P are held out).
_FEATURE_COLUMNS = [c for c in _ACSINCOME_FEATURES if c not in ("SEX", "RAC1P")]

_GITHUB_MIRROR = (
    "https://raw.githubusercontent.com/socialfoundations/folktables/"
    "main/folktables/data/adult.csv"
)


def _fetch_via_package() -> pd.DataFrame:
    """Fetch ACSIncome via the folktables package."""
    from folktables import ACSDataSource, ACSIncome  # type: ignore[import]

    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, _ = ACSIncome.df_to_pandas(acs_data)
    # ACSIncome.df_to_pandas returns features as a DataFrame with the feature
    # columns, label as a Series/array for PINCP>50000.  Reconstruct a full
    # frame so that preprocessing is uniform.
    frame = features.copy()
    frame[_ACSINCOME_TARGET] = label
    return frame


def _fetch_via_mirror() -> pd.DataFrame:
    """Fetch from the GitHub mirror as a last resort before synthetic."""
    with urllib.request.urlopen(_GITHUB_MIRROR, timeout=60) as response:
        payload = response.read()
    frame = pd.read_csv(io.BytesIO(payload))
    return frame


def _make_synthetic(seed: int) -> pd.DataFrame:
    """Return a small deterministic synthetic frame that mirrors ACSIncome."""
    warnings.warn(
        "ACSIncome real data is unavailable (folktables package not installed "
        "and network download failed). Falling back to a synthetic dataset. "
        "Install folktables or enable network access for real results.",
        UserWarning,
        stacklevel=4,
    )
    rng = np.random.default_rng(seed)
    n = 1000
    frame = pd.DataFrame(
        {
            "AGEP": rng.integers(18, 90, size=n).astype(float),
            "COW": rng.integers(1, 9, size=n).astype(float),
            "SCHL": rng.integers(1, 25, size=n).astype(float),
            "MAR": rng.integers(1, 5, size=n).astype(float),
            "OCCP": rng.integers(10, 9800, size=n).astype(float),
            "POBP": rng.integers(1, 100, size=n).astype(float),
            "RELP": rng.integers(0, 17, size=n).astype(float),
            "WKHP": rng.integers(1, 99, size=n).astype(float),
            "SEX": rng.integers(1, 3, size=n).astype(float),
            "RAC1P": rng.integers(1, 10, size=n).astype(float),
            _ACSINCOME_TARGET: rng.exponential(40000, size=n),
        }
    )
    return frame


def _fetch_raw(seed: int) -> pd.DataFrame:
    """Return a raw frame from the first source that succeeds."""
    try:
        return _fetch_via_package()
    except Exception:
        pass
    try:
        return _fetch_via_mirror()
    except Exception:
        pass
    return _make_synthetic(seed)


def _preprocess(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Turn the raw ACSIncome frame into (features, label, sensitive).

    Deterministic and order-stable: the target column is binarised at $50k,
    SEX and RAC1P are mapped to human-readable strings and held out of features,
    all remaining feature columns are forced to float64, and any NaNs are filled
    with the per-column median.
    """
    frame = frame.copy()

    # Ensure mandatory columns exist.
    missing_cols = [c for c in _ACSINCOME_FEATURES + [_ACSINCOME_TARGET] if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"ACSIncome source is missing expected columns: {missing_cols}")

    # Binary label: income > $50k.
    label = (pd.to_numeric(frame[_ACSINCOME_TARGET], errors="coerce") > 50_000).astype("int64")
    label.name = "income"

    # Sensitive attributes.
    sex_raw = pd.to_numeric(frame["SEX"], errors="coerce").round().astype("Int64")
    race_raw = pd.to_numeric(frame["RAC1P"], errors="coerce").round().astype("Int64")

    sensitive = pd.DataFrame(
        {
            "sex": sex_raw.map(_SEX_MAP).astype("object"),
            "race": race_raw.map(_RACE_MAP).astype("object"),
        }
    )

    # Drop rows with missing label or missing sensitive values.
    valid_mask = label.notna() & sensitive["sex"].notna() & sensitive["race"].notna()
    frame = frame.loc[valid_mask].reset_index(drop=True)
    label = label.loc[valid_mask].reset_index(drop=True)
    sensitive = sensitive.loc[valid_mask].reset_index(drop=True)

    # Features: numeric only; median imputation for NaNs.
    features = frame[_FEATURE_COLUMNS].copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    median_vals = features.median()
    features = features.fillna(median_vals)
    features = features.astype("float64")

    return features, label, sensitive


def _validate(features: pd.DataFrame, label: pd.Series, sensitive: pd.DataFrame) -> None:
    """Assert the loader contract: numeric features, binary label, clean sensitive."""
    if features.isna().any().any():
        raise ValueError("ACSIncome features contain NaNs after preprocessing.")
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(f"ACSIncome features have non-numeric columns: {non_numeric}")

    if label.dtype != "int64":
        raise ValueError(f"ACSIncome label dtype must be int64, got {label.dtype}.")
    label_values = set(label.unique().tolist())
    if not label_values <= {0, 1} or not label_values:
        raise ValueError(f"ACSIncome label is not binary 0/1: {sorted(label_values)}.")

    if list(sensitive.columns) != _SENSITIVE_COLUMNS:
        raise ValueError(
            f"sensitive columns must be {_SENSITIVE_COLUMNS}, " f"got {list(sensitive.columns)}."
        )
    if sensitive.isna().any().any():
        raise ValueError("ACSIncome sensitive columns contain NaNs.")
    if sensitive["sex"].nunique() < 2:
        raise ValueError("ACSIncome sex column must be multi-valued.")


def _split_indices(label: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, label-stratified 75/25 positional split.

    Returns ``(train_idx, test_idx)`` over ``label``'s 0..n-1 positions. Same
    seed yields the same partition; the two arrays are disjoint and together
    cover every row.
    """
    idx = np.arange(len(label))
    return train_test_split(idx, test_size=0.25, random_state=seed, stratify=label.to_numpy())


def load_folktables(
    split: str = "test", seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (features, label, sensitive) for ACSIncome (Folktables).

    features: model input columns, sensitive columns excluded.
    label: binary int64 Series, 1 == personal income > $50k.
    sensitive: DataFrame with exactly columns ['sex', 'race'].

    Parameters
    ----------
    split:
        ``'train'`` or ``'test'``. Test set is 25 % of the data.
    seed:
        Random seed for the stratified split.  Same seed always yields the
        same partition.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    raw = _fetch_raw(seed)
    features, label, sensitive = _preprocess(raw)
    _validate(features, label, sensitive)

    train_idx, test_idx = _split_indices(label, seed)
    keep = train_idx if split == "train" else test_idx

    features = features.iloc[keep].reset_index(drop=True)
    label = label.iloc[keep].reset_index(drop=True)
    sensitive = sensitive.iloc[keep].reset_index(drop=True)
    return features, label, sensitive
