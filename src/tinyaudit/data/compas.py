"""ProPublica COMPAS Recidivism loader.

Deterministic preprocessing for the ProPublica COMPAS two-year recidivism
benchmark.  This module is the single source of truth for how COMPAS is
prepared: acquisition, filtering, feature selection, the train/test split, and
schema validation all live here, in code.

Acquisition order:

1. ProPublica GitHub CSV:
   ``https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv``
   Downloaded on first use; re-download on each call (no local cache).
2. Synthetic fallback: a deterministic synthetic dataset that mirrors the COMPAS
   schema (same column names, dtypes, ~6000 rows).  Used only when the real
   source is unavailable (e.g. restricted-network CI).  Prints a warning when
   activated.

Preprocessing (applied identically to real and synthetic data):

- Filter rows where ``days_b_screening_arrest`` is in [-30, 30], ``is_recid``
  != -1, ``c_charge_degree`` != 'O', and ``score_text`` != 'N/A'.
- Label: ``two_year_recid`` (binary int64, 0/1).
- Sensitive: ``sex`` (Male/Female strings), ``race`` (kept as-is).
- Features: ``age``, ``juv_fel_count``, ``juv_misd_count``,
  ``juv_other_count``, ``priors_count`` (all numeric).
- Sanity check: assert at least 5 000 rows survive filtering.
- Spot-check: assert "African-American" and "Caucasian" appear in ``race``.
"""

from __future__ import annotations

import io
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_PROPUBLICA_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

_SENSITIVE_COLUMNS = ["sex", "race"]

_FEATURE_COLUMNS = [
    "age",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
]

_LABEL_COLUMN = "two_year_recid"


def _fetch_via_url() -> pd.DataFrame:
    """Download the ProPublica COMPAS CSV from GitHub."""
    with urllib.request.urlopen(_PROPUBLICA_URL, timeout=60) as response:
        payload = response.read()
    return pd.read_csv(io.BytesIO(payload))


def _make_synthetic(seed: int) -> pd.DataFrame:
    """Return a small deterministic synthetic frame that mirrors COMPAS."""
    warnings.warn(
        "COMPAS real data is unavailable (network download failed). "
        "Falling back to a synthetic dataset. "
        "Enable network access for real results.",
        UserWarning,
        stacklevel=4,
    )
    rng = np.random.default_rng(seed)
    n = 7000  # larger than the 5 000-row sanity threshold after filtering

    races = [
        "African-American",
        "Caucasian",
        "Hispanic",
        "Asian",
        "Native American",
        "Other",
    ]
    sexes = ["Male", "Female"]
    charge_degrees = ["F", "M"]
    score_texts = ["Low", "Medium", "High"]

    frame = pd.DataFrame(
        {
            "days_b_screening_arrest": rng.integers(-30, 31, size=n).astype(float),
            "is_recid": rng.integers(0, 2, size=n),
            "c_charge_degree": rng.choice(charge_degrees, size=n),
            "score_text": rng.choice(score_texts, size=n),
            "two_year_recid": rng.integers(0, 2, size=n),
            "age": rng.integers(18, 70, size=n).astype(float),
            "juv_fel_count": rng.integers(0, 5, size=n).astype(float),
            "juv_misd_count": rng.integers(0, 5, size=n).astype(float),
            "juv_other_count": rng.integers(0, 5, size=n).astype(float),
            "priors_count": rng.integers(0, 30, size=n).astype(float),
            "sex": rng.choice(sexes, size=n),
            "race": rng.choice(races, size=n),
        }
    )
    return frame


def _fetch_raw(seed: int) -> pd.DataFrame:
    """Return a raw frame from the first source that succeeds."""
    try:
        return _fetch_via_url()
    except Exception:
        pass
    return _make_synthetic(seed)


def _filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard ProPublica filtering criteria."""
    mask = (
        frame["days_b_screening_arrest"].between(-30, 30)
        & (frame["is_recid"] != -1)
        & (frame["c_charge_degree"] != "O")
        & (frame["score_text"] != "N/A")
    )
    return frame.loc[mask].reset_index(drop=True)


def _preprocess(
    frame: pd.DataFrame, is_real_data: bool
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Turn the raw COMPAS frame into (features, label, sensitive).

    Deterministic and order-stable: all five numeric feature columns are cast
    to float64, the binary label is an int64 Series, and sex/race are held out
    of features as a two-column sensitive DataFrame.
    """
    frame = frame.copy()

    # Validate required columns.
    required = (
        ["days_b_screening_arrest", "is_recid", "c_charge_degree", "score_text"]
        + [_LABEL_COLUMN]
        + _FEATURE_COLUMNS
        + ["sex", "race"]
    )
    missing_cols = [c for c in required if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"COMPAS source is missing expected columns: {missing_cols}")

    # Apply filters.
    filtered = _filter(frame)

    # Sanity check: at least 5 000 rows must survive (for real data).
    if is_real_data and len(filtered) <= 5000:
        raise ValueError(
            f"COMPAS filter produced only {len(filtered)} rows; "
            "expected > 5 000. Check filter logic."
        )

    frame = filtered

    # Label.
    label = pd.to_numeric(frame[_LABEL_COLUMN], errors="coerce").astype("int64")
    label.name = "two_year_recid"

    # Sensitive attributes.
    sensitive = pd.DataFrame(
        {
            "sex": frame["sex"].astype("object"),
            "race": frame["race"].astype("object"),
        }
    )

    # COMPAS race spot-check (per CLAUDE.md).
    if "African-American" not in sensitive["race"].values:
        raise ValueError("COMPAS race column is missing 'African-American' after filtering.")
    if "Caucasian" not in sensitive["race"].values:
        raise ValueError("COMPAS race column is missing 'Caucasian' after filtering.")

    # Features.
    features = frame[_FEATURE_COLUMNS].copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    # Median imputation for any NaN (rare in real COMPAS but possible in
    # synthetic).
    median_vals = features.median()
    features = features.fillna(median_vals)
    features = features.astype("float64")

    features = features.reset_index(drop=True)
    label = label.reset_index(drop=True)
    sensitive = sensitive.reset_index(drop=True)

    return features, label, sensitive


def _validate(features: pd.DataFrame, label: pd.Series, sensitive: pd.DataFrame) -> None:
    """Assert the loader contract: numeric features, binary label, clean sensitive."""
    if features.isna().any().any():
        raise ValueError("COMPAS features contain NaNs after preprocessing.")
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(f"COMPAS features have non-numeric columns: {non_numeric}")

    if label.dtype != "int64":
        raise ValueError(f"COMPAS label dtype must be int64, got {label.dtype}.")
    label_values = set(label.unique().tolist())
    if not label_values <= {0, 1} or not label_values:
        raise ValueError(f"COMPAS label is not binary 0/1: {sorted(label_values)}.")

    if list(sensitive.columns) != _SENSITIVE_COLUMNS:
        raise ValueError(
            f"sensitive columns must be {_SENSITIVE_COLUMNS}, " f"got {list(sensitive.columns)}."
        )
    if sensitive.isna().any().any():
        raise ValueError("COMPAS sensitive columns contain NaNs.")
    if sensitive["race"].nunique() < 2:
        raise ValueError("COMPAS race column must be multi-valued.")


def _split_indices(label: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, label-stratified 75/25 positional split.

    Returns ``(train_idx, test_idx)`` over ``label``'s 0..n-1 positions.  Same
    seed yields the same partition; the two arrays are disjoint and together
    cover every row.
    """
    idx = np.arange(len(label))
    return train_test_split(idx, test_size=0.25, random_state=seed, stratify=label.to_numpy())


def load_compas(split: str = "test", seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (features, label, sensitive) for the ProPublica COMPAS dataset.

    features: model input columns (age, juv_fel_count, juv_misd_count,
              juv_other_count, priors_count), sensitive columns excluded.
    label: binary int64 Series, 1 == two-year recidivism.
    sensitive: DataFrame with exactly columns ['sex', 'race'], raw values.

    Parameters
    ----------
    split:
        ``'train'`` or ``'test'``.  Test set is 25 % of the data.
    seed:
        Random seed for the stratified split.  Same seed always yields the
        same partition.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    raw = _fetch_raw(seed)
    # Track whether we have real data so the row-count sanity check applies.
    is_real_data = not _is_synthetic(raw)
    features, label, sensitive = _preprocess(raw, is_real_data=is_real_data)
    _validate(features, label, sensitive)

    train_idx, test_idx = _split_indices(label, seed)
    keep = train_idx if split == "train" else test_idx

    features = features.iloc[keep].reset_index(drop=True)
    label = label.iloc[keep].reset_index(drop=True)
    sensitive = sensitive.iloc[keep].reset_index(drop=True)
    return features, label, sensitive


def _is_synthetic(frame: pd.DataFrame) -> bool:
    """Heuristic: the real ProPublica CSV has many more columns than we generate."""
    # The real CSV has ~50 columns; the synthetic frame has exactly the columns
    # we create.
    return len(frame.columns) < 20
