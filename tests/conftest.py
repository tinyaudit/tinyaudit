"""Shared deterministic fixtures.

Unit tests use synthetic data so they are fast and need no network. The
data-loader tests exercise the real datasets separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def binary_classification(rng: np.random.Generator):
    """A small, deterministic binary task with one sensitive column.

    Returns ``(X, y, sensitive)`` where ``sensitive`` has two groups whose
    base rates differ, so fairness metrics are non-trivial.
    """
    n = 400
    sensitive = rng.integers(0, 2, size=n)
    # Group 1 has a higher positive base rate than group 0.
    base = np.where(sensitive == 1, 0.65, 0.35)
    y = (rng.random(n) < base).astype(int)
    x0 = y + rng.normal(0, 0.5, size=n)
    x1 = rng.normal(0, 1, size=n)
    X = pd.DataFrame({"x0": x0, "x1": x1})
    return X, pd.Series(y, name="label"), pd.Series(sensitive, name="group")
