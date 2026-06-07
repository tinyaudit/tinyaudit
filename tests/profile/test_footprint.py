"""Profiler unit tests against a tiny in-file fake model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tinyaudit.profile.footprint import Footprint, profile_model


class FakeModel:
    """Minimal ``AuditedModel``: predict returns zeros, proba is 2-col."""

    @property
    def n_params(self) -> int:
        return 42

    @property
    def framework(self) -> str:
        return "sklearn"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


def test_profile_model_basic() -> None:
    model = FakeModel()
    X = np.arange(20, dtype=float).reshape(10, 2)

    fp = profile_model(model, X)

    assert isinstance(fp, Footprint)
    assert fp.n_params == 42
    assert isinstance(fp.peak_ram_bytes, int)
    assert fp.peak_ram_bytes > 0
    assert fp.flops == 42 * len(X)
    assert isinstance(fp.wall_clock_s_per_sample, float)
    assert fp.wall_clock_s_per_sample >= 0.0


def test_profile_model_accepts_pandas() -> None:
    model = FakeModel()
    X = pd.DataFrame({"a": range(8), "b": range(8)})

    fp = profile_model(model, X)

    assert isinstance(fp, Footprint)
    assert fp.n_params == 42
    assert fp.flops == 42 * len(X)
    assert fp.peak_ram_bytes > 0


def test_profile_model_empty_input_guard() -> None:
    model = FakeModel()
    X = np.empty((0, 3), dtype=float)

    fp = profile_model(model, X)

    assert fp.flops == 0
    assert fp.wall_clock_s_per_sample == 0.0
