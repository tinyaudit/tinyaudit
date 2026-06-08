"""Occlusion explainer: determinism, shape, and hand-verifiable signal.

The occlusion explainer is pure NumPy, so its output is exactly reproducible
and can be checked by hand on a constructed model that provably ignores one
feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tinyaudit.xai.occlusion import importance_flips, occlusion_attributions


class _OnlyFeature0Model:
    """An ``AuditedModel`` whose positive proba depends only on feature 0.

    ``p(+) = sigmoid(x0)`` exactly; feature 1 is never read, so its occlusion
    attribution must be identically zero and feature 0's must be non-zero.
    """

    @property
    def n_params(self) -> int:
        return 1

    @property
    def framework(self) -> str:
        return "sklearn"

    def _p_pos(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.asarray(X, dtype=float)[:, 0]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._p_pos(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._p_pos(X) >= 0.5).astype(int)


def test_shape_matches_inputs(lr_model) -> None:
    model, X = lr_model
    attr = occlusion_attributions(model, X)
    assert attr.shape == (X.shape[0], X.shape[1])
    assert np.isfinite(attr).all()


def test_deterministic(lr_model) -> None:
    model, X = lr_model
    a = occlusion_attributions(model, X)
    b = occlusion_attributions(model, X)
    np.testing.assert_array_equal(a, b)


def test_ignored_feature_is_zero_informative_is_not() -> None:
    model = _OnlyFeature0Model()
    # x0 varies (informative); x1 varies too but the model never reads it.
    X = pd.DataFrame(
        {
            "x0": np.linspace(-3.0, 3.0, 11),
            "x1": np.linspace(10.0, -10.0, 11),
        }
    )
    attr = occlusion_attributions(model, X, baseline="mean")

    # Feature 1 is ignored by the model -> exactly zero attribution.
    np.testing.assert_allclose(attr[:, 1], 0.0, atol=0.0)
    # Feature 0 is informative -> at least some sample has non-zero signal.
    assert np.abs(attr[:, 0]).max() > 1e-6


def test_hand_verifiable_single_case() -> None:
    """One sample, one feature, mean baseline: closed-form check."""
    model = _OnlyFeature0Model()
    # Two rows so the column mean of x0 is 0.0; check the second row.
    X = np.array([[-2.0, 5.0], [2.0, 5.0]], dtype=float)
    attr = occlusion_attributions(model, X, baseline="mean")

    # baseline x0 = mean([-2, 2]) = 0 -> p(+|baseline) = sigmoid(0) = 0.5.
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    expected_row1_f0 = sig(2.0) - 0.5
    expected_row0_f0 = sig(-2.0) - 0.5
    assert attr[1, 0] == pytest.approx(expected_row1_f0)
    assert attr[0, 0] == pytest.approx(expected_row0_f0)
    # x1 is ignored -> zero regardless of baseline.
    np.testing.assert_allclose(attr[:, 1], 0.0, atol=0.0)


def test_median_baseline_runs_and_shapes(tree_model) -> None:
    model, X = tree_model
    attr = occlusion_attributions(model, X, baseline="median")
    assert attr.shape == (X.shape[0], X.shape[1])
    assert np.isfinite(attr).all()


def test_invalid_baseline_raises(lr_model) -> None:
    model, X = lr_model
    with pytest.raises(ValueError, match="baseline must be"):
        occlusion_attributions(model, X, baseline="zero")


# --- importance_flips: top-k membership semantics ----------------------------


def test_flips_empty_with_single_group() -> None:
    pg = {"A": np.array([3.0, 2.0, 1.0, 0.0])}
    assert importance_flips(pg) == []


def test_flips_empty_when_topk_membership_identical() -> None:
    """Same top-k set in both groups -> no flip, even if inner order differs."""
    # Top-2 is {0, 1} for both groups; only their relative order swaps.
    pg = {
        "A": np.array([0.9, 0.8, 0.1, 0.05]),
        "B": np.array([0.8, 0.9, 0.1, 0.05]),
    }
    assert importance_flips(pg, top_k=2) == []


def test_flips_detects_membership_change() -> None:
    """A feature decisive for one group but not the other is a flip."""
    # top_k=2. Group A top-2 = {0, 1}; group B top-2 = {0, 2}. Feature 2 enters
    # B's shortlist, feature 1 leaves it -> both flip.
    pg = {
        "A": np.array([0.9, 0.8, 0.1, 0.05]),
        "B": np.array([0.9, 0.05, 0.8, 0.1]),
    }
    assert importance_flips(pg, top_k=2) == [1, 2]


def test_flips_robust_to_deep_rank_noise() -> None:
    """A one-slot wobble outside the top-k is not a flip."""
    # Identical top-3 {0,1,2}; features 3 and 4 swap far down the ranking.
    pg = {
        "A": np.array([0.9, 0.8, 0.7, 0.20, 0.19]),
        "B": np.array([0.9, 0.8, 0.7, 0.19, 0.20]),
    }
    assert importance_flips(pg, top_k=3) == []


def test_flips_topk_capped_to_feature_count() -> None:
    """top_k larger than n_features means every feature is in every top-k."""
    pg = {
        "A": np.array([0.9, 0.1, 0.5]),
        "B": np.array([0.1, 0.9, 0.5]),
    }
    assert importance_flips(pg, top_k=99) == []


def test_flips_invalid_topk_raises() -> None:
    pg = {"A": np.array([1.0, 2.0]), "B": np.array([2.0, 1.0])}
    with pytest.raises(ValueError, match="top_k must be"):
        importance_flips(pg, top_k=0)
