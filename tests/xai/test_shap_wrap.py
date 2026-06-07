"""SHAP adapter: per-branch shape and finiteness, plus the background cap.

One test per explainer-selection branch (linear / tree / kernel). The
KernelExplainer path keeps ``n`` tiny and asserts the background cap is what
bounds the work.
"""

from __future__ import annotations

import numpy as np
import pytest
import shap

from tinyaudit.xai.shap_wrap import shap_attributions


def test_linear_branch_logistic_regression(lr_model) -> None:
    model, X = lr_model
    attr = shap_attributions(model, X)
    assert attr.shape == (X.shape[0], X.shape[1])
    assert np.isfinite(attr).all()


def test_tree_branch_decision_tree(tree_model) -> None:
    model, X = tree_model
    attr = shap_attributions(model, X)
    assert attr.shape == (X.shape[0], X.shape[1])
    assert np.isfinite(attr).all()


def test_kernel_branch_mlp_uses_background_cap(mlp_model, monkeypatch) -> None:
    model, X_full = mlp_model
    # Keep the kernel path fast: a handful of rows only.
    X = X_full.head(12)

    seen: dict[str, int] = {}
    real_sample = shap.sample

    def spy_sample(data, n, random_state=None):  # noqa: ANN001, ANN202
        seen["n"] = n
        return real_sample(data, n, random_state=random_state)

    monkeypatch.setattr(shap, "sample", spy_sample)

    cap = 5
    attr = shap_attributions(model, X, background_size=cap)

    assert attr.shape == (X.shape[0], X.shape[1])
    assert np.isfinite(attr).all()
    # Background is capped at min(background_size, n_samples) == cap here.
    assert seen["n"] == cap


def test_kernel_branch_cap_falls_back_to_n_samples(mlp_model, monkeypatch) -> None:
    model, X_full = mlp_model
    X = X_full.head(6)

    seen: dict[str, int] = {}
    real_sample = shap.sample

    def spy_sample(data, n, random_state=None):  # noqa: ANN001, ANN202
        seen["n"] = n
        return real_sample(data, n, random_state=random_state)

    monkeypatch.setattr(shap, "sample", spy_sample)

    # background_size larger than the data -> capped at n_samples.
    shap_attributions(model, X, background_size=10_000)
    assert seen["n"] == X.shape[0]


def test_accepts_numpy_and_dataframe_equivalently(lr_model) -> None:
    model, X = lr_model
    a = shap_attributions(model, X)
    b = shap_attributions(model, X.to_numpy())
    assert a.shape == b.shape == (X.shape[0], X.shape[1])
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_rejects_non_2d_input(lr_model) -> None:
    model, _X = lr_model
    with pytest.raises(ValueError, match="2-D"):
        shap_attributions(model, np.zeros(5))
