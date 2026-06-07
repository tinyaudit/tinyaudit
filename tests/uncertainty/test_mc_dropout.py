"""Tests for MC Dropout and the shared sample aggregator.

``aggregate_samples`` is the math every estimator shares, so it is tested here
directly. ``MCDropout`` is tested against a fitted dropout MLP: output shapes
and ranges, the reproducibility of seeded passes, and the two contracts ``fit``
enforces (torch-only, dropout-required).
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyaudit.uncertainty.mc_dropout import MCDropout, aggregate_samples
from tinyaudit.uncertainty.types import UncertaintyOutput

# --------------------------------------------------------------------------- #
# aggregate_samples (pure, no torch needed)
# --------------------------------------------------------------------------- #


class TestAggregateSamples:
    def test_shapes_and_ranges(self, rng: np.random.Generator) -> None:
        # (n_members, n_samples, n_classes)
        samples = rng.dirichlet(np.ones(3), size=(4, 50))
        out = aggregate_samples(samples)
        assert isinstance(out, UncertaintyOutput)
        assert out.mean_proba.shape == (50, 3)
        assert out.predictive_entropy.shape == (50,)
        assert out.predictive_variance.shape == (50,)
        assert out.mutual_information.shape == (50,)
        # mean_proba is a valid distribution per row
        np.testing.assert_allclose(out.mean_proba.sum(axis=1), 1.0, atol=1e-9)

    def test_identical_members_give_zero_epistemic(self, rng: np.random.Generator) -> None:
        """If every member agrees, MI and variance are ~0 but entropy is not."""
        one = rng.dirichlet(np.ones(2), size=30)
        samples = np.stack([one, one, one], axis=0)
        out = aggregate_samples(samples)
        np.testing.assert_allclose(out.mutual_information, 0.0, atol=1e-12)
        np.testing.assert_allclose(out.predictive_variance, 0.0, atol=1e-12)
        assert np.all(out.predictive_entropy >= 0.0)

    def test_mutual_information_non_negative(self, rng: np.random.Generator) -> None:
        samples = rng.dirichlet(np.ones(4), size=(6, 40))
        out = aggregate_samples(samples)
        assert np.all(out.mutual_information >= 0.0)

    def test_rejects_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="n_members, n_samples, n_classes"):
            aggregate_samples(np.zeros((10, 2)))

    def test_rejects_zero_members(self) -> None:
        with pytest.raises(ValueError, match="at least one member"):
            aggregate_samples(np.zeros((0, 5, 2)))


# --------------------------------------------------------------------------- #
# MCDropout (needs torch via the fixtures in conftest)
# --------------------------------------------------------------------------- #


class TestMCDropout:
    def test_init_rejects_too_few_passes(self) -> None:
        with pytest.raises(ValueError, match="n_passes"):
            MCDropout(n_passes=1)

    def test_predict_dist_shapes(self, dropout_model, xy) -> None:
        X, y = xy
        est = MCDropout(n_passes=10, seed=0)
        est.fit(dropout_model, X, y)
        out = est.predict_dist(X)
        assert out.mean_proba.shape == (len(X), 2)
        assert out.predictive_entropy.shape == (len(X),)
        assert np.all(np.isfinite(out.predictive_entropy))

    def test_seeded_passes_are_reproducible(self, dropout_model, xy) -> None:
        X, y = xy
        est = MCDropout(n_passes=8, seed=42)
        est.fit(dropout_model, X, y)
        a = est.predict_dist(X)
        b = est.predict_dist(X)
        np.testing.assert_allclose(a.mean_proba, b.mean_proba)
        np.testing.assert_allclose(a.predictive_entropy, b.predictive_entropy)

    def test_dropout_makes_passes_vary(self, dropout_model, xy) -> None:
        """With dropout active the per-pass probabilities are not all identical,
        so the epistemic terms are strictly positive somewhere."""
        X, y = xy
        est = MCDropout(n_passes=20, seed=0)
        est.fit(dropout_model, X, y)
        out = est.predict_dist(X)
        assert out.predictive_variance.max() > 0.0
        assert out.mutual_information.max() > 0.0

    def test_fit_rejects_non_torch(self, xy) -> None:
        from sklearn.linear_model import LogisticRegression

        from tinyaudit.models.sklearn import SklearnModel

        X, y = xy
        sk = SklearnModel(LogisticRegression(max_iter=200).fit(X, y))
        est = MCDropout(n_passes=5)
        with pytest.raises(TypeError, match="torch"):
            est.fit(sk, X, y)

    def test_fit_rejects_module_without_dropout(self, plain_model, xy) -> None:
        X, y = xy
        est = MCDropout(n_passes=5)
        with pytest.raises(ValueError, match="nn.Dropout"):
            est.fit(plain_model, X, y)

    def test_predict_before_fit_raises(self, xy) -> None:
        X, _ = xy
        est = MCDropout(n_passes=5)
        with pytest.raises(RuntimeError, match="before fit"):
            est.predict_dist(X)
