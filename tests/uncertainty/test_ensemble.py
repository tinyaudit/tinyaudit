"""Tests for the deep-ensemble estimator.

Covers both construction modes (pre-built members and retrain-from-template,
for the sklearn and torch families), the aggregation output contract, and the
guard rails: too-few members, predict-before-fit, an unsupported template type,
and members that disagree on the class set.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tinyaudit.models.sklearn import SklearnModel
from tinyaudit.uncertainty.ensemble import DeepEnsemble
from tinyaudit.uncertainty.types import UncertaintyOutput


def _sk(X: np.ndarray, y: np.ndarray, seed: int) -> SklearnModel:
    return SklearnModel(LogisticRegression(max_iter=200, random_state=seed).fit(X, y))


# --------------------------------------------------------------------------- #
# Construction-time validation
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_too_few_members_template(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            DeepEnsemble(n_members=1)

    def test_too_few_prebuilt_members(self, xy) -> None:
        X, y = xy
        with pytest.raises(ValueError, match="at least 2"):
            DeepEnsemble(members=[_sk(X, y, 0)])

    def test_predict_before_fit_raises(self, xy) -> None:
        X, _ = xy
        with pytest.raises(RuntimeError, match="before fit"):
            DeepEnsemble(n_members=3).predict_dist(X)


# --------------------------------------------------------------------------- #
# Pre-built members
# --------------------------------------------------------------------------- #


class TestPrebuilt:
    def test_prebuilt_members_used_directly(self, xy) -> None:
        X, y = xy
        members = [_sk(X, y, s) for s in range(3)]
        ens = DeepEnsemble(members=members)
        ens.fit(_sk(X, y, 99), X, y)  # template ignored in prebuilt mode
        # The same member objects are kept (the list is copied, contents are not).
        assert ens.members == members
        out = ens.predict_dist(X)
        assert isinstance(out, UncertaintyOutput)
        assert out.mean_proba.shape == (len(X), 2)

    def test_class_set_mismatch_raises(self, xy) -> None:
        """Members with different class counts cannot be stacked."""
        X, y = xy

        class _ThreeClass:
            framework = "sklearn"
            n_params = 1

            def predict(self, X):  # noqa: ANN001
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):  # noqa: ANN001
                return np.full((len(X), 3), 1 / 3)

        ens = DeepEnsemble(members=[_sk(X, y, 0), _ThreeClass()])  # type: ignore[list-item]
        ens.fit(_sk(X, y, 0), X, y)
        with pytest.raises(ValueError, match="number of classes"):
            ens.predict_dist(X)


# --------------------------------------------------------------------------- #
# Retrain from a template
# --------------------------------------------------------------------------- #


class TestTemplateMode:
    def test_sklearn_template_trains_distinct_members(self, xy) -> None:
        X, y = xy
        ens = DeepEnsemble(n_members=4, seed=0)
        ens.fit(_sk(X, y, 0), X, y)
        assert len(ens.members) == 4
        out = ens.predict_dist(X)
        assert out.mean_proba.shape == (len(X), 2)
        assert np.all(np.isfinite(out.predictive_entropy))

    def test_torch_template_trains_members(self, dropout_model, xy) -> None:
        X, y = xy
        ens = DeepEnsemble(n_members=3, seed=0, epochs=10, lr=0.05)
        ens.fit(dropout_model, X, y)
        assert len(ens.members) == 3
        out = ens.predict_dist(X)
        assert out.mean_proba.shape == (len(X), 2)
        # Independently trained members should disagree somewhere -> MI > 0.
        assert out.mutual_information.max() > 0.0

    def test_unsupported_template_type_raises(self, xy) -> None:
        X, y = xy

        class _Weird:
            framework = "other"
            n_params = 0

            def predict(self, X):  # noqa: ANN001
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):  # noqa: ANN001
                return np.full((len(X), 2), 0.5)

        ens = DeepEnsemble(n_members=3)
        with pytest.raises(TypeError, match="SklearnModel and TorchModel"):
            ens.fit(_Weird(), X, y)  # type: ignore[arg-type]
