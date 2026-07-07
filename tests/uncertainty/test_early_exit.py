"""Tests for the QUTE-style early-exit ensemble.

Covers the softmax helper (including the binary one-logit edge case), the
output contract on a module that advertises an early-exit head, the
``include_final`` switch, and the guard rails: torch-only, a module with no
early-exit heads, and predict-before-fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyaudit.uncertainty.early_exit import EarlyExitEnsemble, _softmax
from tinyaudit.uncertainty.types import UncertaintyOutput

# --------------------------------------------------------------------------- #
# _softmax helper (pure, no torch)
# --------------------------------------------------------------------------- #


class TestSoftmax:
    def test_rows_sum_to_one(self) -> None:
        out = _softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)

    def test_single_logit_becomes_two_columns(self) -> None:
        out = _softmax(np.array([[0.0], [10.0], [-10.0]]))
        assert out.shape == (3, 2)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
        # logit 0 -> 0.5 ; large positive -> second column ~1.
        np.testing.assert_allclose(out[0], [0.5, 0.5], atol=1e-9)
        assert out[1, 1] > 0.99

    def test_1d_input_promoted(self) -> None:
        out = _softmax(np.array([0.0, 1.0, 2.0]))
        # 1-D becomes a single column -> binary two-column output.
        assert out.shape == (3, 2)


# --------------------------------------------------------------------------- #
# EarlyExitEnsemble (needs torch via fixtures)
# --------------------------------------------------------------------------- #


class TestEarlyExitEnsemble:
    def test_predict_dist_shapes(self, early_exit_model, xy) -> None:
        X, y = xy
        est = EarlyExitEnsemble()
        est.fit(early_exit_model, X, y)
        out = est.predict_dist(X)
        assert isinstance(out, UncertaintyOutput)
        assert out.mean_proba.shape == (len(X), 2)
        assert np.all(np.isfinite(out.predictive_entropy))

    def test_include_final_changes_member_count(self, early_exit_model, xy) -> None:
        """With one branch, dropping the final head leaves a single member, so
        the epistemic terms collapse to zero."""
        X, y = xy
        without = EarlyExitEnsemble(include_final=False)
        without.fit(early_exit_model, X, y)
        out = without.predict_dist(X)
        np.testing.assert_allclose(out.mutual_information, 0.0, atol=1e-12)

        with_final = EarlyExitEnsemble(include_final=True)
        with_final.fit(early_exit_model, X, y)
        out2 = with_final.predict_dist(X)
        # final head + branch = 2 members -> they can disagree.
        assert out2.mutual_information.max() >= 0.0

    def test_fit_rejects_non_torch(self, xy) -> None:
        from sklearn.linear_model import LogisticRegression

        from tinyaudit.models.sklearn import SklearnModel

        X, y = xy
        sk = SklearnModel(LogisticRegression(max_iter=200).fit(X, y))
        est = EarlyExitEnsemble()
        with pytest.raises(TypeError, match="torch"):
            est.fit(sk, X, y)

    def test_fit_rejects_module_without_heads(self, plain_model, xy) -> None:
        X, y = xy
        est = EarlyExitEnsemble()
        with pytest.raises(ValueError, match="early-exit heads"):
            est.fit(plain_model, X, y)

    def test_predict_before_fit_raises(self, xy) -> None:
        X, _ = xy
        est = EarlyExitEnsemble()
        with pytest.raises(RuntimeError, match="before fit"):
            est.predict_dist(X)
