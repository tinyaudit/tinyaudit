"""Unit tests for the masked fine-tune helper in ``experiments/run_finetune_arm.py``.

The prune-then-fine-tune arm answers a reviewer question (does the Result-2
divergence survive standard fine-tuning?), so its correctness has to be pinned:
if fine-tuning silently refilled the pruned weights, the "fine-tuned" model
would not actually be sparse and the comparison would be meaningless. These
tests lock the three invariants the arm depends on:

1. fine-tuning preserves the target sparsity (pruned weights stay exactly zero);
2. the pruned *positions* are frozen -- the same weights zeroed before
   fine-tuning are the ones zero after (the mask never drifts);
3. fine-tuning is deterministic given a seed, and it actually trains (the
   surviving weights move away from the one-shot pruned values).

Small synthetic data keeps every test well under a second.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

from tinyaudit.compress.prune import magnitude_prune
from tinyaudit.models.sklearn import SklearnModel

_MOD_PATH = Path(__file__).resolve().parents[2] / "experiments" / "run_finetune_arm.py"
_spec = importlib.util.spec_from_file_location("run_finetune_arm", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)


def _fitted_mlp(seed: int = 0) -> tuple[MLPClassifier, np.ndarray, np.ndarray]:
    """A small, quickly-trained MLP on a linearly-separable synthetic problem."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((200, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    est = MLPClassifier(hidden_layer_sizes=(8,), max_iter=200, random_state=seed)
    est.fit(X, y)
    return est, X, y


def _zero_fraction(est: MLPClassifier) -> float:
    total = sum(c.size for c in est.coefs_)
    zeros = sum(int((np.asarray(c) == 0.0).sum()) for c in est.coefs_)
    return zeros / total


class TestMaskedFinetunePreservesSparsity:
    def test_zero_fraction_at_least_target_after_finetune(self) -> None:
        # Arrange
        est, X, y = _fitted_mlp()
        for sparsity in (0.5, 0.9):
            pruned = magnitude_prune(SklearnModel(est), sparsity).estimator
            # Act
            tuned = ft._masked_finetune(pruned, X, y, rounds=3, iters=20)
            # Assert: fine-tuning did not refill pruned weights.
            assert _zero_fraction(tuned) >= sparsity - 1e-9

    def test_pruned_positions_stay_exactly_zero(self) -> None:
        # Arrange
        est, X, y = _fitted_mlp()
        pruned = magnitude_prune(SklearnModel(est), 0.7).estimator
        frozen = [(np.asarray(c) == 0.0) for c in pruned.coefs_]
        # Act
        tuned = ft._masked_finetune(pruned, X, y, rounds=3, iters=20)
        # Assert: every position zeroed by pruning is still exactly zero.
        for mask, coef in zip(frozen, tuned.coefs_, strict=False):
            assert np.all(np.asarray(coef)[mask] == 0.0)


class TestMaskedFinetuneMaskIsFixed:
    def test_surviving_positions_never_grow(self) -> None:
        # The set of nonzero positions after fine-tuning must be a subset of the
        # set that survived pruning: fine-tuning may zero a survivor by chance
        # but must never resurrect a pruned weight.
        est, X, y = _fitted_mlp()
        pruned = magnitude_prune(SklearnModel(est), 0.6).estimator
        survivors = [(np.asarray(c) != 0.0) for c in pruned.coefs_]
        tuned = ft._masked_finetune(pruned, X, y, rounds=3, iters=20)
        for surv, coef in zip(survivors, tuned.coefs_, strict=False):
            now_nonzero = np.asarray(coef) != 0.0
            assert np.all(now_nonzero <= surv)  # no position outside the mask


class TestMaskedFinetuneDeterministicAndTrains:
    def test_two_finetunes_are_identical(self) -> None:
        est, X, y = _fitted_mlp()
        a = ft._masked_finetune(magnitude_prune(SklearnModel(est), 0.9).estimator, X, y, 3, 20)
        b = ft._masked_finetune(magnitude_prune(SklearnModel(est), 0.9).estimator, X, y, 3, 20)
        assert all(np.array_equal(x, z) for x, z in zip(a.coefs_, b.coefs_, strict=False))

    def test_input_estimator_is_not_mutated(self) -> None:
        # Immutability: fine-tuning returns a copy and must leave the caller's
        # pruned estimator untouched (matching compress/prune.py's convention).
        est, X, y = _fitted_mlp()
        pruned = magnitude_prune(SklearnModel(est), 0.7).estimator
        before = [np.asarray(c).copy() for c in pruned.coefs_]
        _ = ft._masked_finetune(pruned, X, y, rounds=3, iters=20)
        pairs = zip(before, pruned.coefs_, strict=False)
        assert all(np.array_equal(b, np.asarray(c)) for b, c in pairs)

    def test_finetune_actually_updates_survivors(self) -> None:
        # Fine-tuning must change the surviving weights (otherwise it is a no-op
        # and the whole arm is meaningless). Compare survivors before/after.
        est, X, y = _fitted_mlp()
        pruned = magnitude_prune(SklearnModel(est), 0.5).estimator
        before = [np.asarray(c).copy() for c in pruned.coefs_]
        tuned = ft._masked_finetune(pruned, X, y, rounds=3, iters=20)
        pairs = zip(before, tuned.coefs_, strict=False)
        moved = any(not np.allclose(b, np.asarray(c)) for b, c in pairs)
        assert moved
