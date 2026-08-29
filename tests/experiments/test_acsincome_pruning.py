"""Unit tests for the pure helpers in ``experiments/run_acsincome_pruning.py``.

The full script is validated by running against real ACSIncome data; the only
non-trivial pure logic is the seeded test-set cap, which must keep the three
frames row-aligned and must be a no-op when no capping is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_MOD_PATH = Path(__file__).resolve().parents[2] / "experiments" / "run_acsincome_pruning.py"
_spec = importlib.util.spec_from_file_location("run_acsincome_pruning", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
acs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acs)


def _frames(n: int):
    X = pd.DataFrame({"f0": np.arange(n), "f1": np.arange(n) * 2})
    y = pd.Series(np.arange(n) % 2, name="label")
    sex = [("M", "F")[i % 2] for i in range(n)]
    race = [("A", "B", "C", "D")[i % 4] for i in range(n)]
    s = pd.DataFrame({"sex": sex, "race": race})
    return X, y, s


class TestCapTest:
    def test_caps_to_requested_rows(self) -> None:
        X, y, s = _frames(100)
        Xc, yc, sc = acs._cap_test(X, y, s, max_rows=20, seed=0)
        assert len(Xc) == len(yc) == len(sc) == 20

    def test_frames_stay_aligned(self) -> None:
        X, y, s = _frames(100)
        Xc, yc, sc = acs._cap_test(X, y, s, max_rows=30, seed=1)
        # Same index across all three frames after sampling.
        assert list(Xc.index) == list(yc.index) == list(sc.index)
        # f0 equals the row's original position, so alignment is checkable.
        assert (Xc["f0"].to_numpy() == np.array(Xc.index)).all()

    def test_zero_is_noop(self) -> None:
        X, y, s = _frames(50)
        Xc, _, _ = acs._cap_test(X, y, s, max_rows=0, seed=0)
        assert len(Xc) == 50

    def test_larger_cap_than_data_is_noop(self) -> None:
        X, y, s = _frames(40)
        Xc, _, _ = acs._cap_test(X, y, s, max_rows=1000, seed=0)
        assert len(Xc) == 40

    def test_seeded_and_deterministic(self) -> None:
        X, y, s = _frames(100)
        a, _, _ = acs._cap_test(X, y, s, max_rows=25, seed=7)
        b, _, _ = acs._cap_test(X, y, s, max_rows=25, seed=7)
        assert list(a.index) == list(b.index)
