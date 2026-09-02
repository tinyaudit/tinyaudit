"""Peak-RAM comparison: TinyAudit's point-fairness stage vs Fairlearn.

Reviewer request. The feasibility section argues the audit is light enough to
run where the model runs. A natural question is how that footprint compares to
the standard fairness toolkit. Fairlearn is TinyAudit's reference oracle for the
point metrics (it is a test-only dependency and never on the package hot path),
so it is the right baseline.

This measures, at several streaming batch sizes, the ``tracemalloc`` peak of
computing the *same* demographic-parity difference two ways on identical
inputs:

* ``tinyaudit``  -- the package's in-house DP on the model predictions.
* ``fairlearn``  -- ``fairlearn.metrics.demographic_parity_difference`` on the
  same predictions and sensitive labels.

Only the metric computation is inside the measured region (predictions are made
first, outside it), so the number isolates the fairness stage's own working set,
not the model's. Methodology mirrors ``run_audit_footprint.py``: tracemalloc
(which sees numpy buffers), one warm-up run to absorb one-off import
allocations, and the same standardized-feature fit.

Run it::

    python experiments/run_fairlearn_ram.py
    python experiments/run_fairlearn_ram.py --batches 256 1024 --datasets adult

Writes ``experiments/results/fairlearn_ram.csv``.
"""

from __future__ import annotations

import argparse
import tracemalloc
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tinyaudit.data import load_adult, load_compas
from tinyaudit.fairness.parity import demographic_parity_difference as ta_dp

RESULTS = Path(__file__).resolve().parent / "results"
DATASETS = {"adult": load_adult, "compas": load_compas}
SENSITIVE = {"adult": "sex", "compas": "race"}


def _prep(dataset: str, batch: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit a standardized logistic model and return (y_pred, sensitive, n)."""
    loader = DATASETS[dataset]
    X_tr, y_tr, _ = loader(split="train", seed=seed)
    X_te, y_te, s_te = loader(split="test", seed=seed)
    scaler = StandardScaler().fit(X_tr.to_numpy())
    Xs_tr = scaler.transform(X_tr.to_numpy())
    model = LogisticRegression(max_iter=1000).fit(Xs_tr, y_tr.to_numpy())

    n = len(X_te) if batch <= 0 else min(batch, len(X_te))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te), size=n, replace=False)
    y_pred = model.predict(scaler.transform(X_te.to_numpy()[idx]))
    sens = s_te[SENSITIVE[dataset]].to_numpy()[idx]
    return np.asarray(y_pred), np.asarray(sens), n


def _peak(fn: Callable[[], float]) -> tuple[int, float]:
    """Return (tracemalloc peak bytes, returned value) for calling ``fn``."""
    tracemalloc.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak), float(value)


def _fairlearn_dp(y_pred: np.ndarray, sens: np.ndarray) -> float:
    from fairlearn.metrics import demographic_parity_difference

    # DP is a pure selection-rate metric and ignores y_true, so passing y_pred
    # as the y_true stand-in keeps the call valid without changing the value.
    return float(demographic_parity_difference(y_pred, y_pred, sensitive_features=sens))


def _run(dataset: str, batch: int, seed: int) -> list[dict[str, Any]]:
    y_pred, sens, n = _prep(dataset, batch, seed)
    rows: list[dict[str, Any]] = []
    for tool, fn in (
        ("tinyaudit", lambda: ta_dp(y_pred, sens)),
        ("fairlearn", lambda: _fairlearn_dp(y_pred, sens)),
    ):
        peak, value = _peak(fn)
        rows.append(
            {
                "dataset": dataset,
                "batch": n,
                "seed": seed,
                "tool": tool,
                "peak_ram_bytes": peak,
                "dp_value": round(value, 6),
            }
        )
        print(
            f"[fairlearn-ram] {dataset} batch={n} {tool}: "
            f"peak={peak / 1024:,.1f} KB DP={value:.4f}",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TinyAudit vs Fairlearn peak-RAM for DP.")
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=["adult", "compas"]
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS / "fairlearn_ram.csv")
    args = parser.parse_args(argv)

    # Warm-up: absorb fairlearn/numpy one-off import allocations.
    _run(args.datasets[0], 64, args.seed)

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for batch in args.batches:
            rows.extend(_run(dataset, batch, args.seed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[fairlearn-ram] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
