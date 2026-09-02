"""Prune-then-fine-tune arm (reviewer request).

The headline compression sweep (``run_compression_sweep.py``) applies one-shot
magnitude pruning with no fine-tuning. Standard deployment practice is to prune
*then* fine-tune to recover the accuracy lost to pruning, and a reviewer will
ask whether the Result-2 divergence -- demographic parity falling while
per-group calibration error rises -- survives that step or is an artifact of
skipping it.

This script answers that directly. For the headline MLP it produces two rows
per (dataset, sensitive, sparsity) cell that differ in exactly one step:

* ``protocol="oneshot"``  -- magnitude-prune, then audit. Same as the sweep.
* ``protocol="finetune"`` -- magnitude-prune, masked-fine-tune, then audit.

Both prune with the same ``magnitude_prune`` code path and audit through the
same ``audit()`` call with ``compression=None`` (the model is already
compressed), so the only difference between the paired rows is the fine-tuning.
That makes the comparison clean: if the divergence persists in the ``finetune``
rows, the finding is not a no-fine-tune artifact.

Masked fine-tuning on an sklearn ``MLPClassifier``: after pruning we hold a
boolean mask of the surviving weights, continue training with ``warm_start`` for
a few short rounds, and re-zero the pruned positions after each round so they
stay pruned while the survivors adapt. This is the sklearn-native equivalent of
masked gradient fine-tuning and keeps the estimator family identical to the
sweep (no torch port, so the numbers stay comparable).

Run it (resumable, one seed per invocation -- background tasks here get killed
at 35-60 min, so a per-seed CSV under that window is the safe unit)::

    python experiments/run_finetune_arm.py --seed 0 --out results/finetune_seed0.csv

Then aggregate the per-seed CSVs with ``analyze_finetune.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from tinyaudit import audit
from tinyaudit.compress.prune import magnitude_prune
from tinyaudit.models.sklearn import SklearnModel

# Reuse the sweep's exact cell machinery so rows are directly comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_compression_sweep import (  # noqa: E402
    _FIELDS,
    DATASETS,
    _row_from_card,
    _scale_split,
)

RESULTS = Path(__file__).resolve().parent / "results"

# Sparsity ladder. 0.0 is the unpruned control (both protocols collapse to the
# same model there, which is a useful built-in sanity check).
SPARSITIES = [0.0, 0.3, 0.5, 0.7, 0.9]
FIELDS = ["protocol", *_FIELDS]


class _AuditInputs(NamedTuple):
    """The fitted headline MLP plus the scaled splits an audit cell needs."""

    est: MLPClassifier
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    s_test: pd.DataFrame


def _masked_finetune(
    est: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    rounds: int,
    iters: int,
) -> MLPClassifier:
    """Fine-tune a pruned MLP while keeping the pruned weights at zero.

    ``est`` is already pruned (its ``coefs_`` carry exact zeros). We snapshot
    the surviving-weight mask, then for a few rounds continue training with
    ``warm_start`` and re-apply the mask, so the survivors adapt to recover
    accuracy while the pruned positions stay pruned. The input estimator is not
    mutated: a deep copy is fine-tuned and returned, matching the immutable
    convention of ``compress/prune.py``.
    """
    est = copy.deepcopy(est)
    masks = [(np.asarray(c) != 0.0) for c in est.coefs_]
    est.warm_start = True
    est.max_iter = iters
    for _ in range(rounds):
        est.fit(X, y)
        est.coefs_ = [c * m for c, m in zip(est.coefs_, masks, strict=False)]
    return est


def _fit_full_mlp(dataset: str, sensitive: str, seed: int) -> _AuditInputs:
    """Fit the standardized headline MLP and return it plus the audit inputs."""
    loader = DATASETS[dataset]
    X_train, y_train, _ = loader(split="train", seed=seed)
    X_test, y_test, s_test = loader(split="test", seed=seed)
    X_train, X_test = _scale_split(X_train, X_test)

    est = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, random_state=seed)
    est.fit(X_train.to_numpy(), y_train.to_numpy())
    return _AuditInputs(est, X_train, y_train, X_test, y_test, s_test)


def _audit_row(
    est: Any,
    protocol: str,
    dataset: str,
    sensitive: str,
    sparsity: float,
    seed: int,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    s_test: pd.DataFrame,
) -> dict[str, Any]:
    """Audit an already-compressed estimator and flatten to a labeled row."""
    card = audit(
        est,
        data=(X_test, y_test),
        sensitive=s_test[sensitive],
        compression=None,  # model is already pruned; do not re-compress
        seed=seed,
        dataset=dataset,
    )
    row = _row_from_card(card, dataset, "mlp", sensitive, f"prune:{sparsity}", seed)
    row["protocol"] = protocol
    return row


def _run_seed(
    datasets: list[str],
    sensitives: list[str],
    seed: int,
    rounds: int,
    iters: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for sensitive in sensitives:
            data = _fit_full_mlp(dataset, sensitive, seed)
            X_tr = data.X_train.to_numpy()
            y_tr = data.y_train.to_numpy()
            for sparsity in SPARSITIES:
                # One prune feeds both protocols: _masked_finetune copies its
                # input, so the one-shot estimator is never touched.
                oneshot_est = magnitude_prune(SklearnModel(data.est), sparsity).estimator
                finetune_est = _masked_finetune(oneshot_est, X_tr, y_tr, rounds, iters)
                for protocol, est in (("oneshot", oneshot_est), ("finetune", finetune_est)):
                    tag = f"{dataset} x {sensitive} x prune:{sparsity} x {protocol}"
                    print(f"[finetune-arm] {tag} ...", flush=True)
                    try:
                        row = _audit_row(
                            est,
                            protocol,
                            dataset,
                            sensitive,
                            sparsity,
                            seed,
                            data.X_test,
                            data.y_test,
                            data.s_test,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ! skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
                        continue
                    rows.append(row)
                    print(
                        f"  DP={row['demographic_parity_difference']} "
                        f"ECE={row['mean_ece_per_group']} "
                        f"acc={row['accuracy']}",
                        flush=True,
                    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prune-then-fine-tune arm.")
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=["adult", "compas"]
    )
    parser.add_argument("--sensitive", nargs="+", choices=["sex", "race"], default=["sex", "race"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=6, help="masked fine-tune rounds")
    parser.add_argument("--iters", type=int, default=50, help="iterations per round")
    parser.add_argument("--out", type=Path, default=RESULTS / "finetune_arm.csv")
    args = parser.parse_args(argv)

    rows = _run_seed(args.datasets, args.sensitive, args.seed, args.rounds, args.iters)
    if not rows:
        print("no rows produced", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[finetune-arm] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
