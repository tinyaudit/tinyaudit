"""Lightweight ACSIncome replication of the main pruning result.

The compression findings are shown on Adult and COMPAS; a reviewer asked whether
the headline pruning effect -- demographic parity falling while per-group
calibration disparity rises as a model is pruned -- also appears on ACSIncome,
the modern Census replacement for Adult. This is that check, kept deliberately
small: one model (logistic regression by default), a few seeds, the pruning
ladder, both sensitive attributes.

ACSIncome's test split is ~49k rows; a full 5-member perturbation ensemble over
all of it for every cell is a multi-hour job for a supporting result. So the
test set is subsampled to ``--max-test-rows`` (default 8000, seeded) to keep the
run to minutes. The subsample is a caveat, not a shortcut: race groups still
carry hundreds of rows each at 8k, enough for the group disparities this script
reports. Pass ``--max-test-rows 0`` to use the whole split.

For each (sensitive, compression, seed) cell it runs the full ``audit()`` and
records the point-fairness gap (demographic parity), the calibration disparity
(per-group ECE gap), the uncertainty disparity (per-group entropy gap), plus
accuracy and positive-prediction rate so a parity drop can be told apart from a
genuine improvement. Output: ``experiments/results/acsincome_pruning.csv``.

Run it::

    python experiments/run_acsincome_pruning.py
    python experiments/run_acsincome_pruning.py --model mlp --seeds 0 1 2 3 4
    python experiments/run_acsincome_pruning.py --max-test-rows 0   # full split
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from tinyaudit import AuditCard, audit
from tinyaudit.data import load_folktables

RESULTS = Path(__file__).resolve().parent / "results"

_MODEL_CTORS = {
    "logreg": lambda seed: LogisticRegression(max_iter=1000),
    "mlp": lambda seed: MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, random_state=seed),
}

SENSITIVE_ATTRS = ["sex", "race"]
DEFAULT_COMPRESSIONS = ["none", "prune:0.3", "prune:0.5", "prune:0.7", "prune:0.9"]
DEFAULT_MAX_TEST_ROWS = 8000

_VERSION_PKGS = ("tinyaudit", "numpy", "pandas", "scikit-learn", "torch", "folktables")

_FIELDS = [
    "dataset",
    "model",
    "sensitive",
    "compression",
    "seed",
    "n_test",
    "demographic_parity_difference",
    "accuracy",
    "positive_prediction_rate",
    "mean_ece_per_group",
    "ece_disparity",
    "entropy_disparity",
    "mi_disparity",
    "config_hash",
    "python",
    "versions",
]


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _VERSION_PKGS:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "unavailable"
    return out


def _config_hash(model: str, sensitive: str, compression: str, seed: int, n_test: int) -> str:
    payload = json.dumps(
        {
            "dataset": "folktables",
            "model": model,
            "sensitive": sensitive,
            "compression": compression,
            "seed": seed,
            "n_test": n_test,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _scale_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features (scaler fit on train only); names/index preserved."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train.to_numpy())
    x_tr = pd.DataFrame(
        scaler.transform(X_train.to_numpy()), columns=X_train.columns, index=X_train.index
    )
    x_te = pd.DataFrame(
        scaler.transform(X_test.to_numpy()), columns=X_test.columns, index=X_test.index
    )
    return x_tr, x_te


def _metric(block: Any, name: str) -> float | str:
    if block is None:
        return ""
    for m in block.metrics:
        if m.name == name:
            return round(float(m.value), 6)
    return ""


def summarize(
    card: AuditCard, model: str, sensitive: str, compression: str, seed: int, n_test: int
) -> dict[str, Any]:
    """Flatten one audit card into a single ACSIncome pruning row (pure)."""
    return {
        "dataset": "folktables",
        "model": model,
        "sensitive": sensitive,
        "compression": compression,
        "seed": seed,
        "n_test": n_test,
        "demographic_parity_difference": _metric(card.fairness, "demographic_parity_difference"),
        "accuracy": _metric(card.performance, "accuracy"),
        "positive_prediction_rate": _metric(card.performance, "positive_prediction_rate"),
        "mean_ece_per_group": _metric(card.uncertainty, "mean_ece_per_group"),
        "ece_disparity": _metric(card.uncertainty, "ece_disparity"),
        "entropy_disparity": _metric(card.uncertainty, "entropy_disparity"),
        "mi_disparity": _metric(card.uncertainty, "mi_disparity"),
        "config_hash": _config_hash(model, sensitive, compression, seed, n_test),
        "python": platform.python_version(),
        "versions": json.dumps(_versions()),
    }


def _cap_test(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    s_test: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Seeded subsample of the test split to keep the ensemble pass cheap.

    ``max_rows <= 0`` or a split already smaller than ``max_rows`` is a no-op.
    The three frames are sampled with the same index so rows stay aligned.
    """
    if max_rows <= 0 or len(X_test) <= max_rows:
        return X_test, y_test, s_test
    idx = X_test.sample(n=max_rows, random_state=seed).index
    return X_test.loc[idx], y_test.loc[idx], s_test.loc[idx]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight ACSIncome pruning replication.")
    parser.add_argument("--model", choices=sorted(_MODEL_CTORS), default="logreg")
    parser.add_argument("--sensitive", nargs="+", choices=SENSITIVE_ATTRS, default=SENSITIVE_ATTRS)
    parser.add_argument("--compressions", nargs="+", default=DEFAULT_COMPRESSIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-test-rows", type=int, default=DEFAULT_MAX_TEST_ROWS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out_path = args.out or (RESULTS / "acsincome_pruning.csv")
    all_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        X_train, y_train, _ = load_folktables(split="train", seed=seed)
        X_test, y_test, s_test = load_folktables(split="test", seed=seed)
        X_train, X_test = _scale_split(X_train, X_test)
        X_test, y_test, s_test = _cap_test(X_test, y_test, s_test, args.max_test_rows, seed)
        n_test = len(X_test)

        est = _MODEL_CTORS[args.model](seed)
        est.fit(X_train.to_numpy(), y_train.to_numpy())

        for sensitive in args.sensitive:
            for compression in args.compressions:
                print(
                    f"[acsincome] {args.model} x {sensitive} x {compression} x seed{seed} "
                    f"(n_test={n_test}) ...",
                    flush=True,
                )
                try:
                    card = audit(
                        est,
                        data=(X_test, y_test),
                        sensitive=s_test[sensitive],
                        compression=None if compression == "none" else compression,
                        seed=seed,
                        dataset="folktables",
                        methods=["fairness", "uncertainty"],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                all_rows.append(summarize(card, args.model, sensitive, compression, seed, n_test))

    if not all_rows:
        print("no rows produced (all cells failed)", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[acsincome] wrote {len(all_rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
