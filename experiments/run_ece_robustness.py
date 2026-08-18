"""Is the calibration-collapse finding an artifact of how ECE is binned?

The paper's second half rests on per-group ECE rising sharply as a model is
pruned. ECE is an estimator with a free parameter -- where the confidence bins
are drawn -- and a degenerate model is exactly the regime where that parameter
bites: a collapsed model emits nearly one repeated confidence value, so under
equal-width binning almost every bin is empty and the estimate rides on
whichever one or two are populated.

Two axes are varied, not one:

- **bin count** (10 vs 15). This is the check that was originally proposed. It
  is the weak axis: both are equal-width, so they mostly agree by construction.
- **bin placement** (equal-width vs equal-mass/adaptive). This is the axis that
  can actually overturn the result. Equal-mass edges sit at quantiles of the
  observed confidences, so every bin is populated by construction and the
  estimate no longer depends on the model's confidences happening to spread
  across [0, 1].

All four configurations are computed from the *same* fitted ensemble output per
cell, so the only thing varying between them is the ECE estimator itself and
the comparison is exactly paired.

Run it::

    python experiments/run_ece_robustness.py
    python experiments/run_ece_robustness.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from tinyaudit.compress import magnitude_prune
from tinyaudit.data import load_adult, load_compas
from tinyaudit.models.sklearn import SklearnModel
from tinyaudit.uncertainty.ensemble import DeepEnsemble
from tinyaudit.uncertainty.metrics import ece_per_group

RESULTS = Path(__file__).resolve().parent / "results"

DATASETS: dict[str, Callable[..., tuple[pd.DataFrame, pd.Series, pd.DataFrame]]] = {
    "adult": load_adult,
    "compas": load_compas,
}

# The cells the paper's calibration claim actually rests on.
CELLS = [
    ("adult", "mlp", "sex"),
    ("adult", "mlp", "race"),
    ("compas", "logreg", "race"),
    ("compas", "logreg", "sex"),
]

SPARSITIES = [0.0, 0.3, 0.5, 0.7, 0.9]

# (binning, n_bins). The first is the pipeline default, so its column
# reproduces the number already in the sweep.
CONFIGS = [
    ("equal_width", 10),
    ("equal_width", 15),
    ("equal_mass", 10),
    ("equal_mass", 15),
]

_VERSION_PKGS = ("tinyaudit", "numpy", "pandas", "scikit-learn")

_FIELDS = [
    "dataset",
    "model",
    "sensitive",
    "sparsity",
    "seed",
    "binning",
    "n_bins",
    "mean_ece_per_group",
    "ece_disparity",
    "min_group_ece",
    "max_group_ece",
    "n_groups_scored",
    "mean_confidence",
    "std_confidence",
    "n_distinct_confidence",
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


def _scale_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train.to_numpy())
    x_tr = pd.DataFrame(
        scaler.transform(X_train.to_numpy()), columns=X_train.columns, index=X_train.index
    )
    x_te = pd.DataFrame(
        scaler.transform(X_test.to_numpy()), columns=X_test.columns, index=X_test.index
    )
    return x_tr, x_te


def _build(model_name: str, seed: int) -> Any:
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000)
    return MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, random_state=seed)


def _rows_for_cell(
    dataset: str, model_name: str, sensitive: str, sparsity: float, seed: int
) -> list[dict[str, Any]]:
    """Fit once, then score ECE four ways off the same predictive distribution."""
    loader = DATASETS[dataset]
    X_train, y_train, _ = loader(split="train", seed=seed)
    X_test, y_test, s_test = loader(split="test", seed=seed)
    X_train, X_test = _scale_split(X_train, X_test)

    est = _build(model_name, seed)
    est.fit(X_train.to_numpy(), y_train.to_numpy())
    audited: Any = SklearnModel(est)
    if sparsity > 0.0:
        audited = magnitude_prune(audited, sparsity)

    feats = X_test.to_numpy()
    y_true = y_test.to_numpy()
    s = s_test[sensitive].to_numpy()

    ens = DeepEnsemble(n_members=5, seed=seed, construction="perturb")
    ens.fit(audited, feats, y_true)
    out = ens.predict_dist(feats)

    confidence = np.max(out.mean_proba, axis=1)
    shared = {
        "dataset": dataset,
        "model": model_name,
        "sensitive": sensitive,
        "sparsity": sparsity,
        "seed": seed,
        "mean_confidence": round(float(np.mean(confidence)), 6),
        "std_confidence": round(float(np.std(confidence)), 6),
        # A collapsed model repeats one confidence value; this counts that
        # directly and is what makes equal-width binning degenerate.
        "n_distinct_confidence": int(np.unique(np.round(confidence, 6)).size),
        "python": platform.python_version(),
        "versions": json.dumps(_versions()),
    }

    rows: list[dict[str, Any]] = []
    for binning, n_bins in CONFIGS:
        per_group = ece_per_group(out, y_true, s, n_bins=n_bins, binning=binning)
        finite = [v for v in per_group.values() if not np.isnan(v)]
        rows.append(
            {
                **shared,
                "binning": binning,
                "n_bins": n_bins,
                "mean_ece_per_group": round(float(np.mean(finite)), 6) if finite else "",
                "ece_disparity": (
                    round(float(max(finite) - min(finite)), 6) if len(finite) >= 2 else 0.0
                ),
                "min_group_ece": round(float(min(finite)), 6) if finite else "",
                "max_group_ece": round(float(max(finite)), 6) if finite else "",
                "n_groups_scored": len(finite),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ECE binning robustness check.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--out", type=Path, default=RESULTS / "ece_robustness.csv")
    args = parser.parse_args(argv)

    total = len(CELLS) * len(SPARSITIES) * len(args.seeds)
    done = 0
    n_rows = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        fh.flush()

        for dataset, model_name, sensitive in CELLS:
            for sparsity in SPARSITIES:
                for seed in args.seeds:
                    done += 1
                    tag = (
                        f"[{done}/{total}] {dataset} x {model_name} x {sensitive} "
                        f"x prune={sparsity} x seed{seed}"
                    )
                    try:
                        rows = _rows_for_cell(dataset, model_name, sensitive, sparsity, seed)
                    except Exception as exc:  # noqa: BLE001
                        print(f"{tag} ! {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                        continue
                    for row in rows:
                        writer.writerow(row)
                        n_rows += 1
                    fh.flush()
                    summary = "  ".join(
                        f"{r['binning'][6:]}{r['n_bins']}={r['mean_ece_per_group']}" for r in rows
                    )
                    print(f"{tag}  {summary}", flush=True)

    if not n_rows:
        print("no rows produced", file=sys.stderr)
        return 1
    print(f"[ece-robustness] wrote {n_rows} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
