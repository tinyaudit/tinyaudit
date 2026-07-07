"""Train the baseline models and write the footprint table.

This is the Week-2 deliverable: for every (dataset, model) cell, fit the model,
measure predictive quality (accuracy, F1, ROC-AUC) and resource footprint
(parameter count, FLOPs proxy, peak RAM, wall-clock per sample), and append one
row to ``experiments/results/baselines.csv``.

Run it::

    python experiments/run_baselines.py                       # all cells
    python experiments/run_baselines.py --datasets adult      # one dataset
    python experiments/run_baselines.py --models logreg mlp   # subset of models
    python experiments/run_baselines.py --seed 0

Every row is stamped with the seed, a short config hash, and the resolved
library versions so a CSV is self-describing and reproducible. Network access
is needed for the real Adult / COMPAS / Folktables data; Folktables falls back
to a synthetic frame (with a warning) when its source is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from tinyaudit.data import load_adult, load_compas, load_folktables
from tinyaudit.models.sklearn import SklearnModel
from tinyaudit.profile.footprint import profile_model

RESULTS = Path(__file__).resolve().parent / "results"

Loader = Callable[..., tuple[pd.DataFrame, pd.Series, pd.DataFrame]]

DATASETS: dict[str, Loader] = {
    "adult": load_adult,
    "compas": load_compas,
    "folktables": load_folktables,
}


def _models(seed: int) -> dict[str, Callable[[], Any]]:
    """Model factories, keyed by the CLI name. Seeded where supported."""
    return {
        "logreg": lambda: LogisticRegression(max_iter=1000),
        "tree": lambda: DecisionTreeClassifier(random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, random_state=seed),
    }


_VERSION_PKGS = ("tinyaudit", "numpy", "pandas", "scikit-learn", "torch")

_FIELDS = [
    "dataset",
    "model",
    "seed",
    "n_train",
    "n_test",
    "n_features",
    "accuracy",
    "f1",
    "roc_auc",
    "n_params",
    "flops",
    "peak_ram_bytes",
    "wall_clock_s_per_sample",
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


def _config_hash(dataset: str, model: str, seed: int) -> str:
    payload = json.dumps({"dataset": dataset, "model": model, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _scale_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features (scaler fit on train only).

    Scale-sensitive models -- the MLP above all, and to a lesser degree
    logistic regression -- are unusable on the raw Adult columns, whose ranges
    span from 0/1 indicators to capital-gain in the tens of thousands. Column
    names and index are preserved so feature attribution stays interpretable.
    Trees are scale-invariant, so standardizing them is harmless and keeps the
    whole table on one preprocessing convention.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train.to_numpy())
    x_tr = pd.DataFrame(
        scaler.transform(X_train.to_numpy()), columns=X_train.columns, index=X_train.index
    )
    x_te = pd.DataFrame(
        scaler.transform(X_test.to_numpy()), columns=X_test.columns, index=X_test.index
    )
    return x_tr, x_te


def _evaluate(
    dataset: str, model_name: str, factory: Callable[[], Any], seed: int
) -> dict[str, Any]:
    """Fit one (dataset, model) cell and return its result row."""
    loader = DATASETS[dataset]
    X_train, y_train, _ = loader(split="train", seed=seed)
    X_test, y_test, _ = loader(split="test", seed=seed)
    X_train, X_test = _scale_split(X_train, X_test)

    est = factory()
    est.fit(X_train.to_numpy(), y_train.to_numpy())

    audited = SklearnModel(est)
    feats = X_test.to_numpy()
    y_true = y_test.to_numpy()
    y_pred = audited.predict(feats)
    proba = audited.predict_proba(feats)[:, 1]

    footprint = profile_model(audited, feats)

    return {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_test.shape[1],
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 6),
        "n_params": footprint.n_params,
        "flops": footprint.flops,
        "peak_ram_bytes": footprint.peak_ram_bytes,
        "wall_clock_s_per_sample": round(footprint.wall_clock_s_per_sample, 9),
        "config_hash": _config_hash(dataset, model_name, seed),
        "python": platform.python_version(),
        "versions": json.dumps(_versions()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baselines and write the footprint CSV.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument(
        "--models", nargs="+", choices=["logreg", "tree", "mlp"], default=["logreg", "tree", "mlp"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS / "baselines.csv")
    args = parser.parse_args(argv)

    factories = _models(args.seed)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model_name in args.models:
            print(f"[baselines] {dataset} x {model_name} (seed={args.seed}) ...", flush=True)
            try:
                row = _evaluate(dataset, model_name, factories[model_name], args.seed)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            rows.append(row)
            print(
                f"  acc={row['accuracy']:.4f} f1={row['f1']:.4f} auc={row['roc_auc']:.4f} "
                f"params={row['n_params']}",
                flush=True,
            )

    if not rows:
        print("no rows produced (all cells failed)", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[baselines] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
