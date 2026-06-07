"""Compression-vs-fairness sweep.

This is the Week-7 deliverable and the paper's headline question: *does the
audit survive compression?* For every (dataset, model, sensitive attribute,
compression) cell it runs the full ``audit()`` pipeline and records the six
metrics plus the footprint into ``experiments/results/compression_sweep.csv``.

Compressions swept: ``none``, ``int8``, and magnitude pruning at 0.30, 0.50,
0.70, 0.90 (the sparsities the spec fixes).

Run it::

    python experiments/run_compression_sweep.py                 # representative subset
    python experiments/run_compression_sweep.py --full          # every cell
    python experiments/run_compression_sweep.py --datasets adult --models logreg

Each ``audit()`` retrains a 5-member deep ensemble for the uncertainty stage,
so the full grid is not cheap; scope it with the flags while iterating. On the
int8 cells the uncertainty stage falls through to empty (a QuantizedOnnxModel
cannot be retrained as an ensemble member) — that is a documented limitation,
recorded as blank uncertainty columns rather than a failure.
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

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from tinyaudit import audit
from tinyaudit.data import load_adult, load_compas, load_folktables

RESULTS = Path(__file__).resolve().parent / "results"

DATASETS: dict[str, Callable[..., tuple[pd.DataFrame, pd.Series, pd.DataFrame]]] = {
    "adult": load_adult,
    "compas": load_compas,
    "folktables": load_folktables,
}

ALL_COMPRESSIONS = ["none", "int8", "prune:0.3", "prune:0.5", "prune:0.7", "prune:0.9"]
DEFAULT_COMPRESSIONS = ["none", "int8", "prune:0.5", "prune:0.9"]
SENSITIVE_ATTRS = ["sex", "race"]

_VERSION_PKGS = ("tinyaudit", "numpy", "pandas", "scikit-learn", "torch")

_FIELDS = [
    "dataset",
    "model",
    "sensitive",
    "compression",
    "seed",
    "demographic_parity_difference",
    "equalized_odds_difference",
    "disparate_impact_ratio",
    "mean_group_predictive_entropy",
    "mean_ece_per_group",
    "selective_fairness_auc",
    "n_params",
    "flops",
    "peak_ram_bytes",
    "python",
    "versions",
]


def _models(seed: int) -> dict[str, Callable[[], Any]]:
    return {
        "logreg": lambda: LogisticRegression(max_iter=1000),
        "tree": lambda: DecisionTreeClassifier(random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, random_state=seed),
    }


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _VERSION_PKGS:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "unavailable"
    return out


def _metric(block: Any, name: str) -> float | str:
    """Pull a metric value by name from a card block, or '' if absent."""
    if block is None:
        return ""
    for m in block.metrics:
        if m.name == name:
            return round(float(m.value), 6)
    return ""


def _run_cell(
    dataset: str,
    model_name: str,
    factory: Callable[[], Any],
    sensitive: str,
    compression: str,
    seed: int,
) -> dict[str, Any]:
    loader = DATASETS[dataset]
    X_train, y_train, _ = loader(split="train", seed=seed)
    X_test, y_test, s_test = loader(split="test", seed=seed)

    est = factory()
    est.fit(X_train.to_numpy(), y_train.to_numpy())

    card = audit(
        est,
        data=(X_test, y_test),
        sensitive=s_test[sensitive],
        compression=None if compression == "none" else compression,
        seed=seed,
        dataset=dataset,
    )

    return {
        "dataset": dataset,
        "model": model_name,
        "sensitive": sensitive,
        "compression": compression,
        "seed": seed,
        "demographic_parity_difference": _metric(card.fairness, "demographic_parity_difference"),
        "equalized_odds_difference": _metric(card.fairness, "equalized_odds_difference"),
        "disparate_impact_ratio": _metric(card.fairness, "disparate_impact_ratio"),
        "mean_group_predictive_entropy": _metric(card.uncertainty, "mean_group_predictive_entropy"),
        "mean_ece_per_group": _metric(card.uncertainty, "mean_ece_per_group"),
        "selective_fairness_auc": _metric(card.uncertainty, "selective_fairness_auc"),
        "n_params": card.footprint.n_params,
        "flops": card.footprint.flops,
        "peak_ram_bytes": card.footprint.peak_ram_bytes,
        "python": platform.python_version(),
        "versions": json.dumps(_versions()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the compression-vs-fairness sweep.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument(
        "--models", nargs="+", choices=["logreg", "tree", "mlp"], default=["logreg", "tree", "mlp"]
    )
    parser.add_argument("--sensitive", nargs="+", choices=SENSITIVE_ATTRS, default=SENSITIVE_ATTRS)
    parser.add_argument(
        "--compressions", nargs="+", choices=ALL_COMPRESSIONS, default=DEFAULT_COMPRESSIONS
    )
    parser.add_argument("--full", action="store_true", help="sweep every compression level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS / "compression_sweep.csv")
    args = parser.parse_args(argv)

    compressions = ALL_COMPRESSIONS if args.full else args.compressions
    factories = _models(args.seed)

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model_name in args.models:
            for sensitive in args.sensitive:
                for compression in compressions:
                    tag = f"{dataset} x {model_name} x {sensitive} x {compression}"
                    print(f"[sweep] {tag} ...", flush=True)
                    try:
                        row = _run_cell(
                            dataset,
                            model_name,
                            factories[model_name],
                            sensitive,
                            compression,
                            args.seed,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ! skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
                        continue
                    rows.append(row)
                    print(
                        f"  DP={row['demographic_parity_difference']} "
                        f"DI={row['disparate_impact_ratio']}",
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
    print(f"[sweep] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
