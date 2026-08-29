"""Compression-vs-fairness sweep.

This is the Week-7 deliverable and the paper's headline question: *does the
audit survive compression?* For every (dataset, model, sensitive attribute,
compression) cell it runs the full ``audit()`` pipeline and records the six
metrics plus the footprint into ``experiments/results/compression_sweep.csv``.

Compressions swept: ``none``, ``int8``, and magnitude pruning at 0.30, 0.50,
0.70, 0.90 (the sparsities the spec fixes).

Alongside the six fairness and uncertainty metrics the sweep records the
audited model's accuracy, balanced accuracy, positive prediction rate, and the
mean and standard deviation of its predicted probabilities. Those columns are
what make the headline result falsifiable: demographic parity improves under
pruning because the model is collapsing toward a single answer, and without an
accuracy column beside it the improvement is indistinguishable from a genuine
one. Two constant-predictor rows per dataset (``majority`` and ``prevalence``)
pin the degenerate end of that scale; pass ``--no-baselines`` to omit them.

Cells that cannot run are written out with a ``skip_reason`` rather than
dropped. Decision trees have no weight array, so every compressed tree cell
lands here.

Run it::

    python experiments/run_compression_sweep.py                 # representative subset
    python experiments/run_compression_sweep.py --full          # every cell
    python experiments/run_compression_sweep.py --datasets adult --models logreg

Each ``audit()`` builds a 5-member perturbation ensemble for the uncertainty
stage: the members are multiplicative weight-jitters of the (already compressed)
audited model, so the uncertainty metrics are computed on the compressed model
rather than on fresh full-precision retrains. On the int8 cells the audited
model is an ONNX-backed ``QuantizedOnnxModel`` whose float weights are not
reachable, so the ensemble cannot be built (``PerturbNotSupportedError``); the
uncertainty stage is recorded as skipped and its columns are left blank -- a
documented limitation, not a failure.
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
    "accuracy",
    "balanced_accuracy",
    "positive_prediction_rate",
    "mean_predicted_prob",
    "std_predicted_prob",
    "mean_group_predictive_entropy",
    "mean_group_predictive_variance",
    "mean_group_mutual_information",
    "mean_ece_per_group",
    "entropy_disparity",
    "variance_disparity",
    "mi_disparity",
    "ece_disparity",
    "selective_fairness_auc",
    "n_params",
    "flops",
    "peak_ram_bytes",
    "skip_reason",
    "python",
    "versions",
]

# Constant-predictor reference rows. Both ignore their input, so both score a
# demographic-parity difference of exactly 0.0 and a disparate-impact ratio of
# exactly 1.0 while being useless. They are the yardstick the pruned cells are
# read against: any compressed model whose fairness numbers approach these is
# not fairer, it has collapsed.
BASELINE_MODES = ["majority", "prevalence"]


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


def _scale_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features (scaler fit on train only) so scale-sensitive
    models train on comparable inputs. Column names/index are preserved so the
    audit's feature attribution stays interpretable; trees are unaffected."""
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
    """Pull a metric value by name from a card block, or '' if absent."""
    if block is None:
        return ""
    for m in block.metrics:
        if m.name == name:
            return round(float(m.value), 6)
    return ""


def _row_from_card(
    card: Any,
    dataset: str,
    model_name: str,
    sensitive: str,
    compression: str,
    seed: int,
) -> dict[str, Any]:
    """Flatten one audit card into a CSV row."""
    return {
        "dataset": dataset,
        "model": model_name,
        "sensitive": sensitive,
        "compression": compression,
        "seed": seed,
        "demographic_parity_difference": _metric(card.fairness, "demographic_parity_difference"),
        "equalized_odds_difference": _metric(card.fairness, "equalized_odds_difference"),
        "disparate_impact_ratio": _metric(card.fairness, "disparate_impact_ratio"),
        "accuracy": _metric(card.performance, "accuracy"),
        "balanced_accuracy": _metric(card.performance, "balanced_accuracy"),
        "positive_prediction_rate": _metric(card.performance, "positive_prediction_rate"),
        "mean_predicted_prob": _metric(card.performance, "mean_predicted_prob"),
        "std_predicted_prob": _metric(card.performance, "std_predicted_prob"),
        "mean_group_predictive_entropy": _metric(card.uncertainty, "mean_group_predictive_entropy"),
        "mean_group_predictive_variance": _metric(
            card.uncertainty, "mean_group_predictive_variance"
        ),
        "mean_group_mutual_information": _metric(card.uncertainty, "mean_group_mutual_information"),
        "mean_ece_per_group": _metric(card.uncertainty, "mean_ece_per_group"),
        "entropy_disparity": _metric(card.uncertainty, "entropy_disparity"),
        "variance_disparity": _metric(card.uncertainty, "variance_disparity"),
        "mi_disparity": _metric(card.uncertainty, "mi_disparity"),
        "ece_disparity": _metric(card.uncertainty, "ece_disparity"),
        "selective_fairness_auc": _metric(card.uncertainty, "selective_fairness_auc"),
        "n_params": card.footprint.n_params,
        "flops": card.footprint.flops,
        "peak_ram_bytes": card.footprint.peak_ram_bytes,
        "skip_reason": "",
        "python": platform.python_version(),
        "versions": json.dumps(_versions()),
    }


def _skipped_row(
    dataset: str,
    model_name: str,
    sensitive: str,
    compression: str,
    seed: int,
    reason: str,
) -> dict[str, Any]:
    """A row for a cell that could not be run, with the reason recorded.

    Emitting the cell rather than dropping it is deliberate. Decision trees
    have no weight array, so neither ``magnitude_prune`` nor ``quantize_int8``
    applies to them, and a bare ``continue`` made twenty of the seventy-two
    cells vanish from the results with nothing in the CSV to say why. A blank
    row with a ``skip_reason`` keeps the gap in the data instead of in a log.
    """
    row: dict[str, Any] = dict.fromkeys(_FIELDS, "")
    row.update(
        {
            "dataset": dataset,
            "model": model_name,
            "sensitive": sensitive,
            "compression": compression,
            "seed": seed,
            "skip_reason": " ".join(reason.split())[:200],
            "python": platform.python_version(),
            "versions": json.dumps(_versions()),
        }
    )
    return row


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
    X_train, X_test = _scale_split(X_train, X_test)

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

    return _row_from_card(card, dataset, model_name, sensitive, compression, seed)


def _run_baseline_cell(
    dataset: str,
    mode: str,
    sensitive: str,
    seed: int,
) -> dict[str, Any]:
    """Audit a constant predictor fitted to the training labels.

    Takes the same path through ``audit()`` as every other cell, so its
    fairness numbers are produced by exactly the same code that produces the
    compressed models'. The uncertainty stage cannot build a perturbation
    ensemble from a model with no weights, so those columns come back blank;
    that is expected and the fairness and performance columns are the point.
    """
    from tinyaudit.models import ConstantModel

    loader = DATASETS[dataset]
    _, y_train, _ = loader(split="train", seed=seed)
    X_test, y_test, s_test = loader(split="test", seed=seed)

    model = ConstantModel.from_labels(y_train.to_numpy(), mode=mode)
    card = audit(
        model,
        data=(X_test, y_test),
        sensitive=s_test[sensitive],
        compression=None,
        seed=seed,
        dataset=dataset,
    )

    return _row_from_card(card, dataset, mode, sensitive, "none", seed)


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
    parser.add_argument(
        "--no-baselines",
        dest="baselines",
        action="store_false",
        help="omit the majority-class and prevalence constant-predictor rows",
    )
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
                        reason = f"{type(exc).__name__}: {exc}"
                        print(f"  ! skipped: {reason}", file=sys.stderr)
                        rows.append(
                            _skipped_row(
                                dataset, model_name, sensitive, compression, args.seed, reason
                            )
                        )
                        continue
                    rows.append(row)
                    print(
                        f"  DP={row['demographic_parity_difference']} "
                        f"DI={row['disparate_impact_ratio']} "
                        f"acc={row['accuracy']} bal={row['balanced_accuracy']}",
                        flush=True,
                    )

        if args.baselines:
            for sensitive in args.sensitive:
                for mode in BASELINE_MODES:
                    tag = f"{dataset} x {mode} (baseline) x {sensitive}"
                    print(f"[sweep] {tag} ...", flush=True)
                    try:
                        row = _run_baseline_cell(dataset, mode, sensitive, args.seed)
                    except Exception as exc:  # noqa: BLE001
                        reason = f"{type(exc).__name__}: {exc}"
                        print(f"  ! skipped: {reason}", file=sys.stderr)
                        rows.append(
                            _skipped_row(dataset, mode, sensitive, "none", args.seed, reason)
                        )
                        continue
                    rows.append(row)
                    print(
                        f"  DP={row['demographic_parity_difference']} "
                        f"DI={row['disparate_impact_ratio']} "
                        f"acc={row['accuracy']} bal={row['balanced_accuracy']}",
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
