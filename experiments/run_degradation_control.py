"""Is the parity collapse caused by compression, or by damage of any kind?

The compression finding on its own cannot answer the obvious objection: maybe
*any* sufficiently broken model shows a shrinking demographic-parity gap, and
pruning is incidental. This script is the control for that.

Rather than matching a single accuracy point, it traces a whole curve through
accuracy space under three different ways of damaging the same model, and asks
whether the curves overlay:

- ``prune`` -- magnitude pruning at increasing sparsity. The mechanism under
  test.
- ``label_noise`` -- a fraction of *training* labels flipped before fitting.
  The test set is never touched. This is damage that has nothing to do with
  compression, and it is the control she asked for.
- ``subsample`` -- fitting on a shrinking fraction of the training rows. A
  second non-compression route to the same accuracies, so a single quirk of
  label noise cannot carry the conclusion.

Read the output by fixing an accuracy and comparing the fairness metrics across
mechanisms. If the three curves lie on top of each other, compression is not
special: it is one common and unusually silent route to a generic collapse, and
the paper's claim has to soften accordingly. If pruning's parity gap closes
*faster* per point of accuracy lost than the controls', compression is doing
something the controls are not, and the causal framing survives.

The explainability stage is skipped throughout. Occlusion over Adult's 101
features dominates the runtime and contributes nothing to this question.

Run it::

    python experiments/run_degradation_control.py
    python experiments/run_degradation_control.py --seeds 0 1 2 --models logreg
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from tinyaudit import audit
from tinyaudit.data import load_adult, load_compas

RESULTS = Path(__file__).resolve().parent / "results"

DATASETS: dict[str, Callable[..., tuple[pd.DataFrame, pd.Series, pd.DataFrame]]] = {
    "adult": load_adult,
    "compas": load_compas,
}

SENSITIVE_ATTRS = ["sex", "race"]

# Levels are chosen so the three mechanisms span a comparable accuracy range.
# Pruning runs past the paper's 0.9 because that is where the collapse lives,
# and label noise stops below 0.5 because at 0.5 the labels carry no signal at
# all and the comparison degenerates.
LEVELS: dict[str, list[float]] = {
    "prune": [0.0, 0.3, 0.5, 0.7, 0.9, 0.95],
    "label_noise": [0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
    "subsample": [1.0, 0.50, 0.20, 0.10, 0.05, 0.02],
}

_VERSION_PKGS = ("tinyaudit", "numpy", "pandas", "scikit-learn", "torch")

_FIELDS = [
    "dataset",
    "model",
    "sensitive",
    "mechanism",
    "level",
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
    "mean_ece_per_group",
    "ece_disparity",
    "selective_fairness_auc",
    "n_params",
    "n_train_effective",
    "label_flip_fraction",
    "skip_reason",
    "python",
    "versions",
]


def _models(seed: int) -> dict[str, Callable[[], Any]]:
    return {
        "logreg": lambda: LogisticRegression(max_iter=1000),
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
    """Standardize features with the scaler fit on train only.

    Mirrors ``run_compression_sweep._scale_split`` exactly so the pruning arm
    here is directly comparable to the main sweep.
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


def _flip_labels(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Flip ``fraction`` of binary labels, chosen uniformly at random.

    Symmetric noise: a flip is equally likely in either direction, so the class
    balance is preserved in expectation and the damage is not itself a source
    of parity change.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"label noise expects a binary target; got classes {classes!r}")
    if fraction <= 0.0:
        return y.copy()
    out = y.copy()
    idx = rng.random(len(y)) < fraction
    lo, hi = classes
    out[idx] = np.where(out[idx] == lo, hi, lo)
    return out


def _subsample(
    X: pd.DataFrame, y: np.ndarray, fraction: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep ``fraction`` of the training rows, chosen uniformly at random."""
    if fraction >= 1.0:
        return X, y
    n_keep = max(int(round(len(y) * fraction)), 10)
    idx = rng.choice(len(y), size=n_keep, replace=False)
    return X.iloc[idx], y[idx]


def _metric(block: Any, name: str) -> float | str:
    if block is None:
        return ""
    for m in block.metrics:
        if m.name == name:
            return round(float(m.value), 6)
    return ""


@dataclass
class _Damaged:
    """One trained-and-damaged model plus the test data to audit it on.

    Built once per (dataset, model, mechanism, level, seed) and reused across
    sensitive attributes. Training is the dominant cost on the Adult MLP, and
    the protected attribute has no influence on it, so refitting per attribute
    would double the run for nothing.
    """

    est: Any
    X_test: pd.DataFrame
    y_test: pd.Series
    s_test: pd.DataFrame
    compression: str | None
    flip_fraction: float
    n_train_effective: int


def _fit_damaged(
    dataset: str,
    factory: Callable[[], Any],
    mechanism: str,
    level: float,
    seed: int,
) -> _Damaged:
    loader = DATASETS[dataset]
    X_train, y_train_s, _ = loader(split="train", seed=seed)
    X_test, y_test, s_test = loader(split="test", seed=seed)
    X_train, X_test = _scale_split(X_train, X_test)
    y_train = y_train_s.to_numpy()

    # A dedicated stream so the damage draw does not disturb the estimator's
    # own RNG; the seed still drives it, so the cell reproduces.
    rng = np.random.default_rng(seed * 1000 + int(level * 100))

    compression: str | None = None
    flip_fraction = 0.0
    if mechanism == "prune":
        compression = None if level == 0.0 else f"prune:{level}"
    elif mechanism == "label_noise":
        flip_fraction = level
        y_train = _flip_labels(y_train, level, rng)
    elif mechanism == "subsample":
        X_train, y_train = _subsample(X_train, y_train, level, rng)
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")

    est = factory()
    est.fit(X_train.to_numpy(), y_train)

    return _Damaged(
        est=est,
        X_test=X_test,
        y_test=y_test,
        s_test=s_test,
        compression=compression,
        flip_fraction=flip_fraction,
        n_train_effective=len(y_train),
    )


def _audit_cell(
    fitted: _Damaged,
    dataset: str,
    model_name: str,
    sensitive: str,
    mechanism: str,
    level: float,
    seed: int,
) -> dict[str, Any]:
    card = audit(
        fitted.est,
        data=(fitted.X_test, fitted.y_test),
        sensitive=fitted.s_test[sensitive],
        methods=["fairness", "uncertainty"],
        compression=fitted.compression,
        seed=seed,
        dataset=dataset,
    )

    return {
        "dataset": dataset,
        "model": model_name,
        "sensitive": sensitive,
        "mechanism": mechanism,
        "level": level,
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
        "mean_ece_per_group": _metric(card.uncertainty, "mean_ece_per_group"),
        "ece_disparity": _metric(card.uncertainty, "ece_disparity"),
        "selective_fairness_auc": _metric(card.uncertainty, "selective_fairness_auc"),
        "n_params": card.footprint.n_params,
        "n_train_effective": fitted.n_train_effective,
        "label_flip_fraction": fitted.flip_fraction,
        "skip_reason": "",
        "python": platform.python_version(),
        "versions": json.dumps(_versions()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accuracy-matched degradation control.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=["logreg", "mlp"], default=["logreg", "mlp"])
    parser.add_argument("--sensitive", nargs="+", choices=SENSITIVE_ATTRS, default=SENSITIVE_ATTRS)
    parser.add_argument(
        "--mechanisms", nargs="+", choices=list(LEVELS), default=list(LEVELS), help="damage routes"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--out", type=Path, default=RESULTS / "degradation_control.csv")
    args = parser.parse_args(argv)

    n_rows = 0
    total = sum(
        len(args.datasets)
        * len(args.models)
        * len(args.sensitive)
        * len(LEVELS[m])
        * len(args.seeds)
        for m in args.mechanisms
    )
    done = 0

    # Rows are streamed and flushed as they are produced. The full grid is a
    # multi-hour job on the Adult MLP, and a crash near the end should not cost
    # every completed cell.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        fh.flush()

        for dataset in args.datasets:
            for model_name in args.models:
                for mechanism in args.mechanisms:
                    for level in LEVELS[mechanism]:
                        for seed in args.seeds:
                            stem = f"{dataset} x {model_name} x {mechanism}={level} x seed{seed}"

                            fitted: _Damaged | None = None
                            fit_error = ""
                            try:
                                fitted = _fit_damaged(
                                    dataset, _models(seed)[model_name], mechanism, level, seed
                                )
                            except Exception as exc:  # noqa: BLE001
                                fit_error = f"{type(exc).__name__}: {exc}"
                                print(f"{stem} ! fit failed: {fit_error}", file=sys.stderr)

                            for sensitive in args.sensitive:
                                done += 1
                                tag = f"[{done}/{total}] {stem} x {sensitive}"
                                reason = fit_error
                                row: dict[str, Any] | None = None
                                if fitted is not None:
                                    try:
                                        row = _audit_cell(
                                            fitted,
                                            dataset,
                                            model_name,
                                            sensitive,
                                            mechanism,
                                            level,
                                            seed,
                                        )
                                        print(
                                            f"{tag}  acc={row['accuracy']} "
                                            f"DP={row['demographic_parity_difference']}",
                                            flush=True,
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        reason = f"{type(exc).__name__}: {exc}"
                                        print(f"{tag} ! {reason}", file=sys.stderr, flush=True)

                                if row is None:
                                    row = dict.fromkeys(_FIELDS, "")
                                    row.update(
                                        {
                                            "dataset": dataset,
                                            "model": model_name,
                                            "sensitive": sensitive,
                                            "mechanism": mechanism,
                                            "level": level,
                                            "seed": seed,
                                            "skip_reason": " ".join(reason.split())[:200],
                                            "python": platform.python_version(),
                                            "versions": json.dumps(_versions()),
                                        }
                                    )
                                writer.writerow(row)
                                fh.flush()
                                n_rows += 1

    if not n_rows:
        print("no rows produced", file=sys.stderr)
        return 1
    print(f"[degradation] wrote {n_rows} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
