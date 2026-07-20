"""Fairness/uncertainty decoupling experiment.

This is the ``uncertainty/metrics.py`` definition-of-done: reproduce, on UCI
Adult, the qualitative claim that *point-prediction fairness and
uncertainty-based fairness are decoupled* -- a model can look fair (or fairer)
on demographic parity while its per-group calibration and predictive-entropy
disparities tell a different story.

For every (sensitive attribute, compression) cell it runs the ``audit()``
pipeline and records, per sensitive group, the point-fairness signal (selection
rate) alongside the uncertainty-fairness signals (mean predictive entropy,
per-group ECE), into ``experiments/results/decoupling_adult.csv``. It also
prints a per-cell summary contrasting the point-fairness gap (demographic-parity
difference = max-min selection rate) with the calibration gap (max-min ECE).

The headline finding it surfaces (uncompressed, properly scaled model): the
point-fairness and uncertainty-fairness lenses *rank the groups differently*.
On ``race``, White is the best-calibrated group (lowest ECE) yet has one of the
highest selection rates, while Asian-Pac-Islander has both the highest selection
rate and the worst calibration -- so which group looks "worst served" depends
entirely on whether you read selection rate, predictive entropy, or ECE. On
``sex``, the Male/Female disparity is large in selection rate and in entropy but
negligible in calibration. That is the decoupling: point-prediction parity does
not imply, and does not predict, uncertainty or calibration parity.

Compression is swept as a secondary axis. This script uses the (small, robust)
logistic model, whose gaps move little under pruning once features are scaled --
the dramatic swings on raw features were largely a conditioning artifact. Higher-
capacity models do swing under compression; that is visible for the MLP in
``compression_sweep.csv`` (pruning collapses its DP while its ECE climbs).

Run it::

    python experiments/run_decoupling.py
    python experiments/run_decoupling.py --sensitive sex
    python experiments/run_decoupling.py --compressions none prune:0.9

Adult is the only dataset used here (it is the dataset the source claim is
stated on and is available offline). Every row is stamped with the seed, a short
config hash, and the resolved library versions.
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

from tinyaudit import AuditCard, audit
from tinyaudit.data import load_adult, load_compas, load_folktables

RESULTS = Path(__file__).resolve().parent / "results"

ALL_COMPRESSIONS = ["none", "prune:0.5", "prune:0.9"]
SENSITIVE_ATTRS = ["sex", "race"]

# The decoupling experiment runs on any dataset with a point/uncertainty split.
# Adult is the default (the dataset the source claim is stated on); COMPAS is a
# second, higher-stakes replication.
DATASETS = {
    "adult": load_adult,
    "compas": load_compas,
    "folktables": load_folktables,
}

_VERSION_PKGS = ("tinyaudit", "numpy", "pandas", "scikit-learn", "torch")

_FIELDS = [
    "dataset",
    "sensitive",
    "compression",
    "group",
    "n",
    "selection_rate",
    "mean_entropy",
    "ece",
    "demographic_parity_difference",
    "seed",
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


def _scale_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features (scaler fit on train only) so the logistic model is
    well conditioned. Column names/index are preserved so per-group signals stay
    interpretable."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train.to_numpy())
    x_tr = pd.DataFrame(
        scaler.transform(X_train.to_numpy()), columns=X_train.columns, index=X_train.index
    )
    x_te = pd.DataFrame(
        scaler.transform(X_test.to_numpy()), columns=X_test.columns, index=X_test.index
    )
    return x_tr, x_te


def _config_hash(sensitive: str, compression: str, seed: int, dataset: str = "adult") -> str:
    payload = json.dumps(
        {"dataset": dataset, "sensitive": sensitive, "compression": compression, "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _point_metric(card: AuditCard, name: str) -> float:
    for m in card.fairness.metrics:
        if m.name == name:
            return float(m.value)
    return float("nan")


def decoupling_rows(
    card: AuditCard, sensitive: str, compression: str, seed: int, dataset: str = "adult"
) -> list[dict[str, Any]]:
    """Reshape one audit card into per-group decoupling rows.

    Pure (no I/O): pairs each group's point-fairness signal (selection rate)
    with its uncertainty-fairness signals (mean entropy, ECE). Groups present in
    the fairness block but missing from the uncertainty block (e.g. uncertainty
    skipped) get ``nan`` uncertainty values rather than being dropped.
    """
    fair_pg = card.fairness.per_group
    unc_pg = card.uncertainty.per_group if card.uncertainty is not None else {}
    dp = _point_metric(card, "demographic_parity_difference")

    rows: list[dict[str, Any]] = []
    for group, fair in fair_pg.items():
        unc = unc_pg.get(group, {})
        rows.append(
            {
                "dataset": dataset,
                "sensitive": sensitive,
                "compression": compression,
                "group": group,
                "n": int(fair.get("n", 0)),
                "selection_rate": round(float(fair.get("selection_rate", float("nan"))), 6),
                "mean_entropy": round(float(unc.get("mean_entropy", float("nan"))), 6),
                "ece": round(float(unc.get("ece", float("nan"))), 6),
                "demographic_parity_difference": round(dp, 6),
                "seed": seed,
                "config_hash": _config_hash(sensitive, compression, seed, dataset),
                "python": platform.python_version(),
                "versions": json.dumps(_versions()),
            }
        )
    return rows


def _gap(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if r[key] == r[key]]  # drop nan
    return float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the fairness/uncertainty decoupling.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="adult")
    parser.add_argument("--sensitive", nargs="+", choices=SENSITIVE_ATTRS, default=SENSITIVE_ATTRS)
    parser.add_argument(
        "--compressions", nargs="+", choices=ALL_COMPRESSIONS, default=ALL_COMPRESSIONS
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out_path = args.out or (RESULTS / f"decoupling_{args.dataset}.csv")
    loader = DATASETS[args.dataset]

    X_train, y_train, _ = loader(split="train", seed=args.seed)
    X_test, y_test, s_test = loader(split="test", seed=args.seed)
    X_train, X_test = _scale_split(X_train, X_test)

    all_rows: list[dict[str, Any]] = []
    for sensitive in args.sensitive:
        for compression in args.compressions:
            print(f"[decoupling] {args.dataset} x {sensitive} x {compression} ...", flush=True)
            est = LogisticRegression(max_iter=1000)
            est.fit(X_train.to_numpy(), y_train.to_numpy())
            try:
                card = audit(
                    est,
                    data=(X_test, y_test),
                    sensitive=s_test[sensitive],
                    compression=None if compression == "none" else compression,
                    seed=args.seed,
                    dataset=args.dataset,
                    methods=["fairness", "uncertainty"],
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ! skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            rows = decoupling_rows(card, sensitive, compression, args.seed, args.dataset)
            all_rows.extend(rows)
            dp_gap = _gap(rows, "selection_rate")
            ece_gap = _gap(rows, "ece")
            print(
                f"  point-fairness gap (DP)={dp_gap:.4f}   calibration gap (ECE)={ece_gap:.4f}",
                flush=True,
            )

    if not all_rows:
        print("no rows produced (all cells failed)", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[decoupling] wrote {len(all_rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
