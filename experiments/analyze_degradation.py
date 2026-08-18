"""Accuracy-matched comparison of the three damage mechanisms.

The control asks one question: at the *same* accuracy, does a pruned model show
the same fairness numbers as a model damaged some other way? Comparing raw
levels cannot answer it, because 90% sparsity and 40% label noise are not
comparable quantities. This script puts every mechanism on a common accuracy
axis and reads the fairness metrics off at matched points.

Two outputs:

- a matched table, printed and written to ``degradation_matched.csv``: for each
  (dataset, model, sensitive), every mechanism's curve linearly interpolated
  onto a shared accuracy grid spanning the range all three actually reach.
- a per-mechanism slope, written to ``degradation_slopes.csv``: the change in
  each fairness metric per point of accuracy given up, fitted by least squares
  over that mechanism's own curve. The slope is the compact form of the claim:
  if pruning's DP slope is steeper than the controls', compression is buying
  apparent fairness at a rate that damage alone does not explain.

Run it after ``run_degradation_control.py``::

    python experiments/analyze_degradation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

MECHANISMS = ["prune", "label_noise", "subsample"]
METRICS = [
    "demographic_parity_difference",
    "equalized_odds_difference",
    "disparate_impact_ratio",
    "mean_ece_per_group",
    "ece_disparity",
    "positive_prediction_rate",
]
_SHORT = {
    "demographic_parity_difference": "DP",
    "equalized_odds_difference": "EO",
    "disparate_impact_ratio": "DI",
    "mean_ece_per_group": "ECE",
    "ece_disparity": "ECEgap",
    "positive_prediction_rate": "posrate",
}


def _load(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise SystemExit(f"no input CSVs found among {[str(p) for p in paths]}")
    df = pd.concat(frames, ignore_index=True)
    if "skip_reason" in df.columns:
        df = df[df["skip_reason"].isna() | (df["skip_reason"] == "")]
    for c in ["accuracy", "level", *METRICS]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["accuracy"])


def _curve(cell: pd.DataFrame, mechanism: str) -> pd.DataFrame:
    """Seed-averaged (accuracy, metrics) curve for one mechanism, sorted by accuracy."""
    sub = cell[cell["mechanism"] == mechanism]
    if sub.empty:
        return sub
    agg = sub.groupby("level")[["accuracy", *METRICS]].mean().reset_index()
    return agg.sort_values("accuracy").reset_index(drop=True)


def _matched_rows(
    dataset: str, model: str, sensitive: str, cell: pd.DataFrame, n_points: int
) -> list[dict[str, object]]:
    """Interpolate every mechanism onto the accuracy range all of them cover."""
    curves = {m: _curve(cell, m) for m in MECHANISMS}
    curves = {m: c for m, c in curves.items() if len(c) >= 2}
    if len(curves) < 2:
        return []

    # The overlap is where a matched comparison is honest. Outside it, at least
    # one mechanism never reached that accuracy and interpolation would be an
    # extrapolation dressed up as a measurement.
    lo = max(float(c["accuracy"].min()) for c in curves.values())
    hi = min(float(c["accuracy"].max()) for c in curves.values())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []

    grid = np.linspace(lo, hi, n_points)
    rows: list[dict[str, object]] = []
    for acc in grid:
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "sensitive": sensitive,
            "accuracy": round(float(acc), 6),
        }
        for mech, c in curves.items():
            for metric in METRICS:
                value = float(np.interp(acc, c["accuracy"].to_numpy(), c[metric].to_numpy()))
                row[f"{_SHORT[metric]}_{mech}"] = round(value, 6)
        rows.append(row)
    return rows


def _slope_rows(
    dataset: str, model: str, sensitive: str, cell: pd.DataFrame
) -> list[dict[str, object]]:
    """Least-squares change in each metric per point of accuracy lost."""
    rows: list[dict[str, object]] = []
    for mech in MECHANISMS:
        c = _curve(cell, mech)
        if len(c) < 2:
            continue
        # x is accuracy *lost* from this mechanism's own undamaged end, so a
        # positive slope means the metric grows as the model gets worse.
        lost = float(c["accuracy"].max()) - c["accuracy"].to_numpy()
        if np.ptp(lost) == 0:
            continue
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "sensitive": sensitive,
            "mechanism": mech,
            "accuracy_range": round(float(np.ptp(c["accuracy"].to_numpy())), 6),
        }
        for metric in METRICS:
            slope = float(np.polyfit(lost, c[metric].to_numpy(), 1)[0])
            row[f"d{_SHORT[metric]}_per_acc"] = round(slope, 4)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accuracy-matched degradation analysis.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            RESULTS / "degradation_control_logreg.csv",
            RESULTS / "degradation_control_mlp.csv",
            RESULTS / "degradation_control.csv",
        ],
    )
    parser.add_argument("--points", type=int, default=5, help="matched accuracy grid size")
    args = parser.parse_args(argv)

    df = _load(args.inputs)

    matched: list[dict[str, object]] = []
    slopes: list[dict[str, object]] = []
    for (dataset, model, sensitive), cell in df.groupby(["dataset", "model", "sensitive"]):
        matched.extend(_matched_rows(dataset, model, sensitive, cell, args.points))
        slopes.extend(_slope_rows(dataset, model, sensitive, cell))

    if not slopes:
        print("no mechanism had enough points to compare")
        return 1

    slope_df = pd.DataFrame(slopes)
    slope_df.to_csv(RESULTS / "degradation_slopes.csv", index=False)
    print("=== change in metric per point of accuracy lost ===")
    print("(positive = metric rises as the model degrades)\n")
    show = ["dataset", "model", "sensitive", "mechanism", "accuracy_range"]
    show += ["dDP_per_acc", "dEO_per_acc", "dECE_per_acc", "dposrate_per_acc"]
    print(slope_df[show].to_string(index=False))

    if matched:
        matched_df = pd.DataFrame(matched)
        matched_df.to_csv(RESULTS / "degradation_matched.csv", index=False)
        print("\n=== accuracy-matched DP by mechanism ===")
        cols = ["dataset", "model", "sensitive", "accuracy"]
        cols += [f"DP_{m}" for m in MECHANISMS if f"DP_{m}" in matched_df.columns]
        print(matched_df[cols].to_string(index=False))
        print(f"\n[analyze] wrote degradation_matched.csv ({len(matched_df)} rows)")
    else:
        print("\nno overlapping accuracy range across mechanisms; matched table skipped")

    print(f"[analyze] wrote degradation_slopes.csv ({len(slope_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
