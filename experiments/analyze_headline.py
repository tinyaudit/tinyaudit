"""Curate the handful of statistics the paper actually argues.

The study runs hundreds of comparisons; a paper should report the few tied
directly to its hypotheses. This script reads the committed result CSVs and
emits one compact table -- ``experiments/results/headline_stats.csv`` plus a
printed Markdown version -- organised by the three claims:

* **H1 (complementarity).** The uncertainty lens ranks sensitive groups
  differently from the calibration lens, so "uncertainty-aware" is earned rather
  than decorative. Source: ``uncertainty_signal.csv`` (Spearman(entropy, ECE) on
  race; entropy-vs-ECE disparity on sex).
* **H2 (compression hides unfairness).** Under pruning, demographic parity falls
  while per-group calibration disparity rises, so a parity-only audit ranks the
  most degraded model as the fairest. Source: ``compression_sweep_multiseed.csv``
  (endpoint contrast, mean +/- std over 10 seeds) with paired significance from
  ``compression_paired.csv``.
* **H3 (ACSIncome replication).** The same pruning direction holds on a third,
  larger dataset. Source: ``acsincome_pruning.csv``.

Missing input files are skipped with a note rather than crashing, so the table
degrades gracefully to whatever has been generated.

Run it::

    python experiments/analyze_headline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

# The cells each hypothesis leads with. Adult logreg is the deployable case;
# the MLP is the limiting illustration.
H2_CELLS = [("adult", "logreg"), ("adult", "mlp")]
ENDPOINT = "prune:0.9"


def _fmt(mean: float, std: float | None = None) -> str:
    if std is None or np.isnan(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def h1_complementarity(unc: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per framing-critical cell: does uncertainty track calibration?"""
    base = unc[unc["compression"] == "none"]
    rows: list[dict[str, Any]] = []
    grouped = base.groupby(["dataset", "model", "sensitive"], dropna=False)
    for (dataset, model, sensitive), g in grouped:
        cell = f"{dataset}/{model}/{sensitive}"
        if sensitive == "race":
            rho = float(np.nanmean(g["spearman_entropy_ece"]))
            rows.append(
                {
                    "hypothesis": "H1 complementarity",
                    "cell": cell,
                    "comparison": "Spearman(entropy, ECE) over seeds",
                    "value": f"{rho:.3f}",
                    "reading": "near 0 => uncertainty ranks groups unlike calibration",
                }
            )
        else:  # sex: 2 groups, contrast the disparities
            ent = float(np.nanmean(g["entropy_disparity"]))
            ece = float(np.nanmean(g["ece_disparity"]))
            rows.append(
                {
                    "hypothesis": "H1 complementarity",
                    "cell": cell,
                    "comparison": "entropy_disparity vs ece_disparity",
                    "value": f"{ent:.4f} vs {ece:.4f}",
                    "reading": "entropy loud where ECE quiet => complementary",
                }
            )
    return rows


def _endpoint(sweep: pd.DataFrame, dataset: str, model: str, sensitive: str, col: str) -> str:
    """Mean +/- std of ``col`` at compression=none vs the pruned endpoint."""

    def _cell(compression: str) -> tuple[float, float]:
        r = sweep[
            (sweep["dataset"] == dataset)
            & (sweep["model"] == model)
            & (sweep["sensitive"] == sensitive)
            & (sweep["compression"] == compression)
        ]
        if r.empty:
            return float("nan"), float("nan")
        return float(r[f"{col}_mean"].iloc[0]), float(r[f"{col}_std"].iloc[0])

    m0, s0 = _cell("none")
    m1, s1 = _cell(ENDPOINT)
    return f"{_fmt(m0, s0)} -> {_fmt(m1, s1)}"


def h2_compression(sweep: pd.DataFrame, paired: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Endpoint DP and ECE-disparity contrasts, with paired significance."""
    rows: list[dict[str, Any]] = []
    for dataset, model in H2_CELLS:
        for sensitive in ("sex", "race"):
            cell = f"{dataset}/{model}/{sensitive}"
            rows.append(
                {
                    "hypothesis": "H2 compression",
                    "cell": cell,
                    "comparison": "demographic parity: none -> 90% prune",
                    "value": _endpoint(
                        sweep, dataset, model, sensitive, "demographic_parity_difference"
                    ),
                    "reading": "parity falls",
                }
            )
            rows.append(
                {
                    "hypothesis": "H2 compression",
                    "cell": cell,
                    "comparison": "per-group ECE disparity: none -> 90% prune",
                    "value": _endpoint(sweep, dataset, model, sensitive, "ece_disparity"),
                    "reading": "calibration disparity rises",
                }
            )
            if paired is not None:
                for level in ("prune:0.7", ENDPOINT):
                    pr = paired[
                        (paired["dataset"] == dataset)
                        & (paired["model"] == model)
                        & (paired["sensitive"] == sensitive)
                        & (paired["compression"] == level)
                        & (paired["metric"] == "demographic_parity_difference")
                    ]
                    if pr.empty:
                        continue
                    diff = float(pr["mean_diff"].iloc[0])
                    lo = float(pr["ci_lo"].iloc[0])
                    hi = float(pr["ci_hi"].iloc[0])
                    p = float(pr["p_value"].iloc[0])
                    sig = "sig" if bool(pr["ci_excludes_zero"].iloc[0]) else "ns"
                    rows.append(
                        {
                            "hypothesis": "H2 compression",
                            "cell": cell,
                            "comparison": f"paired DP diff at {level}",
                            "value": f"{diff:.3f} [{lo:.3f}, {hi:.3f}] p={p:.3f} ({sig})",
                            "reading": "seed-paired significance",
                        }
                    )
    return rows


def h3_acsincome(acs: pd.DataFrame) -> list[dict[str, Any]]:
    """DP, accuracy, and ECE-disparity endpoint contrasts on ACSIncome."""
    rows: list[dict[str, Any]] = []
    for sensitive in ("sex", "race"):
        g = acs[acs["sensitive"] == sensitive]
        for col, reading in (
            ("demographic_parity_difference", "parity falls"),
            ("accuracy", "accuracy collapses (not a real gain)"),
            ("ece_disparity", "calibration disparity"),
        ):

            def _mean(compression: str, column: str = col, frame: pd.DataFrame = g) -> float:
                r = frame[frame["compression"] == compression]
                vals = pd.to_numeric(r[column], errors="coerce").dropna()
                return float(vals.mean()) if not vals.empty else float("nan")

            rows.append(
                {
                    "hypothesis": "H3 ACSIncome",
                    "cell": f"folktables/logreg/{sensitive}",
                    "comparison": f"{col}: none -> 90% prune",
                    "value": f"{_mean('none'):.4f} -> {_mean(ENDPOINT):.4f}",
                    "reading": reading,
                }
            )
    return rows


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  (skipping {path.name}: not found)")
        return None
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curate the paper's headline statistics.")
    parser.add_argument("--out", type=Path, default=RESULTS / "headline_stats.csv")
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []

    unc = _read(RESULTS / "uncertainty_signal.csv")
    if unc is not None:
        rows.extend(h1_complementarity(unc))

    sweep = _read(RESULTS / "compression_sweep_multiseed.csv")
    paired = _read(RESULTS / "compression_paired.csv")
    if sweep is not None:
        rows.extend(h2_compression(sweep, paired))

    acs = _read(RESULTS / "acsincome_pruning.csv")
    if acs is not None:
        rows.extend(h3_acsincome(acs))

    if not rows:
        print("no headline rows produced (no input CSVs found)")
        return 1

    df = pd.DataFrame(rows, columns=["hypothesis", "cell", "comparison", "value", "reading"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n[headline] wrote {len(df)} rows -> {args.out}\n")
    print("| Hypothesis | Cell | Comparison | Value | Reading |")
    print("|---|---|---|---|---|")
    for r in rows:
        cells = [r["hypothesis"], r["cell"], r["comparison"], r["value"], r["reading"]]
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
