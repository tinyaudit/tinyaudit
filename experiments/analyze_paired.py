"""Seed-paired comparison of each compression level against no compression.

The paper currently reports mean +/- std over 10 seeds and, seeing the bands
overlap at the intermediate sparsities, retreats to claiming only the endpoint
contrast. That retreat is an artifact of the wrong test. Some seeds simply
produce better models than others, and that between-seed variance sits in both
bands and swamps the effect. Pairing removes it: seed 0's uncompressed model is
compared against seed 0's pruned model, seed 1 against seed 1, and the
statistic is computed over the 10 *differences*.

For each (dataset, model, sensitive, compression) and each metric this reports
the mean paired difference, a 95% confidence interval, a paired t-test against
zero, and Cohen's dz. A confidence interval that excludes zero supports a claim
at that sparsity; overlapping raw bands do not refute one.

Reads the per-seed rows written by ``run_multiseed.py``
(``compression_sweep_perseed.csv``). Run it after the sweep::

    python experiments/analyze_paired.py
    python experiments/analyze_paired.py --input some/other_perseed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

KEYS = ["dataset", "model", "sensitive"]
BASELINE = "none"

METRICS = [
    "demographic_parity_difference",
    "equalized_odds_difference",
    "disparate_impact_ratio",
    "accuracy",
    "balanced_accuracy",
    "positive_prediction_rate",
    "mean_ece_per_group",
    "ece_disparity",
]


def _t_critical(dof: int) -> float:
    """Two-sided 95% t critical value, falling back to the normal quantile."""
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(0.975, dof))
    except Exception:  # noqa: BLE001 - scipy is optional here
        return 1.96


def _t_sf(t_stat: float, dof: int) -> float:
    """Two-sided p-value for a t statistic; NaN when scipy is unavailable."""
    try:
        from scipy.stats import t as student_t

        return float(2.0 * student_t.sf(abs(t_stat), dof))
    except Exception:  # noqa: BLE001
        return float("nan")


def _paired_stats(diffs: np.ndarray) -> dict[str, float]:
    """Mean, 95% CI, paired t-test against zero, and Cohen's dz."""
    n = int(diffs.size)
    mean = float(np.mean(diffs))
    # Sample standard deviation; ddof=1 because these are 10 draws, not the
    # population, and ddof=0 would understate the interval.
    sd = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")

    if n < 2 or not np.isfinite(sd):
        return {
            "n_pairs": n,
            "mean_diff": mean,
            "std_diff": sd,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "cohens_dz": float("nan"),
        }

    stderr = sd / np.sqrt(n)
    dof = n - 1
    half = _t_critical(dof) * stderr
    # A zero-variance difference is a real outcome here, not an error: pruning
    # can drive every seed to the identical constant predictor.
    t_stat = mean / stderr if stderr > 0 else (np.inf if mean != 0 else 0.0)
    return {
        "n_pairs": n,
        "mean_diff": mean,
        "std_diff": sd,
        "ci_lo": mean - half,
        "ci_hi": mean + half,
        "t_stat": float(t_stat),
        "p_value": _t_sf(t_stat, dof) if np.isfinite(t_stat) else 0.0,
        "cohens_dz": mean / sd if sd > 0 else float("inf") if mean != 0 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed-paired compression comparison.")
    parser.add_argument("--input", type=Path, default=RESULTS / "compression_sweep_perseed.csv")
    parser.add_argument("--out", type=Path, default=RESULTS / "compression_paired.csv")
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(
            f"{args.input} not found. Run run_multiseed.py first; it writes the "
            "per-seed rows this analysis pairs on."
        )

    df = pd.read_csv(args.input)
    if "skip_reason" in df.columns:
        df = df[df["skip_reason"].isna() | (df["skip_reason"] == "")]
    metrics = [m for m in METRICS if m in df.columns]
    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows: list[dict[str, object]] = []
    for key, cell in df.groupby(KEYS):
        base = cell[cell["compression"] == BASELINE].set_index("seed")
        if base.empty:
            continue
        for compression, arm in cell.groupby("compression"):
            if compression == BASELINE:
                continue
            arm = arm.set_index("seed")
            shared = sorted(set(base.index) & set(arm.index))
            if len(shared) < 2:
                continue
            for metric in metrics:
                pairs = arm.loc[shared, metric].to_numpy() - base.loc[shared, metric].to_numpy()
                pairs = pairs[np.isfinite(pairs)]
                if pairs.size < 2:
                    continue
                rows.append(
                    {
                        **dict(zip(KEYS, key, strict=True)),
                        "compression": compression,
                        "metric": metric,
                        **{
                            k: round(v, 6) if isinstance(v, float) else v
                            for k, v in _paired_stats(pairs).items()
                        },
                    }
                )

    if not rows:
        print("no paired comparisons could be formed")
        return 1

    out = pd.DataFrame(rows)
    out["ci_excludes_zero"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    out.to_csv(args.out, index=False)

    print("=== paired difference vs uncompressed: demographic parity ===")
    dp = out[out["metric"] == "demographic_parity_difference"]
    cols = ["dataset", "model", "sensitive", "compression", "n_pairs", "mean_diff"]
    cols += ["ci_lo", "ci_hi", "p_value", "cohens_dz", "ci_excludes_zero"]
    print(dp[cols].to_string(index=False))

    n_sig = int(out["ci_excludes_zero"].sum())
    print(f"\n[paired] {n_sig}/{len(out)} comparisons have a CI excluding zero")
    print(f"[paired] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
