"""Render the paper's figures as PDFs into ``paper/figures/``.

Currently one figure: the compression sweep (Finding 2). It plots the
demographic-parity difference and the per-group ECE of the Adult MLP by
``sex`` against pruning sparsity, with +/- 1 std bands over 10 seeds. It reads
the same aggregated CSV the paper table cites, so the figure and the table can
never drift apart. The web one-pager (web/index.html) draws the same chart.

Run it::

    python experiments/make_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "paper" / "figures"

# Okabe-Ito colourblind-safe pair: blue for parity, vermillion for calibration.
C_DP = "#0072B2"
C_ECE = "#D55E00"

# Pruning levels in sweep order and their tidy axis labels.
_ORDER = ["none", "prune:0.3", "prune:0.5", "prune:0.7", "prune:0.9"]
_LABELS = ["none", "30%", "50%", "70%", "90%"]


def _load_compression() -> pd.DataFrame:
    """Adult MLP by sex across pruning levels, as mean/std columns."""
    df = pd.read_csv(RESULTS / "compression_sweep_multiseed.csv")
    df = df[(df["dataset"] == "adult") & (df["model"] == "mlp") & (df["sensitive"] == "sex")]
    df = df.set_index("compression").reindex(_ORDER)
    return df


def compression_figure() -> Path:
    df = _load_compression()
    dp = df["demographic_parity_difference_mean"].to_numpy()
    dp_s = df["demographic_parity_difference_std"].to_numpy()
    ece = df["mean_ece_per_group_mean"].to_numpy()
    ece_s = df["mean_ece_per_group_std"].to_numpy()
    x = list(range(len(_LABELS)))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for vals, stds, colour, label in (
        (dp, dp_s, C_DP, "Demographic-parity diff"),
        (ece, ece_s, C_ECE, "Per-group ECE"),
    ):
        ax.fill_between(x, vals - stds, vals + stds, color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, vals, color=colour, linewidth=2.0, marker="o", markersize=5, label=label)

    # Direct labels at the endpoints instead of relying on the legend alone.
    ax.annotate(
        f"{dp[-1]:.3f}",
        (x[-1], dp[-1]),
        textcoords="offset points",
        xytext=(6, -10),
        color=C_DP,
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        f"{ece[-1]:.3f}",
        (x[-1], ece[-1]),
        textcoords="offset points",
        xytext=(6, 4),
        color=C_ECE,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS)
    ax.set_xlabel("Pruning sparsity")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 0.42)
    ax.set_title("Compression hides unfairness (Adult MLP, sex)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "compression.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    out = compression_figure()
    print(f"[figures] wrote {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
