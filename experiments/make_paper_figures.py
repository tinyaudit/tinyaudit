"""Render the paper's figures as PDFs into ``paper/figures/``.

Two figures, each read from the same aggregated CSV the matching paper table
cites, so figure and table can never drift apart:

- ``compression.pdf`` (Finding 2): demographic-parity difference and per-group
  ECE of the Adult MLP by ``sex`` across pruning sparsity, +/- 1 std over 10
  seeds. The web one-pager (web/index.html) draws the same chart.
- ``decoupling.pdf`` (Finding 1): a two-panel rank slope chart. For Adult and
  COMPAS by ``race``, each group's rank by selection/flag rate (left) is joined
  to its rank by calibration ECE (right). Crossing lines are the decoupling: the
  two lenses order the groups differently.

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

# Okabe-Ito categorical ramp (yellow dropped for contrast) for the group lines.
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

# Shorten the longest group names so the slope-chart labels do not collide.
_ABBREV = {
    "Asian-Pac-Islander": "Asian-Pac-Isl",
    "Amer-Indian-Eskimo": "Amer-Ind-Esk",
    "African-American": "African-Am.",
}

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


def _load_decoupling(dataset: str) -> pd.DataFrame:
    """Uncompressed race rows for one dataset, groups with n>=10 kept.

    The n<10 filter matches the paper tables, which drop the tiny COMPAS groups
    (Asian, Native American) whose per-seed metrics are too noisy to rank.
    """
    df = pd.read_csv(RESULTS / f"decoupling_{dataset}_multiseed.csv")
    df = df[(df["compression"] == "none") & (df["sensitive"] == "race")]
    df = df[df["n_mean"] >= 10]
    return df.reset_index(drop=True)


def _decoupling_panel(ax: plt.Axes, df: pd.DataFrame, title: str, rate_label: str) -> None:
    """One rank slope chart: selection/flag rank (left) joined to ECE rank (right)."""
    by_rate = df.sort_values("selection_rate_mean", ascending=False).reset_index(drop=True)
    by_ece = df.sort_values("ece_mean", ascending=True).reset_index(drop=True)
    rate_rank = {g: i + 1 for i, g in enumerate(by_rate["group"])}
    ece_rank = {g: i + 1 for i, g in enumerate(by_ece["group"])}
    colour = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(by_rate["group"])}

    for _, row in df.iterrows():
        g = row["group"]
        y0, y1, c = rate_rank[g], ece_rank[g], colour[g]
        ax.plot([0, 1], [y0, y1], color=c, linewidth=2.0, marker="o", markersize=7, zorder=3)
        name = _ABBREV.get(g, g)
        ax.annotate(
            f"{name}  {row['selection_rate_mean']:.2f}",
            (0, y0),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            color=c,
        )
        ax.annotate(
            f"{row['ece_mean']:.3f}",
            (1, y1),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=c,
        )

    n = len(df)
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(n + 0.5, 0.5)  # inverted: rank 1 sits at the top
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{rate_label}\n(high to low)", "ECE\n(low to high)"], fontsize=9)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)


def decoupling_figure() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    _decoupling_panel(axes[0], _load_decoupling("adult"), "Adult by race", "Selection rate")
    _decoupling_panel(axes[1], _load_decoupling("compas"), "COMPAS by race", "Flag rate")
    fig.suptitle(
        "The fairness lenses decouple: group order flips between selection and calibration",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "decoupling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    for out in (decoupling_figure(), compression_figure()):
        print(f"[figures] wrote {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
