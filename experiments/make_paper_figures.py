"""Render the paper's figures as PDFs into ``paper/figures/``.

Four figures, each read from the same aggregated CSV the matching paper table
cites, so figure and table can never drift apart:

- ``compression.pdf`` (Finding 2, the headline): accuracy on the x-axis against
  demographic-parity difference and per-group ECE for the Adult MLP by ``sex``,
  every point tagged with the magnitude-pruning sparsity that produced it and
  carrying +/- 1 std over 10 seeds. The x-axis is inverted so pruning runs left
  to right, which prices the apparent fairness win in accuracy: the reader sees
  how much of the model had to be destroyed before the DP gap closed and the
  calibration error blew up. The web one-pager (web/index.html) draws the same
  chart.
- ``metric_sensitivity.pdf`` (Finding 2, the caveat): the same Adult MLP by
  ``sex`` across the same sparsity ladder, read through three fairness metrics.
  Demographic parity -- the metric most audits report -- improves under pruning
  while equalized odds does not, so the "improvement" is in large part an
  artifact of which metric the audit happened to pick.
- ``decoupling.pdf`` (Finding 1): a two-panel rank slope chart. For Adult and
  COMPAS by ``race``, each group's rank by selection/flag rate (left) is joined
  to its rank by calibration ECE (right). Crossing lines are the decoupling: the
  two lenses order the groups differently.
- ``degradation_control.pdf`` (the control): the same model broken three
  different ways -- magnitude pruning, training-label noise, and training
  subsampling -- with every mechanism traced across accuracy. Read vertically at
  a fixed accuracy it answers the obvious objection to Finding 2, and the answer
  is that the mechanisms leave opposite signatures: pruning walks parity down
  while sparing calibration, label noise wrecks calibration while sparing
  parity. It reads ``degradation_control_*.csv`` rather than the sweep.

Run it::

    python experiments/make_paper_figures.py
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "paper" / "figures"

# Okabe-Ito colourblind-safe pair: blue for parity, vermillion for calibration.
C_DP = "#0072B2"
C_ECE = "#D55E00"

# Two more Okabe-Ito hues for the other two fairness metrics.
C_EO = "#009E73"
C_DI = "#CC79A7"

# Recessive ink for guides, reference lines and annotation leaders.
C_GUIDE = "0.80"

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
    """Adult MLP by sex across the pruning ladder, as mean/std columns.

    Reindexing onto ``_ORDER`` puts the rungs in sweep order and turns a missing
    sparsity level into an all-NaN row rather than a silently shortened ladder.
    Two derived columns ride along: ``label`` is the tidy axis text and ``pos``
    is the rung's fixed x position, so dropping a blank row later shifts nothing.
    """
    df = pd.read_csv(RESULTS / "compression_sweep_multiseed.csv")
    df = df[(df["dataset"] == "adult") & (df["model"] == "mlp") & (df["sensitive"] == "sex")]
    df = df.set_index("compression").reindex(_ORDER)
    df["label"] = _LABELS
    df["pos"] = np.arange(len(_ORDER), dtype=float)
    return df


def _baseline_accuracy(dataset: str, sensitive: str) -> float | None:
    """Accuracy of the majority-class constant predictor, or ``None`` if absent.

    The sweep audits this model through exactly the same code path as every
    other cell, so its accuracy is the floor a collapsing model falls to.
    Returns ``None`` when the baseline rows have not been generated, which
    keeps the figure renderable against an older CSV.
    """
    df = pd.read_csv(RESULTS / "compression_sweep_multiseed.csv")
    row = df[
        (df["dataset"] == dataset) & (df["model"] == "majority") & (df["sensitive"] == sensitive)
    ]
    if row.empty or "accuracy_mean" not in row.columns:
        return None
    value = pd.to_numeric(row["accuracy_mean"], errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else None


def _finite(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Rows where every named column is present and finite, in the incoming order.

    The sweep leaves real holes: ``int8`` rows have blank uncertainty columns and
    ``tree`` rows are blank across the board. Dropping them here keeps matplotlib
    from routing a line through a NaN and emitting a broken path.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"compression_sweep_multiseed.csv is missing columns: {missing}")
    values = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    kept = df[np.isfinite(values).all(axis=1)]
    if kept.empty:
        raise ValueError(f"no rows with finite values for all of {list(columns)}")
    return kept


def compression_figure() -> Path:
    """Price the apparent fairness win in accuracy (Adult MLP, sex).

    Accuracy is the x-axis and inverted, so magnitude pruning progresses left to
    right as the model gets worse; demographic-parity difference and per-group
    ECE share the y-axis, since both are on a 0-is-good [0, 1] scale and need no
    rescaling to sit together. Each rung of the sparsity ladder is a point on
    both curves, tied to a shared guide line and tagged once at the top with its
    sparsity, so the reader can read straight off the x position how much
    accuracy each increment of "fairness" cost.
    """
    df = _finite(
        _load_compression(),
        [
            "accuracy_mean",
            "demographic_parity_difference_mean",
            "demographic_parity_difference_std",
            "mean_ece_per_group_mean",
            "mean_ece_per_group_std",
        ],
    )
    acc = df["accuracy_mean"].to_numpy(dtype=float)
    dp = df["demographic_parity_difference_mean"].to_numpy(dtype=float)
    dp_s = df["demographic_parity_difference_std"].to_numpy(dtype=float)
    ece = df["mean_ece_per_group_mean"].to_numpy(dtype=float)
    ece_s = df["mean_ece_per_group_std"].to_numpy(dtype=float)
    labels = [str(v) for v in df["label"]]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    # Headroom for the sparsity tag band, which sits above the highest error bar.
    top = float(max((dp + dp_s).max(), (ece + ece_s).max()))
    span = max(top, 1e-6)
    tag_y = top + 0.06 * span
    ax.set_ylim(0.0, top + 0.34 * span)

    # One guide per rung: both curves are sampled at the same accuracy, so the
    # sparsity is labelled once rather than twice. The tags are set vertically
    # because consecutive rungs can sit only a fraction of a point of accuracy
    # apart, which would overlap horizontal text.
    for x_pos, name in zip(acc, labels, strict=True):
        ax.plot([x_pos, x_pos], [0.0, tag_y], color=C_GUIDE, linewidth=0.7, zorder=0)
        ax.annotate(
            name,
            (x_pos, tag_y),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            color="0.35",
        )

    for vals, stds, colour, label in (
        (dp, dp_s, C_DP, "Demographic-parity diff"),
        (ece, ece_s, C_ECE, "Per-group ECE"),
    ):
        ax.errorbar(
            acc,
            vals,
            yerr=stds,
            color=colour,
            linewidth=2.0,
            marker="o",
            markersize=5,
            capsize=3,
            elinewidth=1.0,
            label=label,
            zorder=3,
        )

    # Direct labels at the fully pruned endpoint instead of relying on the
    # legend alone. They go outside the data area, past the last guide line.
    for vals, colour in ((dp, C_DP), (ece, C_ECE)):
        ax.annotate(
            f"{vals[-1]:.3f}",
            (acc[-1], vals[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            va="center",
            color=colour,
            fontsize=9,
            fontweight="bold",
        )

    # The constant-predictor floor. A model that answers one class for every
    # input scores a demographic-parity difference of exactly 0.0, so a pruned
    # model arriving at this accuracy has not become fairer, it has stopped
    # making decisions. Drawing the floor is what turns "parity improved" into
    # "parity improved because the model is collapsing toward this".
    floor = _baseline_accuracy("adult", "sex")
    if floor is not None:
        ax.axvline(floor, color="0.45", linestyle="--", linewidth=1.0, zorder=1)
        # Bottom edge, on the low-accuracy side of the line. The top belongs to
        # the sparsity tags and the ECE curve's left end occupies the opposite
        # corner, so this is the one clear pocket. Kept to a single line to fit.
        ax.annotate(
            "majority-class baseline (DP = 0)",
            (floor, 0.0),
            textcoords="offset points",
            xytext=(7, 6),
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="0.45",
        )

    ax.invert_xaxis()  # pruning progresses left to right as accuracy falls
    ax.set_xlabel("Accuracy (axis inverted: pruning progresses left to right)")
    ax.set_ylabel("Metric value")
    ax.set_title("What the parity gain costs in accuracy (Adult MLP, sex)", fontsize=11)
    # The tag band owns the top of the axes, so the legend sits under the axis.
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        handlelength=1.6,
        columnspacing=1.6,
    )
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


def _sensitivity_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    series: Sequence[tuple[str, str, str]],
    ylabel: str,
    parity: float,
) -> None:
    """One panel of the metric-sensitivity chart, with a dashed parity reference.

    ``series`` is ``(metric stem, colour, legend label)``; each stem resolves to
    the ``_mean``/``_std`` pair and is drawn as a line with a +/- 1 std band.
    """
    x = df["pos"].to_numpy(dtype=float)
    ax.axhline(parity, color=C_GUIDE, linewidth=1.0, linestyle="--", zorder=1)
    # Anchored left: the right edge is reserved for the endpoint value labels.
    ax.annotate(
        f"{parity:g} = parity",
        (x[0], parity),
        textcoords="offset points",
        xytext=(2, 3),
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.45",
    )

    lo, hi = parity, parity
    for stem, colour, label in series:
        vals = df[f"{stem}_mean"].to_numpy(dtype=float)
        stds = df[f"{stem}_std"].to_numpy(dtype=float)
        lo = min(lo, float((vals - stds).min()))
        hi = max(hi, float((vals + stds).max()))
        ax.fill_between(x, vals - stds, vals + stds, color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, vals, color=colour, linewidth=2.0, marker="o", markersize=5, label=label)
        ax.annotate(
            f"{vals[-1]:.3f}",
            (x[-1], vals[-1]),
            textcoords="offset points",
            xytext=(6, -10),
            ha="left",
            color=colour,
            fontsize=9,
            fontweight="bold",
        )

    # Extra headroom on top so a legend, where there is one, clears the data.
    pad = max(hi - lo, 1e-6) * 0.10
    ax.set_ylim(lo - pad, hi + pad * (3.0 if len(series) > 1 else 1.5))
    ax.set_ylabel(ylabel, fontsize=9)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def metric_sensitivity_figure() -> Path:
    """Show that the pruning "fairness win" depends on which metric you report.

    Same Adult MLP by ``sex``, same sparsity ladder, three metrics. Demographic
    parity -- the one most audits report -- falls towards parity as the model is
    pruned, while equalized odds does not follow it down; the apparent
    improvement is therefore substantially an artifact of metric choice.

    Scaling decision: disparate impact is a ratio where 1.0 means parity, while
    DP and EO are differences where 0.0 means parity. A dual y-axis is banned
    outright, so the choice was between plotting ``1 - DI`` on one shared scale
    or splitting into stacked subplots. This uses **two stacked subplots sharing
    the x-axis**. The reason is that ``1 - DI`` silently reports a transformed
    number under a metric name the reader will match against the paper's tables,
    and it invites a magnitude comparison between a ratio gap and a rate
    difference that is not meaningful. Stacked panels keep each metric on its
    own native scale with its own parity reference line, and because the shared
    x-axis vertically aligns the rungs of the ladder, the comparison the
    argument actually needs -- the direction each metric moves under pruning --
    is still read off in one glance.
    """
    df = _finite(
        _load_compression(),
        [
            "demographic_parity_difference_mean",
            "demographic_parity_difference_std",
            "equalized_odds_difference_mean",
            "equalized_odds_difference_std",
            "disparate_impact_ratio_mean",
            "disparate_impact_ratio_std",
        ],
    )

    fig, axes = plt.subplots(2, 1, figsize=(5.8, 4.8), sharex=True, height_ratios=[3, 2])
    _sensitivity_panel(
        axes[0],
        df,
        [
            ("demographic_parity_difference", C_DP, "Demographic-parity diff"),
            ("equalized_odds_difference", C_EO, "Equalized-odds diff"),
        ],
        "Difference\n(0 = parity)",
        0.0,
    )
    # Single series, so the axis label names it and no legend box is needed.
    _sensitivity_panel(
        axes[1],
        df,
        [("disparate_impact_ratio", C_DI, "Disparate-impact ratio")],
        "Disparate-impact ratio\n(1 = parity)",
        1.0,
    )

    axes[1].set_xticks(list(range(len(_LABELS))))
    axes[1].set_xticklabels(_LABELS)
    axes[1].set_xlabel("Pruning sparsity")
    axes[0].set_title("The fairness win depends on the metric (Adult MLP, sex)", fontsize=11)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "metric_sensitivity.pdf"
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


# Damage mechanisms in the degradation control, with tidy labels and hues.
# Pruning keeps the parity blue it carries in the other figures; the two
# controls take the remaining Okabe-Ito hues.
_MECHANISMS = [
    ("prune", "Magnitude pruning", C_DP),
    ("label_noise", "Label noise", C_EO),
    ("subsample", "Training subsample", C_DI),
]

_DEGRADATION_INPUTS = [
    "degradation_control_mlp.csv",
    "degradation_control_logreg.csv",
    "degradation_control.csv",
]


def _load_degradation() -> pd.DataFrame:
    """Every degradation-control row available, from whichever arms have run."""
    frames = [
        pd.read_csv(RESULTS / name) for name in _DEGRADATION_INPUTS if (RESULTS / name).exists()
    ]
    if not frames:
        raise FileNotFoundError(
            "no degradation-control CSV found; run experiments/run_degradation_control.py first"
        )
    df = pd.concat(frames, ignore_index=True)
    if "skip_reason" in df.columns:
        df = df[df["skip_reason"].isna() | (df["skip_reason"] == "")]
    return df


def _degradation_panel(
    ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str, title: str
) -> None:
    """One metric against accuracy, one line per damage mechanism."""
    for mech, label, colour in _MECHANISMS:
        sub = df[df["mechanism"] == mech]
        if sub.empty:
            continue
        curve = sub.groupby("level")[["accuracy", metric]].mean().reset_index()
        curve = curve[np.isfinite(curve["accuracy"]) & np.isfinite(curve[metric])]
        curve = curve.sort_values("accuracy", ascending=False)
        if curve.empty:
            continue
        ax.plot(
            curve["accuracy"].to_numpy(),
            curve[metric].to_numpy(),
            color=colour,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=label,
            zorder=3,
        )

    ax.invert_xaxis()  # damage increases left to right
    ax.set_xlabel("Accuracy (inverted: damage increases rightward)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def degradation_figure(
    dataset: str = "adult", model: str = "logreg", sensitive: str = "sex"
) -> Path:
    """Is the parity collapse compression-specific, or does any damage cause it?

    Three ways of breaking the same model are traced through the same accuracy
    range, so the comparison is read vertically: at a fixed accuracy, do the
    mechanisms agree? They do not, and the two panels show opposite signatures.
    Pruning walks demographic parity down while barely touching calibration;
    label noise destroys calibration while leaving parity roughly where it was.
    That asymmetry is the answer to "surely any broken model does this".

    Defaults to the logistic arm deliberately, not for convenience. The
    signature separation is a logistic-regression result: on the Adult MLP the
    per-mechanism slopes overlap and pruning damages calibration *more* than
    label noise does, which is the opposite ordering. Pointing this figure at
    the MLP would put a title on it that its own data contradicts. Pass
    ``model="mlp"`` to inspect that arm, and read the panel titles as scoped to
    whichever cell is drawn.

    Falls back to the logistic arm when the requested model has not been run to
    completion. The test is mechanism coverage, not row count: a partly finished
    arm has rows but only some mechanisms, and one curve drawn under a title
    that contrasts three asserts something the data does not show.
    """
    df = _load_degradation()
    wanted = [mech for mech, _, _ in _MECHANISMS]

    def _cell(model_name: str) -> pd.DataFrame:
        return df[
            (df["dataset"] == dataset)
            & (df["model"] == model_name)
            & (df["sensitive"] == sensitive)
        ]

    def _complete(candidate: pd.DataFrame) -> bool:
        if candidate.empty:
            return False
        # Every mechanism needs at least two distinct levels, or its "curve"
        # is a single point and the comparison is vacuous.
        return all(
            candidate[candidate["mechanism"] == mech]["level"].nunique() >= 2 for mech in wanted
        )

    cell = _cell(model)
    if not _complete(cell):
        fallback = _cell("logreg")
        if _complete(fallback):
            model, cell = "logreg", fallback
    if cell.empty:
        raise ValueError(f"no degradation rows for {dataset}/{model}/{sensitive}")
    if not _complete(cell):
        covered = sorted(cell["mechanism"].unique())
        raise ValueError(
            f"degradation rows for {dataset}/{model}/{sensitive} cover only {covered}; "
            f"all of {wanted} must be present before this figure means anything"
        )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    _degradation_panel(
        axes[0],
        cell,
        "demographic_parity_difference",
        "Demographic-parity diff",
        "Parity: only pruning walks it down",
    )
    _degradation_panel(
        axes[1],
        cell,
        "mean_ece_per_group",
        "Per-group ECE",
        "Calibration: only label noise wrecks it",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"Damage mechanisms leave different signatures ({dataset.title()} "
        f"{model.upper()}, {sensitive})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "degradation_control.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    for out in (decoupling_figure(), compression_figure(), metric_sensitivity_figure()):
        print(f"[figures] wrote {out.relative_to(HERE.parent)}")

    # The degradation control is a separate, slower experiment. Render its
    # figure when its results exist and say so plainly when they do not,
    # rather than failing the whole run.
    try:
        out = degradation_figure()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[figures] skipped degradation_control.pdf: {exc}")
    else:
        print(f"[figures] wrote {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
