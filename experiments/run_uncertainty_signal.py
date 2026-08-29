"""Does the uncertainty signal reveal something calibration does not?

The review's load-bearing question: the paper's uncertainty-aware framing rests
on quantities (per-group ECE, positive rate, confidence spread) that are really
*calibration* and *point* signals. ECE asks whether confidence matches
correctness; it is not itself a measure of predictive uncertainty. If the pure
uncertainty quantities the estimator already produces -- per-group predictive
entropy, ensemble variance, and mutual information (epistemic) -- only re-tell
the calibration or the selection-rate story, then "uncertainty-aware" is
decoration and the honest reframing is "reliability-/calibration-aware". If they
single out a *different* group, the framing earns its place.

This script measures exactly that. For the framing-critical cells it runs the
existing ``audit()`` (fairness + uncertainty only) and, per sensitive group,
records three lenses side by side:

* point fairness   -> selection rate
* calibration      -> per-group ECE
* uncertainty      -> per-group predictive entropy, variance, mutual information

Then it asks whether the *uncertainty* lens is complementary to the other two:

* On attributes with >= 3 groups (race) it computes the Spearman rank
  correlation between the group ordering by entropy/MI and the ordering by ECE
  and by selection rate. Low |rho| == the lenses rank groups differently ==
  complementary.
* On every cell it reports the group *disparity* (max-min) of each lens, so the
  sex case (2 groups, where rank correlation is trivial) is still readable: a
  large entropy disparity where ECE disparity is ~0 is complementarity too.

Output: ``experiments/results/uncertainty_signal.csv`` (one row per cell x seed
x group) plus a printed verdict aggregated over seeds. Every row is stamped with
the seed, a config hash, and resolved library versions.

Run it::

    python experiments/run_uncertainty_signal.py
    python experiments/run_uncertainty_signal.py --seeds 0 1 2 3 4
    python experiments/run_uncertainty_signal.py --cells adult:logreg compas:logreg
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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from tinyaudit import AuditCard, audit
from tinyaudit.data import load_adult, load_compas, load_folktables

RESULTS = Path(__file__).resolve().parent / "results"

DATASETS = {"adult": load_adult, "compas": load_compas, "folktables": load_folktables}
_MODEL_CTORS = {
    "logreg": lambda seed: LogisticRegression(max_iter=1000),
    "mlp": lambda seed: MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, random_state=seed),
}

# The cells the paper's framing actually rests on: the deployable logistic case
# on both datasets, plus the Adult MLP as the high-capacity contrast.
DEFAULT_CELLS = [("adult", "logreg"), ("adult", "mlp"), ("compas", "logreg")]
SENSITIVE_ATTRS = ["sex", "race"]
DEFAULT_COMPRESSIONS = ["none", "prune:0.5", "prune:0.9"]

_VERSION_PKGS = ("tinyaudit", "numpy", "pandas", "scikit-learn", "torch")

_FIELDS = [
    "dataset",
    "model",
    "sensitive",
    "compression",
    "seed",
    "group",
    "n",
    "selection_rate",
    "mean_entropy",
    "mean_variance",
    "mutual_information",
    "ece",
    # Cell-level summaries (repeated across the cell's group rows).
    "selection_disparity",
    "entropy_disparity",
    "variance_disparity",
    "mi_disparity",
    "ece_disparity",
    "spearman_entropy_ece",
    "spearman_mi_ece",
    "spearman_entropy_selection",
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


def _config_hash(dataset: str, model: str, sensitive: str, compression: str, seed: int) -> str:
    payload = json.dumps(
        {
            "dataset": dataset,
            "model": model,
            "sensitive": sensitive,
            "compression": compression,
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _scale_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features (scaler fit on train only) so scale-sensitive models
    train on comparable inputs; names/index preserved for interpretability."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train.to_numpy())
    x_tr = pd.DataFrame(
        scaler.transform(X_train.to_numpy()), columns=X_train.columns, index=X_train.index
    )
    x_te = pd.DataFrame(
        scaler.transform(X_test.to_numpy()), columns=X_test.columns, index=X_test.index
    )
    return x_tr, x_te


def _rank(values: list[float]) -> list[float]:
    """Average ranks of ``values`` (ties share the mean rank). Pure, nan-free."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation between two equal-length sequences.

    Returns ``nan`` when it is undefined: fewer than 3 finite paired points
    (a 2-group attribute makes rank correlation trivially +/-1, so it is not
    reported), or zero variance in either ranking. Pairs containing ``nan`` are
    dropped. scipy is not required; this is a direct Pearson-on-ranks.
    """
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if not (np.isnan(x) or np.isnan(y))]
    if len(pairs) < 3:
        return float("nan")
    xa = np.asarray(_rank([p[0] for p in pairs]))
    xb = np.asarray(_rank([p[1] for p in pairs]))
    if xa.std() == 0.0 or xb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, xb)[0, 1])


def _disparity(values: list[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    if len(finite) >= 2:
        return float(max(finite) - min(finite))
    if len(finite) == 1:
        return 0.0
    return float("nan")


def cell_rows(
    card: AuditCard,
    dataset: str,
    model: str,
    sensitive: str,
    compression: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Reshape one audit card into per-group uncertainty-vs-calibration rows.

    Pure (no I/O). Pairs each group's point (selection rate), calibration (ECE),
    and uncertainty (entropy/variance/MI) signals, and attaches cell-level
    disparities and rank correlations so a group's row carries the whole cell's
    complementarity summary.
    """
    fair_pg = card.fairness.per_group
    unc_pg = card.uncertainty.per_group if card.uncertainty is not None else {}

    groups = list(fair_pg.keys())
    sel = [float(fair_pg[g].get("selection_rate", float("nan"))) for g in groups]
    ent = [float(unc_pg.get(g, {}).get("mean_entropy", float("nan"))) for g in groups]
    var = [float(unc_pg.get(g, {}).get("mean_variance", float("nan"))) for g in groups]
    mi = [float(unc_pg.get(g, {}).get("mutual_information", float("nan"))) for g in groups]
    ece = [float(unc_pg.get(g, {}).get("ece", float("nan"))) for g in groups]

    summary = {
        "selection_disparity": round(_disparity(sel), 6),
        "entropy_disparity": round(_disparity(ent), 6),
        "variance_disparity": round(_disparity(var), 6),
        "mi_disparity": round(_disparity(mi), 6),
        "ece_disparity": round(_disparity(ece), 6),
        "spearman_entropy_ece": round(spearman(ent, ece), 6),
        "spearman_mi_ece": round(spearman(mi, ece), 6),
        "spearman_entropy_selection": round(spearman(ent, sel), 6),
        "config_hash": _config_hash(dataset, model, sensitive, compression, seed),
        "python": platform.python_version(),
        "versions": json.dumps(_versions()),
    }

    rows: list[dict[str, Any]] = []
    for i, g in enumerate(groups):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "sensitive": sensitive,
                "compression": compression,
                "seed": seed,
                "group": g,
                "n": int(fair_pg[g].get("n", 0)),
                "selection_rate": round(sel[i], 6),
                "mean_entropy": round(ent[i], 6),
                "mean_variance": round(var[i], 6),
                "mutual_information": round(mi[i], 6),
                "ece": round(ece[i], 6),
                **summary,
            }
        )
    return rows


def verdict(rows: list[dict[str, Any]]) -> str:
    """Aggregate over seeds and state whether uncertainty is complementary.

    Complementary evidence, on the uncompressed cells only:
      * the group ordering by entropy departs from the ordering by ECE
        (mean |Spearman(entropy, ece)| well below 1), and/or
      * the entropy disparity is non-trivial where the ECE disparity is small
        (the sex case), i.e. uncertainty is loud where calibration is silent.
    """
    df = pd.DataFrame(rows)
    base = df[df["compression"] == "none"]
    # One value per (dataset, model, sensitive): summaries repeat across groups
    # and seeds, so average the per-seed cell summaries.
    cells = base.groupby(["dataset", "model", "sensitive"], dropna=False).agg(
        ent_ece=("spearman_entropy_ece", "mean"),
        ent_sel=("spearman_entropy_selection", "mean"),
        ent_disp=("entropy_disparity", "mean"),
        ece_disp=("ece_disparity", "mean"),
    )
    lines = [
        "",
        "=== uncertainty-vs-calibration verdict (uncompressed) ===",
        cells.round(4).to_string(),
    ]

    race = cells.reset_index()
    race = race[race["sensitive"] == "race"].dropna(subset=["ent_ece"])
    complementary = False
    if not race.empty:
        mean_abs_rho = float(np.mean(np.abs(race["ent_ece"])))
        lines.append(
            f"\nmean |Spearman(entropy, ECE)| on race = {mean_abs_rho:.3f} "
            "(near 0 => uncertainty ranks groups differently from calibration)"
        )
        complementary = mean_abs_rho < 0.7
    # Sex case: entropy loud where ECE is quiet.
    sex = cells.reset_index()
    sex = sex[sex["sensitive"] == "sex"]
    loud_quiet = bool((sex["ent_disp"] > 5 * sex["ece_disp"].clip(lower=1e-6)).any())
    if loud_quiet:
        lines.append(
            "entropy disparity >> ECE disparity on at least one sex cell "
            "(uncertainty loud where calibration is silent)."
        )
    decision = (
        "COMPLEMENTARY -> keep 'uncertainty-aware'"
        if (complementary or loud_quiet)
        else "REDUNDANT with calibration/point -> reframe to 'reliability-/calibration-aware'"
    )
    lines.append(f"\nVERDICT: {decision}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uncertainty-vs-calibration complementarity.")
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="cells as dataset:model (default: adult:logreg adult:mlp compas:logreg)",
    )
    parser.add_argument("--sensitive", nargs="+", choices=SENSITIVE_ATTRS, default=SENSITIVE_ATTRS)
    parser.add_argument("--compressions", nargs="+", default=DEFAULT_COMPRESSIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.cells is None:
        cells = DEFAULT_CELLS
    else:
        cells = [(c.split(":", 1)[0], c.split(":", 1)[1]) for c in args.cells]

    out_path = args.out or (RESULTS / "uncertainty_signal.csv")
    all_rows: list[dict[str, Any]] = []

    for dataset, model in cells:
        loader = DATASETS[dataset]
        for seed in args.seeds:
            X_train, y_train, _ = loader(split="train", seed=seed)
            X_test, y_test, s_test = loader(split="test", seed=seed)
            X_train, X_test = _scale_split(X_train, X_test)
            est = _MODEL_CTORS[model](seed)
            est.fit(X_train.to_numpy(), y_train.to_numpy())

            for sensitive in args.sensitive:
                for compression in args.compressions:
                    print(
                        f"[unc-signal] {dataset}:{model} x {sensitive} x {compression} "
                        f"x seed{seed} ...",
                        flush=True,
                    )
                    try:
                        card = audit(
                            est,
                            data=(X_test, y_test),
                            sensitive=s_test[sensitive],
                            compression=None if compression == "none" else compression,
                            seed=seed,
                            dataset=dataset,
                            methods=["fairness", "uncertainty"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ! skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
                        continue
                    all_rows.extend(cell_rows(card, dataset, model, sensitive, compression, seed))

    if not all_rows:
        print("no rows produced (all cells failed)", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[unc-signal] wrote {len(all_rows)} rows -> {out_path}")
    print(verdict(all_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
