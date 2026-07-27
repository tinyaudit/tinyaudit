"""Do the model's explanations depend on the sensitive group, and does
compression make that worse?

This is the explainability counterpart to ``run_decoupling.py``. That script
asks whether point fairness and calibration fairness single out different
groups. This one asks a third question about the same models: does the model
reach its predictions using *different features* for different groups, and does
compressing it change which features it leans on?

The measurement is the occlusion explainer's per-group importance. For each
sensitive group we rank features by mean |attribution| and take the top k. A
feature "flips" when it is in some group's top k but not in every group's, i.e.
the model treats it as decisive for one group and ignorable for another. See
``tinyaudit.xai.occlusion.importance_flips`` for why top-k membership rather
than exact rank order is the noise-robust notion here.

Two output files:

* ``xai_flips.csv``       one row per (dataset, model, sensitive, compression,
                          seed): the flip count, the flip rate, and the flipped
                          feature names.
* ``xai_group_top.csv``   one row per (…, group): that group's top-k features,
                          for reading which specific features moved.

Occlusion is used rather than SHAP or LIME because it is what ``audit()``
actually runs in the XAI stage, and because it is the only one of the three
cheap enough to belong in the on-device footprint claim.

Usage::

    python experiments/run_xai_flips.py                     # adult, seeds 0-9
    python experiments/run_xai_flips.py --dataset compas
    python experiments/run_xai_flips.py --seeds 0 1 2 --top-k 5
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from tinyaudit import AuditCard, audit
from tinyaudit.data import load_adult, load_compas

RESULTS = Path(__file__).resolve().parent / "results"

DATASETS = {"adult": load_adult, "compas": load_compas}
SENSITIVE_ATTRS = ["sex", "race"]

# Uncompressed plus the pruning ladder used in the compression sweep, so the
# flip counts line up with the fairness numbers reported alongside them. int8 is
# excluded: the quantized model is ONNX-backed and the occlusion explainer needs
# repeated predict_proba calls on perturbed inputs, which is exactly the path
# that is slow there.
COMPRESSIONS = ["none", "prune:0.3", "prune:0.5", "prune:0.7", "prune:0.9"]

_VERSION_PKGS = ["numpy", "pandas", "scikit-learn", "torch"]


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _VERSION_PKGS:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "unavailable"
    return out


def _scale_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features, scaler fit on train only. Column names are kept so
    the flipped features can be reported by name rather than by index."""
    scaler = StandardScaler().fit(X_train.to_numpy())
    x_tr = pd.DataFrame(
        scaler.transform(X_train.to_numpy()), columns=X_train.columns, index=X_train.index
    )
    x_te = pd.DataFrame(
        scaler.transform(X_test.to_numpy()), columns=X_test.columns, index=X_test.index
    )
    return x_tr, x_te


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


def _build_model(name: str, seed: int) -> Any:
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300, random_state=seed)
    raise ValueError(f"unknown model {name!r}")


def flip_rows(
    card: AuditCard,
    dataset: str,
    model: str,
    sensitive: str,
    compression: str,
    seed: int,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reshape one audit card into a flip-summary row plus per-group top-k rows.

    Pure (no I/O). If the XAI stage was skipped the summary row still exists but
    carries nan counts, so a skipped stage is visible in the CSV rather than
    silently absent.
    """
    stamp = {
        "seed": seed,
        "config_hash": _config_hash(dataset, model, sensitive, compression, seed),
        "python": platform.python_version(),
        "versions": json.dumps(_versions()),
    }
    xai = card.explainability

    if xai is None:
        summary = {
            "dataset": dataset,
            "model": model,
            "sensitive": sensitive,
            "compression": compression,
            "top_k": top_k,
            "n_features": 0,
            "n_groups": 0,
            "n_flips": float("nan"),
            "flip_rate": float("nan"),
            "flipped_features": "",
            **stamp,
        }
        return summary, []

    n_groups = len(xai.per_group_importance)
    n_features = len(next(iter(xai.per_group_importance.values()), {}))
    n_flips = len(xai.importance_flips)
    # Normalize by the shortlist size, not the feature count: with k slots per
    # group the flip count is bounded by roughly k*(n_groups-1), so flip_rate is
    # comparable across datasets with very different feature counts.
    denom = float(top_k) if top_k > 0 else float("nan")

    summary = {
        "dataset": dataset,
        "model": model,
        "sensitive": sensitive,
        "compression": compression,
        "top_k": top_k,
        "n_features": n_features,
        "n_groups": n_groups,
        "n_flips": n_flips,
        "flip_rate": round(n_flips / denom, 6),
        "flipped_features": "|".join(xai.importance_flips),
        **stamp,
    }

    group_rows: list[dict[str, Any]] = []
    for group, imp in xai.per_group_importance.items():
        ranked = sorted(imp.items(), key=lambda kv: -abs(kv[1]))[:top_k]
        group_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "sensitive": sensitive,
                "compression": compression,
                "group": group,
                "top_k": top_k,
                "top_features": "|".join(name for name, _ in ranked),
                "top_importances": "|".join(f"{val:.6f}" for _, val in ranked),
                **stamp,
            }
        )
    return summary, group_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure per-group explanation flips across compression levels."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="adult")
    parser.add_argument("--models", nargs="+", default=["logreg", "mlp"])
    parser.add_argument("--sensitive", nargs="+", choices=SENSITIVE_ATTRS, default=SENSITIVE_ATTRS)
    parser.add_argument("--compressions", nargs="+", default=COMPRESSIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Subsample the test set. Occlusion is O(n_features) predict_proba "
        "passes, so the full Adult test set at 101 features is slow with no "
        "change to the ranking.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--group-out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or RESULTS / f"xai_flips_{args.dataset}.csv"
    group_out = args.group_out or RESULTS / f"xai_group_top_{args.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    loader = DATASETS[args.dataset]

    summaries: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        for seed in args.seeds:
            # Reload per seed. The seed drives the train/test split as well as
            # the estimator's random_state, matching run_decoupling.py. Holding
            # the split fixed and varying only random_state makes a logistic
            # model fully deterministic, which reports a spurious +-0.0 spread.
            X_train, y_train, _ = loader(split="train", seed=seed)
            X_test, y_test, sens_test = loader(split="test", seed=seed)
            X_train, X_test = _scale_split(X_train, X_test)

            if args.limit and args.limit < len(X_test):
                X_test = X_test.iloc[: args.limit]
                y_test = y_test.iloc[: args.limit]
                sens_test = sens_test.iloc[: args.limit]

            clf = _build_model(model_name, seed)
            clf.fit(X_train.to_numpy(), y_train.to_numpy())

            for attr in args.sensitive:
                for comp in args.compressions:
                    try:
                        card = audit(
                            model=clf,
                            data=(X_test, y_test),
                            sensitive=sens_test[attr],
                            methods=["fairness", "xai"],
                            compression=None if comp == "none" else comp,
                            seed=seed,
                            dataset=args.dataset,
                            xai_top_k=args.top_k,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"  skip {model_name}/{attr}/{comp}/seed{seed}: {exc}",
                            file=sys.stderr,
                        )
                        continue

                    summary, groups = flip_rows(
                        card, args.dataset, model_name, attr, comp, seed, args.top_k
                    )
                    summaries.append(summary)
                    group_rows.extend(groups)
                    print(
                        f"  {model_name:6s} {attr:5s} {comp:10s} seed{seed}: "
                        f"{summary['n_flips']} flips of top-{args.top_k} "
                        f"across {summary['n_groups']} groups"
                    )

    if not summaries:
        print("no rows produced", file=sys.stderr)
        return 1

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {out} ({len(summaries)} rows)")

    if group_rows:
        with group_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(group_rows[0].keys()))
            writer.writeheader()
            writer.writerows(group_rows)
        print(f"wrote {group_out} ({len(group_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
