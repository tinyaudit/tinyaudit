"""Profile the *audit itself*, not the audited model.

Every other script here measures the model's footprint. This one measures what
it costs to *run the audit*: the peak memory the pipeline allocates and the
wall-clock it spends, as a function of how many rows are audited at once.

That framing is the point. An audit's working set is dominated by per-sample
buffers (the feature matrix, the ensemble's stacked prediction arrays, the
explainer's perturbation batch), so peak RAM is a function of the audit batch
size, not of the dataset. If the audit fits in the device's SRAM budget at some
batch size, it can be streamed over an arbitrarily large test set at that batch
size. This script measures the curve so the feasibility claim is quantitative.

Memory is measured with ``tracemalloc``, which does see numpy buffers (numpy
routes array allocations through the Python allocator hooks), so the number is
a real working-set proxy rather than a bookkeeping-only count. It excludes the
interpreter itself, which is the right exclusion: the comparison is against a
compiled on-device implementation of the same arithmetic.

Run it::

    python experiments/run_audit_footprint.py
    python experiments/run_audit_footprint.py --batches 64 256 --seeds 0

Writes ``experiments/results/audit_footprint.csv`` (one row per
dataset/model/batch/seed) and ``audit_footprint_multiseed.csv`` (mean +/- std).
"""

from __future__ import annotations

import argparse
import json
import tracemalloc
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from tinyaudit import audit
from tinyaudit.data import load_adult, load_compas, load_folktables

RESULTS = Path(__file__).resolve().parent / "results"

DATASETS = {"adult": load_adult, "compas": load_compas, "folktables": load_folktables}
SENSITIVE = {"adult": "sex", "compas": "race", "folktables": "sex"}

_KEYS = ["dataset", "model", "batch"]
_METRICS = [
    "audit_peak_ram_bytes",
    "audit_wall_s",
    "profile_s",
    "fairness_s",
    "uncertainty_s",
    "xai_s",
]


def _fit(name: str, seed: int, X: pd.DataFrame, y: pd.Series) -> Any:
    if name == "logreg":
        model: Any = LogisticRegression(max_iter=1000)
    else:
        model = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, random_state=seed)
    model.fit(X, y)
    return model


def _one_run(dataset: str, model_name: str, batch: int, seed: int) -> dict[str, Any] | None:
    loader = DATASETS[dataset]
    X_tr, y_tr, _ = loader(split="train", seed=seed)
    X_te, y_te, s_te = loader(split="test", seed=seed)

    sens = SENSITIVE[dataset]
    if sens not in s_te.columns:
        return None

    scaler = StandardScaler().fit(X_tr.to_numpy())
    Xs_tr = pd.DataFrame(scaler.transform(X_tr.to_numpy()), columns=X_tr.columns)
    model = _fit(model_name, seed, Xs_tr, y_tr)

    n = len(X_te) if batch <= 0 else min(batch, len(X_te))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te), size=n, replace=False)
    Xb = X_te.iloc[idx]
    yb = y_te.iloc[idx]
    sb = s_te[sens].to_numpy()[idx]
    Xb_scaled = pd.DataFrame(scaler.transform(Xb.to_numpy()), columns=Xb.columns, index=Xb.index)

    tracemalloc.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            card = audit(
                model=model,
                data=(Xb_scaled, yb),
                sensitive=sb,
                methods=["fairness", "uncertainty", "xai"],
                seed=seed,
                dataset=dataset,
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    manifest = json.loads(Path(card.manifest_path).read_text())
    stages = manifest["stages"]

    def _sec(stage: str) -> float:
        return float(stages.get(stage, {}).get("seconds", float("nan")))

    stage_total = sum(
        _sec(s) for s in ("profile", "fairness", "uncertainty", "xai") if not np.isnan(_sec(s))
    )
    return {
        "dataset": dataset,
        "model": model_name,
        "batch": n,
        "seed": seed,
        "n_features": Xb.shape[1],
        "audit_peak_ram_bytes": int(peak),
        "audit_wall_s": stage_total,
        "profile_s": _sec("profile"),
        "fairness_s": _sec("fairness"),
        "uncertainty_s": _sec("uncertainty"),
        "xai_s": _sec("xai"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the audit pipeline itself.")
    parser.add_argument("--datasets", nargs="+", default=["adult", "compas"])
    parser.add_argument("--models", nargs="+", default=["logreg", "mlp"])
    parser.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024, 4096])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args(argv)

    # Warm-up. The explainability stage imports shap/lime lazily, so the very
    # first audit in a process pays tens of MB of one-off module-import
    # allocations that tracemalloc would otherwise charge to that run.
    _one_run(args.datasets[0], args.models[0], 64, args.seeds[0])

    rows: list[dict[str, Any]] = []
    for ds in args.datasets:
        for m in args.models:
            for b in args.batches:
                for s in args.seeds:
                    row = _one_run(ds, m, b, s)
                    if row is None:
                        continue
                    rows.append(row)
                    kb = row["audit_peak_ram_bytes"] / 1024
                    print(
                        f"[footprint] {ds}/{m} batch={row['batch']} seed={s} "
                        f"peak={kb:,.0f} KB wall={row['audit_wall_s']:.3f}s",
                        flush=True,
                    )

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "audit_footprint.csv", index=False)

    grouped = df.groupby(_KEYS, dropna=False)
    agg = grouped[_METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = grouped.size()
    agg.reset_index().to_csv(RESULTS / "audit_footprint_multiseed.csv", index=False)
    print(f"[footprint] wrote {len(df)} rows, {len(agg)} groups", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
