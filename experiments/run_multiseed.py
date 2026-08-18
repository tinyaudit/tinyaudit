"""Multi-seed aggregation: run the experiment scripts over seeds 0-9 and write
mean +/- std summary CSVs into ``experiments/results/``.

Every headline number in the brief and the results one-pager is reported as a
mean +/- standard deviation over these seeds; this script is how those
``*_multiseed.csv`` files are produced. Scope is chosen to keep the runtime
tractable while covering every headline number:

- ``baselines``: all datasets.
- ``decoupling``: Adult and COMPAS (the two datasets the finding is stated on).
- ``compression_sweep``: Adult and COMPAS. The Folktables sweep is excluded here
  because its ~49k-row test set makes a 10-seed sweep a multi-hour job for a
  supporting result; its single-seed rows live in ``compression_sweep.csv``.

Run it::

    python experiments/run_multiseed.py
    python experiments/run_multiseed.py --seeds 0 1 2      # fewer seeds

Determinism is guaranteed only on Linux/x86 (see the top-level README); minor
float drift elsewhere is expected, which is exactly why the std columns matter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Identifier columns per experiment; every other numeric column is a metric that
# gets aggregated. Seed/hash/version columns are dropped before aggregation.
_NONMETRIC = {"seed", "config_hash", "python", "versions", "skip_reason"}
_KEYS = {
    "baselines": ["dataset", "model"],
    "compression_sweep": ["dataset", "model", "sensitive", "compression"],
    "decoupling": ["dataset", "sensitive", "compression", "group"],
}


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd[1:]), flush=True)
    proc = subprocess.run(cmd, cwd=HERE.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def _aggregate(name: str, keys: list[str], per_seed: list[Path]) -> None:
    frames = [pd.read_csv(p) for p in per_seed if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    metric_cols = [c for c in df.columns if c not in keys and c not in _NONMETRIC]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    grouped = df.groupby(keys, dropna=False)
    agg = grouped[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = grouped.size()

    # Carry the skip reason through instead of dropping it. A cell that could
    # not run aggregates to a row of NaNs, and without this the reader of the
    # summary CSV has no way to tell an unsupported cell from a crashed one.
    if "skip_reason" in df.columns:
        agg["skip_reason"] = grouped["skip_reason"].agg(
            lambda col: next((str(v) for v in col if isinstance(v, str) and v.strip()), "")
        )

    # Keep the per-seed rows next to the summary. Paired analysis (seed 0
    # uncompressed against seed 0 pruned, and so on) needs them, and it is far
    # more sensitive than comparing two overlapping mean +/- std bands, because
    # it cancels the fact that some seeds simply produce better models. The
    # aggregate alone cannot support that comparison.
    df.to_csv(RESULTS / f"{name}_perseed.csv", index=False)

    out = RESULTS / f"{name}_multiseed.csv"
    agg.reset_index().to_csv(out, index=False)
    print(f"[multiseed] wrote {out.name}: {len(agg)} groups over {df['seed'].nunique()} seeds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate experiments over seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["baselines", "decoupling", "compression_sweep"],
        default=["baselines", "decoupling", "compression_sweep"],
        help="re-run only these experiments (the sweep is the slow one)",
    )
    args = parser.parse_args(argv)
    py = sys.executable
    only = set(args.only)

    with tempfile.TemporaryDirectory(prefix="tinyaudit_multiseed_") as tmp:
        tmpd = Path(tmp)

        if "baselines" in only:
            print("=== baselines (all datasets) ===", flush=True)
            paths = []
            for s in args.seeds:
                p = tmpd / f"baselines_{s}.csv"
                paths.append(p)
                _run([py, str(HERE / "run_baselines.py"), "--seed", str(s), "--out", str(p)])
            _aggregate("baselines", _KEYS["baselines"], paths)

        for ds in ("adult", "compas") if "decoupling" in only else ():
            print(f"=== decoupling {ds} ===", flush=True)
            paths = []
            for s in args.seeds:
                p = tmpd / f"decoup_{ds}_{s}.csv"
                paths.append(p)
                _run(
                    [
                        py,
                        str(HERE / "run_decoupling.py"),
                        "--dataset",
                        ds,
                        "--seed",
                        str(s),
                        "--out",
                        str(p),
                    ]
                )
            _aggregate(f"decoupling_{ds}", _KEYS["decoupling"], paths)

        if "compression_sweep" in only:
            print("=== compression sweep (adult, compas) ===", flush=True)
            paths = []
            for s in args.seeds:
                p = tmpd / f"sweep_{s}.csv"
                paths.append(p)
                _run(
                    [
                        py,
                        str(HERE / "run_compression_sweep.py"),
                        "--datasets",
                        "adult",
                        "compas",
                        "--full",
                        "--seed",
                        str(s),
                        "--out",
                        str(p),
                    ]
                )
            _aggregate("compression_sweep", _KEYS["compression_sweep"], paths)

    print("[multiseed] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
