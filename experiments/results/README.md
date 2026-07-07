# Experiment results

Tracked CSV artifacts produced by the scripts in `experiments/`. Regenerate with
the commands below. Per-run manifest directories (`results/<hash>/`) are
git-ignored; only the aggregate CSVs are tracked.

All CSVs are stamped with the seed, a short config hash, and resolved library
versions. Determinism is guaranteed only on Linux/x86 (see the top-level
README); minor float drift elsewhere is expected.

## Files

| File | Script | What it holds |
|------|--------|---------------|
| `baselines.csv` | `run_baselines.py` | One row per (dataset, model): accuracy/F1/ROC-AUC and footprint (params, FLOPs, peak RAM, wall-clock). |
| `compression_sweep.csv` | `run_compression_sweep.py --full` | One row per (dataset, model, sensitive attr, compression): the six fairness+uncertainty metrics and footprint. The headline "does the audit survive compression?" grid. |
| `decoupling_adult.csv` | `run_decoupling.py` | Per sensitive group on Adult: point-fairness (selection rate) vs uncertainty-fairness (mean entropy, per-group ECE) across compression levels. |

Regenerate:

```bash
python experiments/run_baselines.py
python experiments/run_compression_sweep.py --full
python experiments/run_decoupling.py
```

## Headline finding (decoupling)

Point-prediction fairness and uncertainty/calibration fairness are **decoupled**.
On Adult, as the model is pruned the demographic-parity gap *shrinks* (the model
looks fairer on point predictions) while the per-group ECE gap *grows*
(calibration fairness degrades):

| Adult × sex | DP gap | ECE gap |
|-------------|--------|---------|
| none        | 0.177  | 0.0004  |
| prune:0.5   | 0.078  | 0.075   |
| prune:0.9   | 0.020  | 0.122   |

## Caveats (read before citing a number)

- **Folktables is synthetic in an offline run.** The ACSIncome loader falls back
  to a synthetic frame (emitting a warning) when the `folktables` package and
  network are unavailable, so folktables rows here are *not* real ACSIncome
  results (they sit near chance, ROC-AUC ~0.5). Install `folktables` / enable
  network access and rerun for real numbers. Adult and COMPAS are real offline.
- **Decision trees are absent from the compression grid.** `DecisionTreeClassifier`
  has no dense weight matrix to magnitude-prune and no int8 path, so tree cells
  appear only at `compression=none`; the compressed tree cells are skipped by
  design (recorded to stderr, not written).
- **int8 uncertainty columns are blank.** The int8 audited model is an
  ONNX-backed `QuantizedOnnxModel` whose float weights are unreachable, so the
  perturbation ensemble cannot be built (`PerturbNotSupportedError`); the
  uncertainty stage is recorded as skipped. Point-fairness and footprint columns
  are still populated.
- **COMPAS has only ~5 features.** Pruning it past ~0.3 sparsity drives the
  logistic model to a near-constant predictor, which makes DP collapse to 0 and
  disparate impact rise to 1.0 (trivially "fair") while ECE spikes — an extreme
  but honest instance of the decoupling, not a well-conditioned operating point.
