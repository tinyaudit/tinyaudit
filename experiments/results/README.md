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

Point-prediction fairness and uncertainty/calibration fairness are **decoupled**:
the two lenses rank the sensitive groups differently, so a model that looks fair
(or unfair) on one need not on the other. Numbers below are the uncompressed,
feature-scaled Adult logistic model.

On **race**, the group that is *best* calibrated is not the group that is best
served by the point predictions -- White has the lowest ECE (best calibration)
yet one of the highest selection rates, while Asian-Pac-Islander has both the
highest selection rate and the worst calibration:

| Adult × race | selection rate | mean entropy | ECE |
|--------------|---------------|--------------|-----|
| White              | 0.208 | 0.329 | **0.009** |
| Asian-Pac-Islander | 0.235 | 0.309 | **0.071** |
| Amer-Indian-Eskimo | 0.106 | 0.296 | 0.063 |
| Other              | 0.087 | 0.196 | 0.064 |
| Black              | 0.084 | 0.218 | 0.025 |

On **sex**, the Male/Female disparity is large in selection rate (0.254 vs 0.077)
and in predictive entropy (0.370 vs 0.207) but negligible in calibration
(ECE 0.010 vs 0.007) -- loud on two lenses, silent on the third.

Point-prediction parity therefore neither implies nor predicts uncertainty or
calibration parity.

Compression sharpens the decoupling for higher-capacity models. In
`compression_sweep.csv`, pruning the Adult **MLP** drives its demographic-parity
difference from 0.20 down to 0.00 while its per-group ECE climbs from 0.03 to
0.32 -- it looks progressively *fairer* on point predictions as it silently
loses calibration. The small logistic model has little capacity to lose, so its
gaps barely move under pruning (see the scaling caveat). The
`decoupling_adult.csv` experiment uses the robust logistic model on purpose; the
MLP swing above is read off the sweep.

## Caveats (read before citing a number)

- **Features are standardized before fitting.** Each runner fits a
  `StandardScaler` on the training split and applies it to both splits (the model
  stays a bare estimator so compression and the perturbation ensemble still reach
  its weights). Without this the MLP on Adult is unusable (accuracy ~0.54); with
  it the MLP is a credible baseline (~0.84). Scaling also *conditions* the model,
  which is why the compression-driven decoupling swings are far smaller here than
  on raw features; those large swings were largely a conditioning artifact.
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
- **COMPAS has only ~5 features.** Pruning it to 0.9 sparsity drives the logistic
  model to a near-constant predictor, which makes DP collapse to 0 and disparate
  impact rise to 1.0, "trivially fair" because it treats everyone the same, not
  because it is well behaved. Read heavily-pruned COMPAS rows as a degenerate
  corner case, not a meaningful operating point.
