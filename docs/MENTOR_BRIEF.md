# TinyAudit mentor brief

*A short, honest walkthrough of what this project is, why it is novel, and what
we have actually measured so far. Every number below comes from a tracked CSV in
`experiments/results/` and can be regenerated from the scripts named.*

## One sentence

TinyAudit audits small, on-device machine-learning models for **fairness,
uncertainty, and explainability at the same time**, inside the same tiny memory
and compute budget the audited model itself runs in, and prints a one-page
"nutrition label" for the model.

## The problem

Two trends collide. (1) ML is moving onto microcontroller-class hardware
(phones, wearables, sensors), where models are aggressively **compressed**
(pruned, quantized) to fit. (2) Responsible-AI audits (is the model fair? does it
know when it is unsure? which features drive it?) are almost always run on big
full-precision models on a workstation. Nobody checks what compression does to
those audit properties, and nobody checks whether the audit itself can run in the
device's budget. TinyAudit does both.

## The core idea

Give it a fitted model and a test set; it runs a seven-stage pipeline and emits a
single traffic-light **audit card** plus a machine-readable manifest:

1. **Profile:** parameters, FLOPs, peak RAM, wall-clock per sample.
2. **Point fairness:** demographic parity, equalized odds, disparate impact.
3. **Uncertainty:** how confident the model is, and how much a small committee
   of near-copies disagrees.
4. **Uncertainty-aware fairness:** per-group predictive entropy, per-group
   calibration error (ECE), and selective fairness (fairness as you let the model
   abstain on its least-confident predictions).
5. **Explainability:** which features matter, and whether the important-feature
   set *flips* between demographic groups.
6. **Compression:** optionally int8-quantize or magnitude-prune first, so every
   metric above is measured **on the compressed model**.
7. **Render:** the one-page card.

## What is novel

- **It couples three audit axes that are normally studied separately** (fairness,
  uncertainty, explainability) and measures them **through compression**.
- **The audit respects an on-device budget**, so it is a feasibility proof, not
  just an offline analysis.
- **It surfaces a concrete, measurable "decoupling" result** (below) that a
  point-fairness-only audit would miss entirely.

## Headline result: fairness is *decoupled* across lenses

The central finding, measured on UCI Adult (logistic model, features
standardized), `experiments/run_decoupling.py`:

**Point-prediction fairness and calibration fairness rank the groups
differently.** By `race`:

| group (race)       | selection rate | mean entropy | ECE (calibration error) |
|--------------------|:-:|:-:|:-:|
| White              | 0.208 | 0.329 | **0.009**  ← best calibrated |
| Asian-Pac-Islander | **0.235** ← most selected | 0.309 | **0.071**  ← worst calibrated |
| Amer-Indian-Eskimo | 0.106 | 0.296 | 0.063 |
| Other              | 0.087 | 0.196 | 0.064 |
| Black              | 0.084 | 0.218 | 0.025 |

The best-calibrated group (White) is *not* the one that is worst served by the
point predictions, and the most-selected group (Asian-Pac-Islander) is also the
*worst* calibrated. **Which group looks "worst served" depends entirely on which
fairness lens you read.** A demographic-parity-only audit would never see the
calibration story. By `sex`, the Male/Female gap is large in selection rate
(0.254 vs 0.077) and in entropy (0.370 vs 0.207) but tiny in calibration (0.010
vs 0.007): loud on two lenses, silent on a third.

**Takeaway for a mentor:** "fair on demographic parity" does not imply, and does
not predict, "fair in the model's uncertainty or calibration." That is exactly
the claim the uncertainty-aware-fairness literature makes qualitatively; here it
is reproduced with a concrete, regenerable measurement.

## Second result: compression can hide unfairness

From the full compression sweep (`experiments/run_compression_sweep.py --full`),
the Adult **MLP** as it is pruned harder and harder:

| pruning | demographic-parity diff | per-group ECE |
|---------|:-:|:-:|
| none      | 0.198 | 0.025 |
| prune 30% | 0.272 | 0.074 |
| prune 50% | 0.229 | 0.167 |
| prune 70% | 0.085 | 0.263 |
| prune 90% | **0.000** | **0.320** |

As it is compressed the model looks **progressively fairer** on demographic
parity (0.20 → 0.00) while its calibration **collapses** (ECE 0.03 → 0.32). A
compressed model can pass a point-fairness check precisely because it has
degraded into a near-constant predictor. (The small logistic model has little
capacity to lose, so it is robust to pruning; the effect scales with model
capacity.)

## The models are real baselines

`experiments/run_baselines.py`, real offline datasets:

| dataset | model | accuracy | ROC-AUC | params |
|---------|-------|:-:|:-:|:-:|
| Adult  | logreg | 0.854 | 0.904 | 102 |
| Adult  | MLP    | 0.841 | 0.886 | 4,353 |
| Adult  | tree   | 0.820 | 0.754 | 10,479 |
| COMPAS | logreg | 0.683 | 0.735 | 6 |
| COMPAS | MLP    | 0.686 | 0.739 | 1,281 |

These are sane, published-range numbers (Adult logreg ~0.85 accuracy, COMPAS
~0.68), which is what makes the fairness/uncertainty findings above trustworthy.

## How it is built (the rigor to show a mentor)

- **Everything is tested.** 216 passing tests: unit, property-based (Hypothesis),
  and reference-oracle tests that check our in-house fairness metrics against
  Fairlearn/scikit-learn.
- **Reproducible.** One `seed` flag seeds every RNG; every result CSV is stamped
  with the seed, a config hash, and library versions.
- **Green CI.** black + ruff + mypy(strict on the metric modules) + the full test
  suite run on every push.
- **Honest by construction.** Findings that look bad (compression wrecking
  calibration) are reported, not hidden; every experiment CSV ships with a README
  of caveats.

## What to read next to a number honestly (caveats)

- Features are standardized before fitting (without it the MLP is unusable).
- Folktables is a synthetic stand-in when its data cannot be downloaded offline;
  Adult and COMPAS are real.
- Decision trees have no dense weights, so they are absent from the compression
  grid; int8 models' weights are unreachable, so their uncertainty columns are
  blank.
- The uncertainty "ensemble" is 5 lightly-jittered copies of the one (compressed)
  model, not 5 independent retrains, so read uncertainty magnitudes as relative,
  and heavily-pruned COMPAS (only ~5 features) as a degenerate corner case.
- Exact reproducibility is guaranteed on Linux; elsewhere expect last-digit float
  drift.

Full caveats: `experiments/results/README.md`.

## Status and target

The instrument is built and green; the study is underway. Baselines, the full
compression×fairness grid, and the decoupling experiment are done and tracked.
Target venue: NeurIPS Responsible-AI workshop 2026 (reach: ACM FAccT 2027).
