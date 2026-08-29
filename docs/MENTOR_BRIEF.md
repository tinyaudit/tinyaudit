# TinyAudit mentor brief

*A short, honest walkthrough of what this project is, why it is novel, and what
we have actually measured so far. Every number below comes from a tracked CSV in
`experiments/results/` and can be regenerated from the scripts named. Headline
metrics are reported as mean ± standard deviation over 10 seeds (0-9); the
per-seed aggregates live in the `*_multiseed.csv` files.*

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
| Asian-Pac-Islander | 0.240 ± 0.014 | 0.318 ± 0.010 | 0.037 ± 0.014 |
| White              | 0.207 ± 0.005 | 0.337 ± 0.007 | **0.008 ± 0.002** ← best calibrated |
| Amer-Indian-Eskimo | 0.098 ± 0.026 | 0.287 ± 0.025 | 0.056 ± 0.019 |
| Black              | 0.091 ± 0.008 | 0.235 ± 0.009 | 0.024 ± 0.006 |
| Other              | 0.081 ± 0.024 | 0.223 ± 0.020 | 0.046 ± 0.016 |

The best-calibrated group by a wide margin (White, ECE 0.008 ± 0.002) is one of
the *most*-selected, not the worst served by the point predictions, so the two
rankings do not line up. The smaller groups (Amer-Indian-Eskimo, Other) carry
higher, noisier calibration error that the selection-rate view never surfaces.
**Which group looks "worst served" depends on which fairness lens you read.** A
demographic-parity-only audit would never see the calibration story. By `sex`,
the Male/Female gap is large in selection rate (0.253 ± 0.007 vs 0.079 ± 0.004)
and in entropy (0.376 vs 0.222) but tiny in calibration (0.009 vs 0.014): loud on
two lenses, silent on a third.

**Takeaway for a mentor:** "fair on demographic parity" does not imply, and does
not predict, "fair in the model's uncertainty or calibration." That is exactly
the claim the uncertainty-aware-fairness literature makes qualitatively; here it
is reproduced with a concrete, regenerable measurement.

## The decoupling replicates on COMPAS (a second, higher-stakes dataset)

The same experiment on COMPAS criminal-risk scoring
(`experiments/run_decoupling.py --dataset compas`, logistic model, features
standardized) shows the same structure. Here the positive outcome is being
flagged **high-risk**, so a *higher* selection rate is *worse* for the group.
By `race` (uncompressed; the two smallest groups, Asian n=5 and Native American
n=3, are omitted as too small to rank):

| group (race)     | n~  | high-risk flag rate | ECE (calibration error) |
|------------------|:-:|:-:|:-:|
| African-American | 791 | **0.441 ± 0.015** ← most flagged | 0.057 ± 0.010 |
| Hispanic         | 129 | 0.249 ± 0.046 | 0.068 ± 0.024 |
| Caucasian        | 526 | 0.227 ± 0.013 | **0.035 ± 0.016** ← best calibrated |
| Other            | 85  | 0.164 ± 0.033 ← least flagged | **0.094 ± 0.029** ← worst calibrated |

African-American defendants are flagged at roughly **twice** the rate of the next
group (0.441 vs 0.249), the well-documented COMPAS racial bias. But the
point-fairness ranking and the calibration ranking still **do not agree**: the
group the model is hardest on by selection rate is *not* the group
whose confidence is least reliable, and the worst-calibrated group ("Other") is
the one flagged *least*. A demographic-parity-only audit would flag
African-American and stop; a calibration-only audit would flag "Other." Both are
right, and they disagree, which is the whole point.

**Why this matters for the write-up:** the headline finding is now shown on two
independent datasets with opposite "favorable" directions (high income is good;
high risk is bad), so it is not an Adult-specific artifact.

## Second result: compression can hide unfairness

From the full compression sweep (`experiments/run_compression_sweep.py --full`),
the Adult **MLP** as it is pruned harder and harder:

| pruning | demographic-parity diff | per-group ECE |
|---------|:-:|:-:|
| none      | 0.175 ± 0.016 | 0.027 ± 0.006 |
| prune 30% | 0.181 ± 0.096 | 0.194 ± 0.108 |
| prune 50% | 0.136 ± 0.080 | 0.193 ± 0.100 |
| prune 70% | 0.084 ± 0.064 | 0.250 ± 0.133 |
| prune 90% | 0.014 ± 0.028 | 0.282 ± 0.106 |

From uncompressed to heavily pruned, the demographic-parity gap net **falls**
(0.175 → 0.014) while per-group calibration error net **climbs roughly tenfold**
(0.027 → 0.282): the model looks fairer on parity precisely as its calibration
collapses, because it degrades into a near-constant predictor. Read this as an
**endpoint contrast** (uncompressed vs heavily pruned), not a smooth monotone
slide: the intermediate pruning levels are seed-noisy (parity std ±0.06 to ±0.10),
so the *direction* is the robust claim, not the exact per-level values. (The small
logistic model has little capacity to lose, so it is robust to pruning; the effect
scales with model capacity.)

## The pruning failure mode replicates on ACSIncome

A reviewer asked whether the "compression can hide unfairness" result is
Adult/COMPAS-specific. `experiments/run_acsincome_pruning.py` re-runs the main
pruning arm on **ACSIncome** (the modern Census replacement for Adult; logistic
model, 3 seeds, test split seeded-subsampled to 8,000 rows to keep it to
minutes). The same failure mode appears:

| ACSIncome (logreg) | none → prune 90% | reading |
|--------------------|:-:|---|
| demographic parity (`sex`) | 0.034 → 0.000 | parity "improves" to perfect |
| accuracy               | 0.781 → 0.589 | but accuracy collapses below deployable |
| positive-prediction rate | 0.395 → 0.000 | because the model becomes a constant predictor |
| per-group ECE disparity (`sex`) | 0.008 → 0.112 | calibration disparity rises ~14× |

Pruning to 90% drives demographic parity to ~0 **not** because the model got
fairer but because it degenerated into a near-constant predictor (positive rate
→ 0, accuracy 0.78 → 0.59) — the "fairest" model in the sweep is the most broken,
exactly the trap the Adult MLP showed. On `sex`, per-group calibration disparity
climbs as parity falls (0.008 → 0.112). On `race` the 90% model is a degenerate
corner (as on heavily-pruned COMPAS), reported rather than hidden. **Caveat:** the
~49k-row test split is subsampled to 8k for speed; race groups still carry
hundreds of rows each. So the headline holds on a *third* dataset.

## Uncertainty is complementary to calibration, not redundant

A reviewer pressed the sharpest question about the framing: ECE already measures
whether the model's confidence is trustworthy, so is "uncertainty-aware" just
"calibration-aware" with extra words? `experiments/run_uncertainty_signal.py`
answers it directly. It computes the *pure* uncertainty quantities the estimator
already produces — per-group predictive entropy, ensemble variance, mutual
information — and asks whether they single out a **different** group than
calibration does. Over 10 seeds the answer is **yes, they are complementary**:

| framing-critical race cell | Spearman(entropy, ECE) | reading |
|----------------------------|:-:|---|
| Adult logreg | −0.40 | uncertainty ranks groups *opposite* to calibration |
| Adult MLP    | +0.18 | weakly aligned at most |
| COMPAS logreg | −0.06 | essentially unrelated |

The mean **|Spearman(entropy, ECE)| is 0.21** across these cells — the two
lenses do not rank the groups the same way. Entropy is not noise, either: on
Adult it tracks the *selection-rate* ordering (Spearman ≈ +0.75), so it is its
own coherent axis, distinct from calibration. On `sex` (2 groups) the same story
shows as a disparity contrast — the entropy gap is large exactly where the
calibration gap is negligible (Adult **0.154 vs 0.006**; Adult MLP 0.155 vs
0.023): uncertainty is loud precisely where calibration is silent. The one cell
where the two lenses instead agree (COMPAS `sex`, ECE louder than entropy) is
reported, not hidden.

**Takeaway for a mentor:** the "uncertainty-aware" framing is *earned*, not
decorative — a calibration-only audit would rank the groups differently and miss
what the uncertainty lens surfaces. `paper/figures/uncertainty_vs_calibration.pdf`
draws this as a rank slope chart; the crossing lines are the whole finding.

## The models are real baselines

`experiments/run_baselines.py`, real datasets (Adult and COMPAS load offline;
Folktables/ACSIncome is real 2018 California Census data via the `folktables`
package):

| dataset | model | accuracy | ROC-AUC | params |
|---------|-------|:-:|:-:|:-:|
| Adult      | logreg | 0.852 | 0.906 | 102 |
| Adult      | MLP    | 0.837 | 0.884 | 4,353 |
| Adult      | tree   | 0.817 | 0.751 | 10,479 |
| COMPAS     | logreg | 0.680 | 0.731 | 6 |
| COMPAS     | MLP    | 0.683 | 0.736 | 1,281 |
| Folktables | logreg | 0.781 | 0.856 | 9 |
| Folktables | MLP    | 0.808 | 0.889 | 1,377 |

These are 10-seed means (accuracy/ROC-AUC std ≤ 0.01 across the board, so the
models are stable). They sit in the published range (Adult logreg ~0.85 accuracy,
COMPAS ~0.68, ACSIncome ~0.78-0.81), which is what makes the fairness/uncertainty
findings above trustworthy.

## How it is built (the rigor to show a mentor)

- **Everything is tested.** 311 passing tests: unit, property-based (Hypothesis),
  and reference-oracle tests that check our in-house metrics against trusted
  oracles (fairness vs Fairlearn/scikit-learn; the experiment's Spearman vs
  scipy).
- **Reproducible.** One `seed` flag seeds every RNG; every result CSV is stamped
  with the seed, a config hash, and library versions.
- **Green CI.** black + ruff + mypy(strict on the metric modules) + the full test
  suite run on every push.
- **Honest by construction.** Findings that look bad (compression wrecking
  calibration) are reported, not hidden; every experiment CSV ships with a README
  of caveats.

## What to read next to a number honestly (caveats)

- Features are standardized before fitting (without it the MLP is unusable).
- All three datasets are real: Adult and COMPAS load offline, Folktables/ACSIncome
  downloads real 2018 California Census data through the `folktables` package.
  (Without that package and network the loader falls back to a clearly-warned
  synthetic frame, used only in restricted-network CI.)
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
compression×fairness grid, the decoupling experiment, the uncertainty-vs-
calibration complementarity check (10 seeds), and the ACSIncome pruning
replication are done and tracked. Target venue: NeurIPS Responsible-AI workshop
2026 (reach: ACM FAccT 2027).
