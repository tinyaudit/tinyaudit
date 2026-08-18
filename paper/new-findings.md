# New findings from the post-review experiments

Working notes on what the new experiments actually returned. Written to be read
before deciding what goes in the workshop paper. Nothing here is estimated or
recalled; every number came out of a run.

Provenance, because it differs by finding and that matters:

- Corrections A and B read `compression_sweep_multiseed.csv`, 10 seeds, 80 cells.
- Findings 1, 2 and 6 read `degradation_control_logreg.csv`,
  `degradation_slopes.csv` and `degradation_matched.csv`.
- Finding 3 reads `compression_sweep_multiseed.csv`.
- Finding 4 reads `compression_paired.csv`, built from
  `compression_sweep_perseed.csv`, 10 seeds.
- Finding 5 reads `ece_robustness.csv`, 400 rows.

All are committed under `experiments/results/`.

One reproducibility check worth recording: the regenerated sweep matches the
previous one to **exactly zero difference** on all 18 shared metric columns
across all 52 previously existing cells. The new pipeline stage adds columns
without perturbing any existing number, so the paper's current tables remain
valid. Only `peak_ram_bytes` moves, by a few bytes in both directions on
measurements of 63 KB to 13 MB, which is `tracemalloc` allocator jitter: it is
sampled inside `profile_model`, before the new stage runs, so the new code
cannot be its cause.

Status key: **confirmed** means the full run finished and the number is stable
across seeds. **partial** means the run is still going and the number may move.

---

## Two corrections to the paper's core claims

Both came out of the regenerated sweep, both are measured, and both change what
the paper should say. Read these before the findings below.

### Correction A: the mechanism claim is right; the seed-averaged table is not

**This section previously argued the opposite and was wrong. It is corrected
here, and the retraction is the finding.**

The seed-averaged sweep row for Adult MLP at 90% sparsity reads accuracy 0.4980,
balanced accuracy 0.5049, positive rate 0.5107. Read as a summary of a typical
run, that says the model has become a coin flip, and the draft's "degradation
toward a near-constant predictor" would be wrong for the headline model.

It says no such thing. The per-seed rows:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| positive rate | 1.000 | 0.844 | 1.000 | 0.000 | 0.262 | 0.000 | 0.992 | 1.000 | 0.009 | 0.000 |
| accuracy | 0.239 | 0.335 | 0.239 | 0.761 | 0.646 | 0.761 | 0.244 | 0.239 | 0.755 | 0.761 |

The distribution is **bimodal, not central**. Eight of ten seeds sit at or within
one percent of a constant predictor. Three collapse to always-positive (positive
rate 1.000, accuracy 0.2393, which is exactly the positive-class prevalence) and
four to always-negative (positive rate 0.000, accuracy 0.7607, exactly the
majority-class rate). Only seeds 1 and 4 are genuinely intermediate.

The mean of a set of ones and zeros is 0.5. That is all the 0.5107 was.
Balanced accuracy could not catch it either, because a constant predictor scores
exactly 0.5 too, and so does a coin flip.

**So the draft's stated mechanism is correct and should stand.** The Adult MLP
does degrade toward a near-constant predictor. What varies is *which* constant,
and that varies by seed.

The real finding, which is more useful than the false one it replaces:

> At the collapsed end of the sparsity ladder, mean plus or minus standard
> deviation over seeds describes a model that **no seed actually produced**. The
> paper currently reports these cells that way and calls the intermediate
> sparsities "seed-noisy". The cause is not noise, it is bimodality, and the
> summary statistic is the wrong one.

Report those cells per seed, or report the modal outcome and the split, not the
mean. This also explains the large standard deviations the draft already noticed
without diagnosing.

### Correction B: the headline model is the wrong one to headline

The crux question was: at what sparsity does the illusion appear, and would
anyone deploy there? Answer, with the majority-class baseline as the yardstick:

| cell | acc @ 90% | trivial baseline | still deployable? | DP change |
|---|---|---|---|---|
| **adult / logreg / sex** | **0.8248** | 0.7607 | **yes** | 0.174 → 0.126 (−28%) |
| **adult / logreg / race** | **0.8248** | 0.7607 | **yes** | 0.171 → 0.101 (−41%) |
| adult / mlp / sex | 0.4980 | 0.7607 | no | 0.175 → 0.014 |
| adult / mlp / race | 0.4980 | 0.7607 | no | 0.150 → 0.019 |
| compas / logreg / * | 0.5450 | 0.5450 | no | → 0.000 |
| compas / mlp / * | 0.4967 | 0.5450 | no | → 0.004 / 0.070 |

The spectacular 12x number is produced by a model **less accurate than always
guessing the majority class**. A reviewer will notice, and the correct response
is to concede it rather than defend it: no practitioner ships that model, and
any accuracy check catches it.

Stated per seed so it does not rest on an average, which is the mistake
Correction A came from: across all ten seeds the Adult MLP at 90% sparsity
reaches a maximum accuracy of **0.7607**, which *equals* the majority-class
baseline and never exceeds it. Not "the mean is below the trivial baseline" but
"no individual run beats it". That version of the claim is immune to the
bimodality.

The Adult **logistic** cell is the one that matters. Losing 2.7 points of
accuracy, staying comfortably above the trivial baseline, and remaining a model
anyone would ship, it still shows demographic parity improving 28% (sex) to 41%
(race) for no fairness reason. On sex its per-group ECE disparity simultaneously
gets 2.7x worse (0.0057 → 0.0153). That is a real, deployable, silent illusion,
and it is the honest headline.

Recommendation: lead with Adult logreg as the deployable case and keep the MLP
as the limiting illustration of where the trend ends. Smaller headline number,
far more defensible paper.

## The short version

The review asked one hard question: is the parity collapse caused by
compression, or would any equally damaged model do the same thing? The control
experiment answers it, and the answer is better than expected. It also produced
a finding nobody asked for, which is arguably the more interesting one, plus the
two corrections above.

1. **The parity collapse is compression-specific.** Damaging a model by
   corrupting its training labels, or by starving it of training data, does not
   reproduce it. On Adult by race the effects run in *opposite directions*.
2. **Damage mechanisms leave distinguishable signatures, in logistic regression.**
   Pruning collapses parity while sparing calibration; label noise does the
   reverse. The pair (ΔDP, ΔECE) says *how* a model was broken. This was written
   up before the MLP arm finished and **does not generalise to the MLP**, where
   the ordering inverts. Scoped accordingly below.
2b. **Pruning destabilises the MLP and only the MLP.** 27x the run-to-run
   variance of either control, spanning the full [0, 1] range of positive rate.
   On logistic regression pruning is *more* stable than the controls.
3. **No point-prediction fairness metric survives the collapse.** DP, EO, and DI
   all read perfect simultaneously. "Report EO as well" is not a fix.
4. **Seed-pairing does not rescue the intermediate sparsities.** The draft's
   retreat to the endpoint contrast is statistically correct, not timid.
5. **The draft's mechanism claim is correct**, but the seed-averaged table for
   the collapsed cells is bimodal and its mean describes a model no seed
   produced (Correction A), and **the deployable case is Adult logreg, not the
   Adult MLP** (Correction B).

---

## Finding 1: the collapse is compression-specific (confirmed, logistic arm)

Three ways of damaging the same model, each traced across its own accuracy
range, from `experiments/results/degradation_control_logreg.csv` (360 rows,
5 seeds, 2 datasets, 2 attributes).

Change in each metric per point of accuracy given up, least-squares over each
mechanism's own curve. Negative ΔDP means parity *improves* as the model gets
worse, which is the illusion the paper is about.

| dataset / attr | mechanism | ΔDP | ΔECE | Δpos-rate |
|---|---|---|---|---|
| adult / race | **prune** | **−2.35** | +0.35 | **−2.53** |
| adult / race | label noise | **+1.66** | **+8.89** | −0.53 |
| adult / race | subsample | +0.18 | +0.88 | +0.73 |
| adult / sex | **prune** | **−2.07** | +0.72 | **−2.53** |
| adult / sex | label noise | −1.22 | **+9.89** | −0.53 |
| adult / sex | subsample | +0.45 | +0.83 | +0.73 |
| compas / race | **prune** | −4.22 | +0.05 | −2.41 |
| compas / race | label noise | −4.60 | +1.73 | −0.85 |
| compas / sex | **prune** | −1.00 | +0.03 | −2.41 |
| compas / sex | label noise | −0.77 | +1.89 | −0.85 |
| compas / sex | subsample | −1.42 | +0.33 | +1.81 |

**Adult by race is the clean refutation.** Pruning drives DP down (−2.35);
label noise drives it *up* (+1.66). Same model, same dataset, comparable
accuracy cost, opposite sign. Whatever pruning is doing to parity, generic
damage is not doing it.

**The controls cannot even reach the collapse regime.** Accuracy range spanned
by each mechanism at its most extreme setting:

| mechanism | extreme setting | accuracy range covered (adult) |
|---|---|---|
| prune | 95% sparsity | 0.052 |
| label noise | 40% of labels flipped | 0.026 |
| subsample | 2% of rows kept | 0.021 |

Subsampling Adult from 36,631 training rows to about 730 costs only two points
of accuracy and *raises* DP. Flipping 40% of the labels costs less than three
points. Only weight-space damage pushes the model far enough to collapse. This
is worth stating on its own: for these models, data-side damage degrades
gracefully while weight-side damage destroys the decision function.

A caveat to state in the paper rather than hide: because the controls degrade so
little, the honest accuracy-matched comparison window is narrow (Adult:
0.831–0.852). Inside that window pruning already shows the lower DP
(`degradation_matched.csv`), but the strongest part of the pruning curve lies
outside any range the controls reach.

## Finding 2: mechanisms have opposite signatures (confirmed, logistic arm)

Read the ΔDP and ΔECE columns above as a pair.

- **Pruning** is a parity-collapse signature: ΔDP −1.0 to −4.2, ΔECE +0.03 to
  +0.72.
- **Label noise** is a calibration-collapse signature: ΔECE +1.7 to +9.9, which
  is 5× to 30× pruning's rate, while ΔDP is inconsistent in sign.
- **Subsampling** is neither; it barely moves anything and the positive rate
  goes *up*.

This is a new claim and it is a constructive one. The paper's existing
contribution is a warning ("your audit can be fooled"). This adds a diagnostic:
the two-metric pair separates *how* a deployed model was degraded. That is a
better fit for an on-device venue than the warning alone, because on-device is
exactly where you get a degraded model and no access to the training pipeline
that produced it.

Figure: `paper/figures/degradation_control.pdf`, two panels, three curves each.

## Finding 3: no point-prediction metric survives (confirmed)

Measured, from `run_compression_sweep.py` on COMPAS logreg at 90% sparsity, with
the constant-predictor baselines now in the same table:

| model | DP | EO | DI | accuracy | balanced acc | pos. rate |
|---|---|---|---|---|---|---|
| logreg, uncompressed | 0.163 | 0.149 | 0.558 | 0.683 | 0.670 | 0.339 |
| logreg, pruned 90% | **0.000** | **0.000** | **1.000** | 0.545042 | 0.500 | 0.000 |
| majority-class baseline | **0.000** | **0.000** | **1.000** | 0.545042 | 0.500 | 0.000 |
| prevalence baseline | **0.000** | **0.000** | **1.000** | 0.545042 | 0.500 | 0.000 |

The pruned model is not *approaching* the constant predictor. It matches it on
every column to six decimal places. It is one.

This corrects a claim in the review discussion and in the earlier prep notes:
equalized odds is **not** harder to fool at the collapsed endpoint. A constant
predictor has a true-positive rate of 0 and a false-positive rate of 0 in every
group, so both gaps are 0 and EO is exactly 0.0. Adding EO to the report does
not protect the audit.

Only the calibration lens catches it. Over the same collapse, per-group ECE
disparity *rose* from 0.039 to 0.086.

## Finding 4: pairing extends the claim to 70%, but no further (confirmed, 10 seeds)

`analyze_paired.py` compares seed *s* uncompressed against seed *s* pruned and
takes statistics over the differences, which removes the between-seed variance
that swamps the unpaired bands. Demographic parity, 95% CI on the paired
difference:

| cell | prune:0.3 | prune:0.5 | prune:0.7 | prune:0.9 |
|---|---|---|---|---|
| adult / mlp / sex | ns | ns | **−0.091** (p=0.004) | **−0.160** (p<0.001) |
| adult / mlp / race | ns | ns | ns | **−0.131** (p<0.001) |
| compas / logreg / sex | ns | ns | **−0.017** (p=0.008) | **−0.139** (p<0.001) |
| compas / logreg / race | ns | ns | ns | **−0.625** (p<0.001) |
| compas / mlp / sex | ns | ns | **−0.063** (p<0.001) | **−0.123** (p<0.001) |
| compas / mlp / race | ns | ns | **−0.157** (p=0.044) | **−0.485** (p<0.001) |

Going from 5 to 10 seeds changed this conclusion, which is worth noting as a
reminder not to settle a power question on half the data. At 5 seeds only the
90% rung survived and the answer looked like a flat "pairing does not help." At
10 it **extends the defensible claim down to 70% in four of six cells**, while
30% and 50% stay firmly null everywhere.

So the paper can claim the 70–90% range rather than the endpoint alone, and can
say explicitly that 30% and 50% were tested and show nothing. That is a stronger
and more honest position than the current retreat.

127 of 304 comparisons across all metrics have a CI excluding zero
(`compression_paired.csv`).

**int8 is an exact no-op.** Paired difference 0.000000, p = 1.0, on COMPAS
logreg for both attributes. Dynamic int8 quantization changes no fairness
number at all on these models. Worth one sentence; it means the paper's "two
compression methods" is really one method plus a null.

## Finding 5: the calibration collapse survives adaptive binning (confirmed)

`run_ece_robustness.py`, 400 rows, one fitted ensemble per cell scored four ways
so the comparison is exactly paired. Ratio of mean per-group ECE at 90% sparsity
to uncompressed:

| cell | equal-width 10 | equal-width 15 | equal-mass 10 | equal-mass 15 |
|---|---|---|---|---|
| adult / mlp / sex | 10.93× | 10.54× | 10.92× | 10.79× |
| adult / mlp / race | 6.04× | 5.64× | 6.13× | 5.78× |
| compas / logreg / sex | 1.10× | 1.04× | 1.03× | 0.86× |
| compas / logreg / race | 1.07× | 0.95× | 0.77× | 0.72× |

**The headline holds.** The ~10× calibration blow-up on the Adult MLP is stable
to within 4% across all four estimators, so it is not a binning artifact. The
paper can state this as tested.

Two side results worth keeping:

- **Placement matters more than count, as predicted.** Changing 10→15 bins moves
  ECE by at most 4%. Changing equal-width→equal-mass moves the *absolute* COMPAS
  numbers by up to 39% (race, uncompressed: 0.133 vs 0.184). The originally
  proposed 10-vs-15 check would have found nothing. Worth saying so gently,
  since the instinct to check the estimator was the right one.
- **`n_distinct_confidence` independently confirms Correction A.** Distinct
  predicted-confidence values, uncompressed → 90%: COMPAS logreg 570 → 31 →
  **1**; Adult MLP 9541 → **2752**. COMPAS logreg literally becomes a single
  repeated confidence, which is why all four binnings return the identical
  number there. The Adult MLP keeps thousands of distinct values. Same DP → 0,
  demonstrably different mechanisms.

## Finding 6: the collapse does not reduce to a single law (negative)

Tested and did not hold. Regressing the per-seed change in DP on the change in
positive prediction rate, pooled over all 360 logistic rows: slope 0.87,
R² = 0.30. Under pruning alone R² = 0.44. Against accuracy change instead,
pooled R² = 0.41.

The hypothesis was that the apparent fairness gain is a deterministic readout of
collapse, which would have made a tidy claim. It is a real correlation but not a
tight one, and the per-mechanism slopes disagree in sign (prune +1.12, label
noise −0.42, subsample +0.04). Reporting it as a law would be overclaiming. The
signature finding above is what the data supports.

---

## Still running

- **Degradation control, MLP arm** (Adult, 3 seeds, all three mechanisms). The
  logistic arm is complete and carries Findings 1 and 2. Note that Correction B
  makes this less load-bearing than it first appeared: the logistic cell is the
  deployable one and is therefore the one the control most needs to cover, and
  it already does. The MLP arm is confirmation, not a dependency.

## What this changes for the paper

- **Lead with Adult logreg, not the Adult MLP.** The 12× number comes from a
  model less accurate than always guessing the majority class. The logistic
  cell loses 2.7 accuracy points, stays deployable, and still shows parity
  improving 28–41%. Smaller number, real claim.
- **Restate the mechanism as output-independence, not near-constancy.** Both
  routes are now measured, in the same table, with `n_distinct_confidence` as
  the discriminator.
- **The causal framing survives.** A title asserting compression-induced effects
  is defensible on the logistic arm, where the control is complete.
- **The strongest framing is no longer "compression fools your audit."** It is
  "damage mechanisms are distinguishable from audit output alone, and
  compression is the one that fools parity." That is a contribution rather than
  a warning, and it fits an on-device venue better, because on-device is exactly
  where you inherit a degraded model with no access to the pipeline that made
  it.
- **Drop the "report EO as well" mitigation.** It is wrong; EO reads 0.0 on a
  constant predictor.
- **Claim the 70–90% range**, and say that 30% and 50% were tested and show
  nothing.
- **State the ECE result as tested**, not assumed: the ~10× blow-up is stable
  across four binning estimators.
- **int8 is a null result** and should be reported as one rather than as a
  second compression method.

## What is not done

Tier 3 remains untouched and should be named as future work rather than quietly
omitted: the ACSIncome compression arm, Brier and NLL, reliability diagrams, a
third compression method, small-group instability, and per-group recall
breakdowns. The paper text itself has not been edited; every change above is in
code, data, and figures.
