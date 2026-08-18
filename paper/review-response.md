# Response to review: what was implemented, analyzed, and found

Complete record of the work done on the TinyAudit workshop draft in response to
the review. Three parts: every suggestion and what happened to it, every
analysis run and how, and the findings that came out.

All numbers trace to a CSV under `experiments/results/`. Nothing here is
estimated or recalled from an earlier run.

**Verify any of it with:**

```bash
pip install -e ".[dev]"
pytest --cov=src/tinyaudit          # 275 tests, ~81s
python experiments/run_multiseed.py --only compression_sweep
python experiments/run_degradation_control.py
python experiments/run_ece_robustness.py
python experiments/analyze_paired.py
python experiments/analyze_degradation.py
python experiments/make_paper_figures.py
```

---

# Part 1: Every suggestion, and what happened to it

## Implemented in full

### 1. Collapse diagnostics

**Asked for:** positive prediction rate, mean and standard deviation of
predicted probability, number of unique predicted probability values.

**Done, except the last one.** A new `PerformanceBlock` runs as an unconditional
pipeline stage and records accuracy, balanced accuracy, positive prediction
rate, and the mean and standard deviation of the predicted probabilities. It is
computed on the *audited* model, so under compression it describes the
compressed model, which is the only version that matters.

This was the central gap. The draft asserted a mechanism and never measured it;
`compression_sweep_multiseed.csv` carried no accuracy column and no
output-distribution statistic at all. It now carries six new metric pairs.

The unique-values count was deliberately dropped and replaced. On continuous
inputs an MLP produces close to *n* distinct probabilities even when it is
functionally useless, so the count would not have shown what it was meant to
show. The equivalent measurement that does work is `n_distinct_confidence` in
`ece_robustness.csv`, computed on rounded confidences, and it turned out to be
one of the more informative columns in the whole set (see Finding A).

**Where:** `src/tinyaudit/pipeline.py`, `src/tinyaudit/card/schema.py`,
`src/tinyaudit/card/templates/audit_card.html.j2`, `tests/test_performance.py`.

### 2. Constant-predictor baselines

**Asked for:** majority-class and prevalence-probability baselines.

**Done.** A new `ConstantModel` conforms to the existing `AuditedModel`
protocol, so both baselines go through exactly the same `audit()` code path as
every other cell rather than being computed separately and pasted in. Two rows
per dataset now sit in the sweep table beside the real models.

This turned out to carry more weight than expected. It is what let us state that
the 90% pruned COMPAS model does not merely *approach* a constant predictor but
matches it to six decimal places on every column.

**Where:** `src/tinyaudit/models/constant.py`, `tests/models/test_constant.py`
(34 tests, including a Hypothesis property test over arbitrary group counts).

### 3. Metric sensitivity: does parity improve while equalized odds does not

**Done, and it produced a correction rather than a confirmation.** Equalized
odds and disparate impact were already computed for every cell of the sweep and
simply never plotted, so this was close to free. A new
`metric_sensitivity.pdf` plots all three across the sparsity ladder.

The premise did not survive contact with the constant-predictor baseline. See
Finding 3: equalized odds reads exactly 0.0 on a collapsed model, so it is not
the safer metric the framing assumed.

**Where:** `experiments/make_paper_figures.py`.

### 4. Calibration quality versus calibration disparity

**Asked for:** separate "did calibration get worse on average" from "did one
group get much worse".

**Done.** `ece_disparity` (max minus min per-group ECE) is now reported beside
`mean_ece_per_group`, nan-safe for single-group and single-class cases, with a
lone group scoring 0.0 rather than NaN because one group has no gap.

This was a real conceptual gap. A single averaged ECE cannot distinguish
"everyone got less trustworthy" from "one group got much less trustworthy", and
only the second is a fairness problem.

**Where:** `src/tinyaudit/pipeline.py`.

### 5. Degradation control

**Asked for:** break a model a different way (label noise), tune it to the same
accuracy as the pruned model, and check whether fairness does the same thing.

**Done, and deliberately made stronger than proposed.** Rather than matching a
single accuracy point, the experiment traces whole curves through accuracy space
under three different damage mechanisms:

- `prune`, magnitude pruning at increasing sparsity, the mechanism under test
- `label_noise`, a fraction of training labels flipped before fitting, test set
  untouched, which is the control as proposed
- `subsample`, fitting on a shrinking fraction of training rows, a second
  non-compression route so a single quirk of label noise cannot carry the
  conclusion

Curve comparison is better than point matching here for a reason that only
became visible once it was run: the controls cannot reach the accuracies pruning
reaches, so a single matched point would have been picked from a narrow window
and would have understated the difference. The curves show the whole picture
including where they stop.

Also added: `analyze_degradation.py`, which puts every mechanism on a common
accuracy axis by interpolation, restricted to the range all mechanisms actually
cover, and fits a slope per mechanism.

**Where:** `experiments/run_degradation_control.py`,
`experiments/analyze_degradation.py`, `paper/figures/degradation_control.pdf`.

### 6. More statistical analysis

**Asked for:** unspecified, "more statistical analysis".

**Interpreted as, and implemented as, seed-paired comparison.** The draft reports
mean plus or minus standard deviation over 10 seeds, sees the bands overlap at
intermediate sparsities, and retreats to the endpoint contrast. That retreat was
testing the wrong thing. Between-seed variance sits in both bands and swamps the
effect. Pairing removes it: seed 0 uncompressed against seed 0 pruned, and so
on, with statistics over the differences.

Reports mean paired difference, 95% confidence interval, paired t-test, and
Cohen's dz for eight metrics across every compression level.

This required a change to `run_multiseed.py`, which previously wrote per-seed
CSVs to a temporary directory and deleted them. It now keeps
`compression_sweep_perseed.csv` alongside the aggregate, because paired analysis
is impossible without it.

**Where:** `experiments/analyze_paired.py`, `experiments/run_multiseed.py`.

### 7. Robustness of ECE

**Asked for:** 10 versus 15 bins, with adaptive binning listed as optional.

**Done, with the priority inverted, and the inversion was justified by the
result.** Both axes were tested, but the argument made beforehand was that bin
*count* is the weak axis (both are equal-width, so they agree largely by
construction) and bin *placement* is the axis that can actually overturn the
result. That is what the data showed: changing 10 to 15 bins moves ECE by at
most 4%, while changing equal-width to equal-mass moves absolute COMPAS numbers
by up to 39%.

`ece_per_group` gained `n_bins` and `binning` parameters with defaults that
reproduce the previous behavior exactly, verified by a test. The experiment fits
one ensemble per cell and scores it four ways, so the four configurations are
exactly paired rather than being four separate runs.

**Where:** `src/tinyaudit/uncertainty/metrics.py`,
`experiments/run_ece_robustness.py`, `tests/uncertainty/test_ece_binning.py`.

## Already present before the review

### 8. A second compression method

int8 dynamic quantization was already in the sweep across both datasets and all
three models. Worth saying so, since it shrinks the ask.

It is worth saying something else too, which the new paired analysis established:
**int8 changes no fairness number at all.** Paired difference exactly 0.000000,
p = 1.0. It should be reported as a null result rather than as a second method.

### 9. Latency and memory measurement

The review said the work "lacks latency, memory measurement". The paper does
report peak RAM and wall-clock in Table 1, profiled with `tracemalloc`, and
states plainly that it is x86 and not a microcontroller. What is genuinely
missing is embedded hardware validation and energy measurement. Worth correcting
once, gently, and not litigating.

## Deferred, and why

### 10. ACSIncome compression arm

Not run. The ~49k-row test set makes a 10-seed sweep a multi-hour job for a
supporting result, and the workshop page limit cannot accommodate a third
dataset in the compression table. Belongs in the full version.

### 11. Brier score, NLL, reliability diagrams

Not run. These strengthen the calibration section but the calibration claim is
already carried by per-group ECE plus the new disparity split plus the binning
robustness check. Diminishing returns against a 5-page limit.

### 12. Small-group instability, per-group recall breakdowns

Not run. Same reason.

## Framing suggestions, not implemented because they are decisions rather than tasks

- **Narrowing the focus to the compression thread.** Agreed, and the new results
  support it, but which sections get cut is a call for the authors.
- **The three proposed titles.** The causal framing they assert is now
  defensible on the logistic arm, which it was not before the control ran. Still
  a decision.
- **The three paper identities.** Note a genuine tension: the recommended
  "empirical ML paper" identity is a worse fit for an on-device workshop, not a
  better one. The signature finding below actually argues for keeping the
  on-device framing, since a diagnostic that works from audit output alone is
  most useful exactly where you inherit a degraded model and cannot see the
  pipeline that produced it.

---

# Part 2: What was analyzed

Six analyses. Each names the question, the method, and the output file.

### A. Regenerated compression sweep, 10 seeds

**Question:** what do the six existing metrics look like beside accuracy and the
output-distribution statistics that were missing?

**Method:** full re-run of the sweep over 10 seeds, 2 datasets, 3 models, 2
sensitive attributes, 6 compression levels, plus 2 constant-predictor baselines
per dataset. 80 aggregated cells, up from 52.

**Also fixed a data-integrity bug found along the way.** Twenty of the seventy-two
cells were silently missing. Decision trees have no weight array, so neither
`magnitude_prune` nor `quantize_int8` applies to them, and a bare
`except Exception: continue` made every compressed tree cell vanish with nothing
in the CSV to say why. The paper's "three models" was really two under
compression. Failed cells are now written out with a `skip_reason` column, so
the gap lives in the data instead of in a log that scrolled past.

**Output:** `compression_sweep_multiseed.csv` (36 columns, up from 23),
`compression_sweep_perseed.csv`.

### B. Reproducibility check, old against new

**Question:** did adding a pipeline stage perturb any existing number?

**Method:** merge the previous aggregate against the new one on the four key
columns and take the maximum absolute difference per shared metric column.

**Result:** exactly 0.000e+00 on all 18 shared metric columns across all 52
previously existing cells. The only movement is `peak_ram_bytes`, by a few bytes
in both directions on measurements ranging from 63 KB to 13 MB. That is
`tracemalloc` allocator jitter, not a systematic shift: peak RAM is sampled
inside `profile_model`, which runs *before* the new stage, so the new code
cannot be its cause.

**Consequence:** the paper's existing tables remain valid as printed.

### C. Degradation control, three mechanisms

**Question:** is the parity collapse caused by compression, or would any equally
damaged model do it?

**Method:** 360 rows on the logistic arm (2 datasets, 2 attributes, 3
mechanisms, 6 levels each, 5 seeds). Explainability stage skipped throughout,
since occlusion over Adult's 101 features dominates runtime and contributes
nothing here. Training is shared across sensitive attributes, which halves the
expensive part.

**Output:** `degradation_control_logreg.csv` (360 rows),
`degradation_control_mlp.csv` (180 rows, Adult, 5 seeds),
`degradation_slopes.csv`, `degradation_matched.csv`.

**Both arms are now complete, and they disagree**, which is the reason the MLP
arm was worth running rather than assuming. The signature separation reported in
Finding 2 holds on logistic regression and inverts on the MLP. Because
Correction A established that the collapsed MLP cells are bimodal across seeds,
the MLP arm is analysed by per-seed median slope with the interquartile range,
not by fitting a slope to the seed-averaged curve, which would repeat exactly
the mistake Correction A documents.

### D. Seed-paired statistics

**Question:** does pairing rescue the intermediate sparsities the draft retreats
from?

**Method:** 304 paired comparisons, 8 metrics across every compression level and
cell, 10 pairs each.

**Output:** `compression_paired.csv`.

**Methodological note worth keeping:** this analysis was run first at 5 seeds
and gave a different answer than at 10. At 5 seeds only the 90% rung survived
and the conclusion looked like a flat "pairing does not help". At 10 seeds it
extends the claim to 70% in four of six cells. A power question settled on half
the data would have been settled wrongly.

### E. ECE binning robustness

**Question:** is the calibration-collapse finding an artifact of where the
confidence bins are drawn?

**Method:** 400 rows. One fitted ensemble per cell, scored under four
configurations (equal-width and equal-mass, at 10 and 15 bins), so the
comparison is exactly paired.

**Output:** `ece_robustness.csv`.

### F. Collapse-explains-parity regression

**Question:** is the apparent fairness gain a deterministic readout of collapse?

**Method:** regress the per-cell change in demographic parity on the change in
positive prediction rate, pooled and per mechanism, over all 360 logistic rows.

**Result:** no. Reported as a negative in Finding 6.

---

# Part 3: Novel findings

Two of these are corrections to claims the draft currently makes. Read those
first, because they change what the paper should say.

## Correction A: the mechanism claim is right; the seed-averaged table is not

An earlier draft of this document argued that the paper's stated mechanism was
wrong for the Adult MLP. That argument was itself wrong, it is retracted here,
and the retraction is the more useful finding.

The seed-averaged sweep row for Adult MLP at 90% sparsity reads accuracy 0.4980,
balanced accuracy 0.5049, positive rate 0.5107. Taken as a description of a
typical run, that says the model has become a coin flip rather than a constant,
which would contradict the draft. The per-seed rows say otherwise:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| positive rate | 1.000 | 0.844 | 1.000 | 0.000 | 0.262 | 0.000 | 0.992 | 1.000 | 0.009 | 0.000 |
| accuracy | 0.239 | 0.335 | 0.239 | 0.761 | 0.646 | 0.761 | 0.244 | 0.239 | 0.755 | 0.761 |

The distribution is **bimodal, not central**. Eight of ten seeds sit at or within
one percent of a constant predictor: three collapse to always-positive (positive
rate 1.000, accuracy 0.2393, exactly the positive-class prevalence) and four to
always-negative (positive rate 0.000, accuracy 0.7607, exactly the majority-class
rate). Only seeds 1 and 4 are genuinely intermediate.

The mean of a set of ones and zeros is 0.5. That is all the 0.5107 was. Balanced
accuracy could not disambiguate either, because a constant predictor scores
exactly 0.5 and so does a coin flip.

**The draft's stated mechanism is correct and should stand.** The Adult MLP does
degrade toward a near-constant predictor. What varies is *which* constant, and
that varies by seed. COMPAS logistic regression collapses the same way, and its
distinct-confidence count falling 570 to 31 to **1** is independent confirmation
of the constant-predictor reading.

The finding that replaces the false one:

> At the collapsed end of the sparsity ladder, mean plus or minus standard
> deviation over seeds describes a model that **no seed actually produced**. The
> paper reports these cells that way and calls the intermediate sparsities
> "seed-noisy". The cause is not noise, it is bimodality, and the summary
> statistic is the wrong one.

Report those cells per seed, or report the modal outcome and the split. This
also explains the large standard deviations the draft noticed without
diagnosing.

**Method note worth carrying forward.** This error was caught only by looking at
per-seed rows, which existed solely because `run_multiseed.py` was changed to
retain `compression_sweep_perseed.csv` for the paired analysis. Without that
change the aggregate would have been the only available view and the false
claim would have survived.

## Correction B: the headline model is the wrong one to headline

The crux question the review circled but did not name: at what sparsity does the
illusion appear, and would anyone deploy there? Measured against the
majority-class baseline.

| cell | accuracy at 90% | trivial baseline | deployable | parity change |
|---|---|---|---|---|
| **adult / logreg / sex** | **0.8248** | 0.7607 | **yes** | 0.174 to 0.126, down 28% |
| **adult / logreg / race** | **0.8248** | 0.7607 | **yes** | 0.171 to 0.101, down 41% |
| adult / mlp / sex | 0.4980 | 0.7607 | no | 0.175 to 0.014 |
| adult / mlp / race | 0.4980 | 0.7607 | no | 0.150 to 0.019 |
| compas / logreg, both attrs | 0.5450 | 0.5450 | no | to 0.000 |
| compas / mlp, both attrs | 0.4967 | 0.5450 | no | to 0.004 and 0.070 |

The spectacular 12x number is produced by a model **less accurate than always
guessing the majority class**. A reviewer will notice, and the right response is
to concede rather than defend: nobody ships that model, and any accuracy check
catches it.

Stated per seed, so that it does not rest on an average and cannot fall to the
mistake in Correction A: across all ten seeds the Adult MLP at 90% sparsity
reaches a maximum accuracy of **0.7607**, which equals the majority-class
baseline and never exceeds it. Not "the mean is below the trivial baseline" but
"no individual run beats it".

The Adult **logistic** cell is the one that matters. It gives up 2.7 points of
accuracy, stays comfortably above the trivial baseline, remains a model anyone
would deploy, and still shows demographic parity improving 28% to 41% for no
fairness reason. On sex its per-group ECE disparity simultaneously gets 2.7x
worse, from 0.0057 to 0.0153.

**Recommendation:** lead with Adult logreg as the deployable case and keep the
MLP as the limiting illustration of where the trend ends. Smaller headline
number, far more defensible paper.

## Finding 1: the parity collapse is compression-specific

Change in each metric per point of accuracy given up, least-squares over each
mechanism's own curve. Negative parity change means parity *improves* as the
model gets worse, which is the illusion.

| dataset / attr | mechanism | parity | ECE | positive rate |
|---|---|---|---|---|
| adult / race | **prune** | **-2.35** | +0.35 | **-2.53** |
| adult / race | label noise | **+1.66** | **+8.89** | -0.53 |
| adult / race | subsample | +0.18 | +0.88 | +0.73 |
| adult / sex | **prune** | **-2.07** | +0.72 | **-2.53** |
| adult / sex | label noise | -1.22 | **+9.89** | -0.53 |
| adult / sex | subsample | +0.45 | +0.83 | +0.73 |
| compas / race | **prune** | -4.22 | +0.05 | -2.41 |
| compas / race | label noise | -4.60 | +1.73 | -0.85 |
| compas / sex | **prune** | -1.00 | +0.03 | -2.41 |
| compas / sex | label noise | -0.77 | +1.89 | -0.85 |
| compas / sex | subsample | -1.42 | +0.33 | +1.81 |

**Adult by race is the clean refutation.** Pruning drives parity down at -2.35;
label noise drives it *up* at +1.66. Same model, same dataset, comparable
accuracy cost, opposite sign. Whatever pruning does to parity, generic damage
does not do it.

**Scope, added after the MLP arm completed.** This separation is clean on
logistic regression for both datasets. On the Adult MLP the per-mechanism slopes
overlap once seed spread is accounted for, so the opposite-sign result should be
stated as a logistic-regression finding rather than a general one. See Finding 2
and 2b.

**A second, structural point:** the controls cannot even reach the collapse
regime. Accuracy range spanned at each mechanism's most extreme setting on
Adult: pruning at 95% sparsity covers 0.052, label noise at 40% of labels
flipped covers 0.026, subsampling to 2% of rows covers 0.021. Subsampling Adult
from 36,631 training rows to about 730 costs two points of accuracy and *raises*
parity. For these models, data-side damage degrades gracefully while weight-side
damage destroys the decision function.

**Caveat to state rather than hide:** because the controls degrade so little, the
honest accuracy-matched window is narrow, 0.831 to 0.852 on Adult. Inside it
pruning already shows the lower parity, but the strongest part of the pruning
curve lies outside any range the controls reach.

## Finding 2: damage mechanisms leave opposite signatures, in logistic regression only

This is the finding nobody asked for and it is the most interesting one. It is
also **narrower than first reported**: it was written up from the logistic arm
before the MLP arm existed, and the MLP arm does not reproduce it. Scope stated
plainly below.

**On logistic regression, both datasets.** Read the parity and ECE columns above
as a pair.

- **Pruning** is a parity-collapse signature: parity -1.0 to -4.2, ECE +0.03 to
  +0.72.
- **Label noise** is a calibration-collapse signature: ECE +1.7 to +9.9, five to
  thirty times pruning's rate, with parity inconsistent in sign.
- **Subsampling** is neither, and moves the positive rate *up*.

**On the Adult MLP, it does not hold.** Per-seed median slopes, which is the
robust statistic given the bimodality established in Correction A, with the
interquartile range across seeds in brackets:

| mechanism | median parity slope (sex) | median ECE slope (sex) |
|---|---|---|
| prune | -0.317 [IQR 0.295] | **0.617** |
| label noise | -0.503 [IQR 0.204] | **0.203** |
| subsample | -0.017 [IQR 0.129] | **2.536** |

The ordering **inverts**: on the MLP, pruning damages calibration three times
faster than label noise does, which is the opposite of the logistic result. The
parity slopes overlap heavily once the interquartile ranges are taken into
account. There is no clean separation to read.

**The honest claim is therefore scoped:** the (parity, ECE) signature separates
damage mechanisms for logistic regression on both datasets, and does not
generalise to the MLP. It is still a diagnostic worth reporting and still argues
for the on-device framing, but as a linear-model result rather than a universal
one. The paper should say which architecture it holds for.

## Finding 2b: pruning destabilises the MLP, and only the MLP

Replacing the generality that Finding 2 lost, the MLP arm produced a cleaner
discriminator of its own. Standard deviation of the positive prediction rate
across seeds, at each mechanism's most extreme setting, on Adult:

| model | prune | label noise | subsample | ratio |
|---|---|---|---|---|
| logreg | 0.008, range [0.11, 0.13] | 0.030 | 0.014 | **0.3x** |
| mlp | **0.461, range [0.00, 1.00]** | 0.017 | 0.016 | **26.9x** |

On the MLP, pruning produces roughly **27 times** the run-to-run variance of
either control, and the range spans the entire interval: some seeds collapse to
always-positive, others to always-negative. On logistic regression pruning is
*more* stable than the controls, at 0.3x.

So the discriminator is architectural rather than universal, in the opposite
direction to Finding 2:

- **Logistic regression:** pruning walks the model to a constant predictor
  smoothly and reproducibly. The signature separation works.
- **MLP:** pruning jumps the model to a constant predictor unpredictably, and
  which constant it lands on is a coin flip across seeds. The signature
  separation fails, but the instability itself is unmistakable.

Both architectures end at a near-constant predictor with parity at zero, which is
the robust common claim. How they get there differs, and neither the slope
signature nor the instability signature generalises across both.

## Finding 3: no point-prediction fairness metric survives the collapse

| COMPAS logreg | parity | equalized odds | disparate impact | accuracy | balanced acc |
|---|---|---|---|---|---|
| uncompressed | 0.163 | 0.149 | 0.558 | 0.683 | 0.670 |
| pruned 90% | **0.000** | **0.000** | **1.000** | 0.545042 | 0.500 |
| majority baseline | **0.000** | **0.000** | **1.000** | 0.545042 | 0.500 |
| prevalence baseline | **0.000** | **0.000** | **1.000** | 0.545042 | 0.500 |

The pruned model is not approaching the constant predictor. It matches it on
every column to six decimal places.

**This overturns a mitigation that was on the table.** Equalized odds is *not*
harder to fool at the collapsed endpoint. A constant predictor has a
true-positive rate of 0 and a false-positive rate of 0 in every group, so both
gaps are 0 and equalized odds reads exactly 0.0. Adding it to the report does
not protect the audit.

Only the calibration lens catches it. Over the same collapse, per-group ECE
disparity *rose*, from 0.039 to 0.086.

## Finding 4: pairing extends the claim to 70%, but no further

95% confidence interval on the paired difference in demographic parity, 10 seeds.

| cell | 30% | 50% | 70% | 90% |
|---|---|---|---|---|
| adult / mlp / sex | ns | ns | **-0.091**, p=.004 | **-0.160**, p<.001 |
| adult / mlp / race | ns | ns | ns | **-0.131**, p<.001 |
| compas / logreg / sex | ns | ns | **-0.017**, p=.008 | **-0.139**, p<.001 |
| compas / logreg / race | ns | ns | ns | **-0.625**, p<.001 |
| compas / mlp / sex | ns | ns | **-0.063**, p<.001 | **-0.123**, p<.001 |
| compas / mlp / race | ns | ns | **-0.157**, p=.044 | **-0.485**, p<.001 |

The claim extends down to **70% in four of six cells**, while 30% and 50% stay
firmly null everywhere. The paper can claim the 70 to 90 range and state plainly
that 30% and 50% were tested and show nothing, which is stronger and more honest
than the current retreat to the endpoint alone.

Across all metrics, 127 of 304 comparisons have a confidence interval excluding
zero.

## Finding 5: the calibration blow-up survives adaptive binning

Ratio of mean per-group ECE at 90% sparsity to uncompressed.

| cell | equal-width 10 | equal-width 15 | equal-mass 10 | equal-mass 15 |
|---|---|---|---|---|
| adult / mlp / sex | 10.93x | 10.54x | 10.92x | 10.79x |
| adult / mlp / race | 6.04x | 5.64x | 6.13x | 5.78x |
| compas / logreg / sex | 1.10x | 1.04x | 1.03x | 0.86x |
| compas / logreg / race | 1.07x | 0.95x | 0.77x | 0.72x |

**The headline holds.** The roughly 10x blow-up on the Adult MLP is stable to
within 4% across all four estimators, so it is not a binning artifact and can be
stated as tested.

Two side results worth keeping. **Placement matters more than count:** changing
10 to 15 bins moves ECE by at most 4%, while equal-width to equal-mass moves
absolute COMPAS numbers by up to 39%, for example 0.133 against 0.184 on race
uncompressed. And COMPAS shows **no calibration collapse at all**, ratios near
1.0, which is consistent rather than contradictory: it becomes a constant
predictor whose single confidence value happens to be well calibrated to the
base rate.

## Finding 6: the collapse does not reduce to a single law

Tested and did not hold, reported as a negative.

Regressing the change in demographic parity on the change in positive prediction
rate, pooled over all 360 logistic rows: slope 0.87, R-squared 0.30. Under
pruning alone, R-squared 0.44. Against accuracy change instead, pooled R-squared
0.41.

The hypothesis was that the apparent fairness gain is a deterministic readout of
collapse, which would have made a tidy claim. It is a real correlation but not a
tight one, and the per-mechanism slopes disagree in sign: pruning +1.12, label
noise -0.42, subsampling +0.04. Reporting it as a law would be overclaiming. The
signature finding is what the data supports.

## Finding 7: int8 quantization is an exact no-op

Paired difference exactly 0.000000, p = 1.0, on COMPAS logreg for both
attributes. Dynamic int8 quantization changes no fairness number at all on these
models. The paper's "two compression methods" is one method plus a null, and
saying so is more useful than leaving it implicit.

---

# Part 4: What this changes for the paper

- **Lead with Adult logreg, not the Adult MLP.** The deployable illusion is the
  claim worth making.
- **Keep the near-constant-predictor mechanism.** It is correct, and it is now
  measured rather than asserted, with distinct-confidence counts and positive
  rates behind it.
- **Stop summarising the collapsed cells with a mean.** They are bimodal. Report
  per seed, or report the modal outcome and the split, and say that the large
  standard deviations at high sparsity are bimodality rather than noise.
- **The causal framing survives**, so a title asserting compression-induced
  effects is defensible on the logistic arm where the control is complete.
- **The strongest available framing is no longer "compression fools your
  audit".** It is "damage mechanisms are distinguishable from audit output
  alone, and compression is the one that fools parity". State it as a
  logistic-regression result, because it inverts on the MLP.
- **Report the architecture split.** Both families end at a near-constant
  predictor with parity at zero, and that is the robust common claim. How they
  get there differs: logistic regression walks there smoothly and reproducibly,
  the MLP jumps there with 27 times the run-to-run variance and lands on either
  constant depending on the seed. Neither discriminator generalises across both.
- **Drop the "report equalized odds as well" mitigation.** It is wrong.
- **Claim the 70 to 90 range**, and say 30% and 50% were tested and show nothing.
- **State the ECE result as tested**, not assumed.
- **Report int8 as a null result** rather than as a second compression method.
- **Report the decision-tree gap honestly.** Trees cannot be compressed by either
  method in this codebase, so the compression results cover two model families,
  not three.

## Open decisions, which are not ours to make

- Venue: ODI against a venue that fits an empirical-ML identity better.
- Whether Result 1, the decoupling, stays a headline, drops to supporting, or is
  cut.
- Title, now that the causal framing has evidence behind it.
- Author list and order.
- Byline details flagged earlier and still unconfirmed.

---

# Part 5: Inventory

## New source

| File | What it is |
|---|---|
| `src/tinyaudit/models/constant.py` | `ConstantModel`, majority and prevalence baselines |
| `experiments/run_degradation_control.py` | Three-mechanism degradation control |
| `experiments/run_ece_robustness.py` | Four-way ECE binning comparison |
| `experiments/analyze_degradation.py` | Accuracy-matched comparison and per-mechanism slopes |
| `experiments/analyze_paired.py` | Seed-paired statistics with confidence intervals |

## Modified source

| File | Change |
|---|---|
| `src/tinyaudit/pipeline.py` | Performance stage, `ece_disparity`, single shared prediction pass |
| `src/tinyaudit/card/schema.py` | `PerformanceBlock` |
| `src/tinyaudit/card/templates/audit_card.html.j2` | Performance section on the card |
| `src/tinyaudit/uncertainty/metrics.py` | `n_bins` and `binning` on `ece_per_group` |
| `experiments/run_compression_sweep.py` | New columns, baseline rows, `skip_reason` |
| `experiments/run_multiseed.py` | `--only`, per-seed CSV retention, skip-reason passthrough |
| `experiments/make_paper_figures.py` | Figure 2 redrawn on accuracy, two new figures |

## New tests, 58 total

| File | Count | Covers |
|---|---|---|
| `tests/models/test_constant.py` | 34 | Protocol conformance, exact parity, Hypothesis property test |
| `tests/test_performance.py` | 14 | Performance block, bands, `ece_disparity`, the collapse signature |
| `tests/uncertainty/test_ece_binning.py` | 10 | Default unchanged, equal-mass path, degenerate single-bin case |

Suite stands at 275 tests, 81 seconds, one pre-existing failure
(`test_render_pdf_basic`, WeasyPrint cannot load `libgobject-2.0-0` on Windows;
it fails identically with all of this work stashed).

## New and regenerated data

`compression_sweep_multiseed.csv` (regenerated, 36 columns, 80 cells),
`compression_sweep_perseed.csv`, `compression_paired.csv`,
`degradation_control_logreg.csv`, `degradation_control_mlp.csv`,
`degradation_slopes.csv`, `degradation_matched.csv`, `ece_robustness.csv`.

## Figures

`compression.pdf` redrawn with accuracy on the x-axis and the constant-predictor
floor marked, `metric_sensitivity.pdf` and `degradation_control.pdf` new,
`decoupling.pdf` unchanged.

## Not done

Tier 3 stands: the ACSIncome compression arm, Brier and NLL, reliability
diagrams, a third compression method, small-group instability, per-group recall
breakdowns.

The Adult MLP degradation arm is re-running. Its first run was discarded: it was
launched against the wrong Python interpreter, in which the uncertainty stage
failed for every one of the 108 cells, leaving all calibration columns empty.
The failure was silent in the output because `audit()` records a failed
uncertainty stage as skipped and carries on, which is correct behaviour for a
library and unhelpful for a batch run. Two things follow, both worth keeping:

- Experiment scripts should be invoked through the project interpreter
  explicitly rather than a bare `python`, which is what
  `pip install -e ".[dev]"` plus an activated virtualenv is supposed to
  guarantee and did not here.
- A batch runner should treat an all-null metric column as a failure rather than
  a result. Nothing in the pipeline flagged 108 consecutive skipped uncertainty
  stages.

The logistic arm, which carries Findings 1 and 2, was produced with the correct
interpreter and is unaffected. Correction B established that the logistic cell is
the one the control most needs to cover, so the MLP arm remains confirmation
rather than a dependency.

**The paper source has not been edited.** Every change above is in code, data,
tests, and figures.
