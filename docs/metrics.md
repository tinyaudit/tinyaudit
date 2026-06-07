# Metrics Reference

TinyAudit reports six metrics across two families, supported by three
uncertainty estimators and three explainers. Every metric is implemented from
scratch and tested against an oracle on a fixed seed.

## Point-prediction fairness (3 metrics)

| Metric | Symbol | Range | Notes |
| --- | --- | --- | --- |
| Demographic-parity difference | DP diff | `[0, 1]` | Difference in positive-prediction rate across groups |
| Equalized-odds difference | EO diff | `[0, 1]` | Difference in TPR and FPR across groups |
| Disparate-impact ratio | DI ratio | `>= 0` | Ratio of positive rates; the direction convention is pinned |

The disparate-impact direction convention — which group is the numerator, which
is the denominator, and what counts as the favored outcome — is the single
easiest place in the system to silently ship a bug. It is fixed in
the metric's docstring and in a dedicated test case, and it must not change
without updating both.

Property-test invariants enforced via `hypothesis`:

- DP diff stays in `[0, 1]`.
- Metrics are invariant to a permutation of group labels. Relabeling groups
  must not change the magnitude of a symmetric metric.
- Reference-oracle: the in-house metric matches Fairlearn or scikit-learn on a
  fixed seed.

## Uncertainty-aware fairness (3 metrics)

| Metric | What it captures |
| --- | --- |
| Group predictive entropy | Whether the model is systematically more uncertain on one group than another |
| Expected calibration error per group | Whether predicted confidence matches observed accuracy, separately per group |
| Selective-fairness AUC | How fairness behaves as low-confidence predictions are progressively abstained on, across coverage levels |

The headline replication: a model can pass DP and EO while failing group
entropy — fair on point, biased on uncertainty. A point-only audit will not
catch this. These three metrics are the contribution.

Calibration-error edge cases, such as empty bins and single-class groups, have
to be explicitly tested, not just executed without an error.

## Uncertainty estimators

| Estimator | Applies to | Returns |
| --- | --- | --- |
| MC Dropout | MLP only | predictive entropy, variance, mutual information |
| Deep ensemble | every model family (five seeds) | predictive entropy, variance, mutual information |
| Early-exit-assisted ensemble (QUTE-style) | MLP | predictive entropy, variance, mutual information |

Every estimator is profiled for parameter count, FLOPs, and peak RAM, because
the audit's own cost has to stay inside the budget it audits against. Entropy
is sanity-checked to correlate with error rate. The default estimator in the
public API is chosen from the resulting cost versus quality table.

## Explainers

| Explainer | Model mapping |
| --- | --- |
| SHAP | KernelExplainer (MLP), TreeExplainer (decision tree), LinearExplainer (logistic regression) |
| LIME | model-agnostic |
| Feature occlusion | the lightweight, microcontroller-feasible alternative |

Per-group feature-importance comparisons are produced and checked for
importance flips across sensitive groups. A flip is paper material. SHAP
KernelExplainer background samples are capped around 1k to keep runtimes
bounded. A cost table (explainer by model by dataset, in FLOPs and wall-clock)
goes with the results.

## Compression axis

Every one of the six metrics is re-run under compression:

- Post-training int8 quantization (onnxruntime for sklearn and ONNX,
  `torch.quantization` for the MLP).
- Magnitude pruning at 30, 50, 70, and 90 percent sparsity.

The headline question, does compression preserve fairness, is answered
empirically as a `(model, compression)` grid. int8 can wreck calibration. That
is a documented finding, and temperature scaling on the audit set is the
considered recovery step, not a way to hide the result.
