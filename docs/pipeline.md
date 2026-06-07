# The Pipeline, End to End

A single audit run takes a trained model, a dataset, and a sensitive attribute
name, and produces an audit card. Internally it runs seven stages in order.

```
trained model  \
dataset         >  1 ingest -> 2 profile -> 3 point fairness -> 4 uncertainty
sensitive attr /                                                     |
                                                                     v
  audit card <- 7 render <- 6 explainability <- 5 uncertainty-aware fairness
```

## 1. Ingest

Load the model (scikit-learn, PyTorch, or ONNX), load the dataset, and validate
the schema. A mismatched column, an unexpected sensitive-attribute encoding, or
a missing label fails here, loudly, before any compute is spent.

## 2. Profile

Measure parameter count, peak RAM, FLOPs, and wall-clock inference per sample.
These numbers are what make the "sized for microcontroller-class hardware"
claim checkable, so they are recorded for every model and every compression
setting.

## 3. Fairness (point)

Compute the demographic-parity difference, the equalized-odds difference, and
the disparate-impact ratio. These are implemented from scratch. Fairlearn is
used only as a test-time oracle, never in this hot path. The disparate-impact
direction convention is pinned in code and in a test, because it is the easiest
place in the whole system to silently ship a bug.

## 4. Uncertainty

Estimate predictive uncertainty with MC Dropout (MLP only), deep ensembles
(five seeds, every model family), and a QUTE-style early-exit-assisted ensemble
(MLP). Each estimator returns predictive entropy, predictive variance, and
mutual information, and each one is itself profiled for parameter count, FLOPs,
and peak RAM. The audit's own cost has to stay inside the budget it audits
against.

## 5. Fairness (uncertainty-aware)

This is the stage the contribution hinges on. Compute group predictive entropy,
expected calibration error per group, and selective-fairness AUC across
coverage levels. The target replication is a model that passes demographic
parity and equalized odds but fails group entropy — fair on point, biased on
uncertainty. A confidence-gated constraint-activation lens reads the model's
behavior without modifying the model.

## 6. Explainability

Run SHAP (KernelExplainer for the MLP, TreeExplainer for the decision tree,
LinearExplainer for logistic regression) and LIME, plus a lightweight
feature-occlusion explainer as the microcontroller-feasible alternative.
Produce per-group feature-importance comparisons and flag importance flips
across sensitive groups. SHAP KernelExplainer background samples are capped
around 1k to keep runtimes sane.

## 7. Render

Generate the one-page audit card: Markdown to HTML to PDF via Jinja2 and
WeasyPrint, from a pydantic schema. The card is traffic-light coded and
readable at a glance by a non-expert, because non-experts demonstrably
misread raw fairness metrics.

## Invariants

These hold for every run, and they are why the system is reproducible.

- Every stage logs to a single JSON manifest.
- Every figure has a corresponding CSV.
- The audit card is generated from the manifest and never hand-edited.
- The CLI and the Python API call the same `audit()` function, so any card
  reproduces from a YAML config.

See [metrics.md](metrics.md) for the precise definition and direction
convention of every metric named above.
