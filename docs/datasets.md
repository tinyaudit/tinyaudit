# Datasets

TinyAudit is benchmarked on three standard datasets. The sensitive attributes
throughout are `sex` and `race`.

| Dataset | Task | Role |
| --- | --- | --- |
| UCI Adult Income | Predict income over \$50K | Primary replication target |
| Folktables (ACSIncome subset) | Predict income threshold | Headline benchmark |
| COMPAS Recidivism | Predict two-year recidivism | Secondary, historical comparability |

## Preprocessing principle

Every preprocessing decision is documented in code, not in a notebook. The
loaders in `src/tinyaudit/data/{adult,folktables,compas}.py` are deterministic
and are the single source of truth for how each dataset is prepared. Notebooks
can explore the data but they never define the canonical preprocessing.

## UCI Adult Income

The classic income-prediction benchmark, and the dataset on which the point
versus uncertainty fairness decoupling is replicated. Preprocessing is
sanity-checked against a published reference so the baseline fairness numbers
are comparable to the literature.

Sanity value: the demographic-parity difference should be on the order of 0.15.
A baseline far from this is a signal to investigate the preprocessing before
trusting any downstream metric.

## Folktables (ACSIncome subset)

The modern replacement for Adult, drawn from US Census microdata. It is the
headline benchmark because it avoids the well-documented problems that led the
field to retire Adult, while staying directly comparable. Row counts are
verified against the official Folktables documentation on ingest.

## COMPAS Recidivism

Kept as a secondary benchmark for historical comparability with the broad
fairness literature. Reviewer fatigue with COMPAS is real, so results lead with
Folktables and treat COMPAS as supporting evidence rather than the headline.
The `race` column is spot-checked on a sample of records during ingest
validation.

Sanity value: the disparate-impact ratio should be well below 0.8.

## Validation checklist

On every ingest, the loaders and their tests verify:

- The schema matches the expected columns and dtypes.
- Sensitive attribute encodings are explicit. There is no implicit binary
  assumption for a multi-valued attribute like `race`.
- Folktables row counts match the published documentation.
- A random sample of COMPAS records is spot-checked on `race`.
- Baseline fairness numbers land near the sanity values above. Any field that
  looks wrong is filed as an issue, not silently accepted.
