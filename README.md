# TinyAudit

[![CI](https://github.com/tinyaudit/tinyaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/tinyaudit/tinyaudit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pypi-unreleased-lightgrey)](#)
[![Coverage](https://img.shields.io/badge/coverage-target%2090%25-lightgrey)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-on%20release-lightgrey)](#)

A lightweight pipeline that audits small AI models for fairness,
explainability, and uncertainty, sized for the same hardware the model itself
runs on.

## What is this?

The common fairness toolkits (AIF360, Fairlearn, Aequitas) assume cloud-scale
compute, test-time demographic labels, and unlimited notebook memory. None of
that holds on a 256 KB microcontroller, and compressed models are already
running in clinics, on farms, and in risk-scoring tools. Those toolkits also
ignore predictive uncertainty, so a compressed edge model can pass demographic
parity while being more confident on majority-group inputs and hesitant on
minority-group inputs. A point-prediction audit will not catch that.

TinyAudit is the first auditing pipeline that measures point-prediction
fairness, group-conditional predictive uncertainty, and feature-level
explainability together, under a fixed memory and FLOPs budget consistent with
microcontroller-class deployment.

## What does it produce?

You give it a trained model, a dataset, and a sensitive attribute name. You get
back a one-page audit card that is traffic-light coded and readable at a
glance, backed by a JSON manifest and per-figure CSVs. Every reported number
(parameter count, peak RAM, FLOPs, demographic-parity diff, equalized-odds
diff, ECE per group, selective-fairness AUC) reproduces from a pinned config.

## Install

```bash
pip install tinyaudit
```

Status: scoped, package not yet released. Until the first tagged release,
install from source with `pip install -e ".[dev]"`.

## Quickstart: audit a model on Adult in 60 seconds

```python
from tinyaudit import audit, AuditCard
from tinyaudit.data import load_adult

X_test, y_test, sensitive = load_adult(split="test")

card: AuditCard = audit(
    model=clf,                       # a fitted sklearn estimator (auto-wrapped)
    data=(X_test, y_test),
    sensitive=sensitive["sex"],      # one protected attribute at a time
    dataset="adult",
)
print(card.fairness.metrics)         # DP, EO, DI with traffic-light bands
open("adult_lr.html", "w").write(card.to_html())
```

The CLI wraps the same function, so any audit reproduces from a config file:

```bash
tinyaudit run --config experiments/configs/adult_logreg.json
```

Implemented today: scikit-learn and PyTorch models; the Adult, COMPAS, and
Folktables loaders; point-prediction fairness (DP/EO/DI); the three uncertainty
estimators (MC dropout, deep ensemble, QUTE-style early exit) and the
uncertainty-aware metrics; SHAP/LIME/occlusion explainability; int8
quantization and magnitude pruning; profiling; and the audit card (HTML
always; PDF where WeasyPrint's native libraries are available). Still to come:
the experiment grid CSVs, the Kuzucu and FairlyUncertain replications, and the
paper.

## Architecture at a glance

```
trained model  \
dataset         >  ingest -> profile -> point fairness -> uncertainty
sensitive attr /                                              |
                                                              v
       audit card <- render <- explainability <- uncertainty-aware fairness
```

Seven stages, one JSON manifest, one audit card. See
[docs/pipeline.md](docs/pipeline.md) for the full walkthrough.

## Documentation

The full documentation lives in [docs/](docs/README.md):

- [Overview](docs/overview.md)
- [Architecture and module spec](docs/architecture.md)
- [Pipeline](docs/pipeline.md) and [repository layout](docs/repository-layout.md)
- [Datasets](docs/datasets.md) and [metrics reference](docs/metrics.md)
- [Conventions and standards](docs/conventions.md)
- [Engineering constraints](docs/scope-and-limitations.md)
- [Implementation risks](docs/risks.md)

## Citation

A `CITATION.cff` will land with the first tagged release. Cite the Zenodo DOI
issued on release. Until then, cite this repository directly.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Authors

Atharva Doke and Kasyap Tumuluri, Downingtown East High School.
