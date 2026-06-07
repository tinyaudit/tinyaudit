# CLAUDE.md

This is the fast orientation for working in this repository. For the full
picture, read `docs/README.md` and follow the links from there.

## What this is

TinyAudit audits small AI models for fairness, explainability, and uncertainty.
It does this inside a memory and compute envelope sized for the same
microcontroller-class hardware the audited model runs on. It is benchmarked on
UCI Adult, Folktables (ACSIncome subset), and COMPAS. The sensitive attributes
are `sex` and `race`.

The primary artifact is an installable Python package (`pip install
tinyaudit`) plus a reproducible repository with full CI. The secondary artifact
is a one-page audit card schema and renderer. The target venue is the NeurIPS
Responsible AI workshop 2026, with ACM FAccT 2027 as the reach.

Status: scoped, baseline experiments not started. Most of the source tree below
does not exist yet. Build it against the contracts in `docs/architecture.md`,
bottom-up: the model protocol and adapters first, then data, profile, fairness,
uncertainty, xai, compress, card, the pipeline that wires them, and the CLI
last.

## Repository layout

- `src/tinyaudit/` is the package source.
  - `data/` holds the dataset loaders (`adult.py`, `folktables.py`,
    `compas.py`).
  - `models/` holds model wrappers. `base.py` defines the `AuditedModel`
    protocol.
  - `fairness/` holds the point metrics: `parity.py` (DP/EO/DI) and
    `_frames.py`.
  - `uncertainty/` holds MC dropout, the ensemble, the early-exit estimator,
    and the uncertainty-aware metrics.
  - `xai/` holds SHAP, LIME, and a lightweight occlusion explainer.
  - `compress/` holds int8 quantization and magnitude pruning.
  - `profile/` measures parameter count, FLOPs, and peak RAM.
  - `card/` holds the pydantic schema and the Jinja2 to HTML to PDF renderer.
  - `cli.py` is a thin wrapper over `audit()`.
- `tests/` mirrors the `src/` layout. `fixtures/` holds tiny deterministic
  data.
- `experiments/` holds reproducible scripts and Hydra configs. `results/` CSVs
  are tracked in git.
- `notebooks/` is exploratory only. It is never the source of truth.
- `paper/` holds the LaTeX source.
- `docs/` holds the project documentation and the mkdocs-material site.

## Tech stack

Python 3.11. Runtime deps: `numpy<2`, `pandas`, `scikit-learn`, `torch`,
`onnx`, `onnxruntime`, `shap`, `lime`, `captum`, `pydantic`, `jinja2`,
`weasyprint`, `click`, `rich`. `fairlearn` is a reference-only dependency. It
is used in tests as an oracle and must never be imported in the package hot
path. Dev tooling: `pytest`, `pytest-cov`, `hypothesis`, `mypy`, `ruff`,
`black`, `pre-commit`, `mkdocs-material`. Pin everything in a lockfile (uv or
pip-tools).

## Conventions

- Annotate types throughout. `mypy --strict` has to pass on `fairness/`,
  `uncertainty/`, and `card/`. Those are the modules that have to be
  unambiguous.
- Black with line length 100, Ruff for lint, isort for imports. Pre-commit
  enforces all of it and CI rechecks it. Do not push cleanup to a later branch.
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `exp:`). One logical
  change per commit. Co-author trailers on paired commits.
- `main` is always green and tagged. Work on short-lived `feature/<topic>`
  branches. Squash on merge. There is no long-running dev branch.

## Public API contract

The public surface is small. There are three entry points:

```python
from tinyaudit import audit, AuditCard

card: AuditCard = audit(
    model=clf,
    data=adult_test,
    sensitive="sex",
    methods=["fairness", "uncertainty", "xai"],
    compression="int8",
)
card.to_pdf("adult_lr_int8.pdf")
```

Everything else is internal. The CLI is a thin wrapper over `audit()` so any
run reproduces from a YAML config. Do not widen the public API without updating
`docs/conventions.md` in the same change.

## The pipeline

`audit()` runs seven stages in order. It ingests and schema-validates, profiles
params, RAM, and FLOPs, computes point fairness (DP/EO/DI), estimates
uncertainty (MC dropout, ensemble, early exit), computes uncertainty-aware
fairness (group entropy, ECE per group, selective-fairness AUC), runs
explainability (SHAP, LIME, occlusion), and renders the one-page card. Every
stage logs to a single JSON manifest. Every figure has a backing CSV. The card
is generated from the manifest and never hand-edited. See `docs/pipeline.md`
for the full description.

## Testing

There are three levels. Unit tests cover every metric, estimator, and
preprocessing step. Hypothesis property tests cover the metric module, for
example DP diff staying in `[0, 1]` and permutation invariance of group labels.
Reference-oracle tests check the in-house metric against Fairlearn or
scikit-learn on a fixed seed. Coverage targets are 90% on
`src/tinyaudit/fairness` and `src/tinyaudit/uncertainty`, and 70% overall. Keep
the full suite under five minutes. Mark known-flaky tests with
`@pytest.mark.flaky` and open a tracking issue.

## Reproducibility

Every experiment runs from a Hydra config in `experiments/configs` pinned to a
git SHA. Every output CSV is stamped with the config hash and the library
versions. `PYTHONHASHSEED`, torch, and numpy seeds all come from a single
`seed` flag. Determinism is guaranteed only on Linux/x86. Minor float drift
elsewhere is expected and is documented in the README.

## Correctness pitfalls

Verify these, do not assume them.

- The disparate-impact direction convention is the easiest place to silently
  ship a bug. Pin it in the docstring and in a dedicated test case.
- Multi-class and multi-valued sensitive attributes need explicit handling. Do
  not assume a binary attribute.
- Calibration-error edge cases (empty bins, single-class groups) have to be
  tested, not just executed.
- Sanity values: Adult DP diff is around 0.15 and COMPAS DI is well below 0.8.
  A result far from these is a signal to investigate, not to commit.
- int8 quantization can wreck calibration. That is a documented finding, not a
  bug to hide. Report it and consider temperature scaling on the audit set.

## Definition of done, per module

A module is not done when it runs. It is done when it is verified.

- `data/`: deterministic preprocessing, loader tests against known
  scikit-learn outputs, Folktables row counts and COMPAS `race` spot-checks
  passing.
- `fairness/`: three metrics, unit plus hypothesis plus one reference-oracle
  test each, DI direction convention pinned in a test.
- `uncertainty/`: three estimators returning entropy, variance, and mutual
  information, each profiled into a footprint CSV, entropy correlating with
  error rate.
- `uncertainty/metrics.py`: the decoupling result replicates on Adult and
  matches the source paper's qualitative claim.
- `compress/`: int8 and magnitude pruning at 30, 50, 70, and 90 percent, full
  six-metric grid re-run as a CSV.
- `card/`: pydantic schema validates, renderer is deterministic from the
  manifest, card is one page and traffic-light coded.

## Glossary

- DP, EO, DI: demographic-parity diff, equalized-odds diff, disparate-impact
  ratio. These are the point-prediction fairness metrics.
- ECE per group: expected calibration error computed separately per sensitive
  group.
- Selective-fairness AUC: how fairness behaves as low-confidence predictions
  are progressively abstained on, integrated across coverage.
- MC Dropout: Monte-Carlo dropout uncertainty estimator, MLP only.
- QUTE-style early exit: an early-exit-assisted ensemble, the on-device
  feasibility proof point.
- Audit card: the one-page, manifest-generated PDF deliverable.
- Manifest: the single JSON log every pipeline stage writes to.

## Build and test commands

```bash
pip install -e ".[dev]"      # editable install with dev extras
pre-commit run --all-files   # black + ruff + isort + mypy
pytest --cov=src/tinyaudit   # full suite with coverage
mkdocs serve                 # local docs preview
```

## Where to read more

`docs/architecture.md` is the build blueprint: the public API, the model
protocol, every module's responsibility, and the key signatures to implement
against. `docs/README.md` indexes the rest: the overview, the pipeline,
datasets, the metrics reference, conventions, the engineering constraints, and
the implementation risks. When a signature or behavior is ambiguous,
`docs/architecture.md` is the contract.
