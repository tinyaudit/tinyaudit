# Conventions and Standards

## Python version

Python 3.11. Annotate types throughout. Run `mypy --strict` on the `fairness`,
`uncertainty`, and `card` modules, since those are the parts that have to be
unambiguous.

## Style

Black with line length 100, Ruff for linting, isort for imports. Pre-commit
enforces all of it and CI rechecks it. No exceptions, and no "I'll clean it up
later" branches.

## Dependencies

Pinned via uv or pip-tools in a lockfile.

- Runtime: `numpy<2`, `pandas`, `scikit-learn`, `torch`, `onnx`,
  `onnxruntime`, `shap`, `lime`, `captum`, `pydantic`, `jinja2`, `weasyprint`,
  `click`, `rich`.
- Reference only: `fairlearn`. It is imported in tests as an oracle, never in
  the package hot path.
- Dev: `pytest`, `pytest-cov`, `hypothesis`, `mypy`, `ruff`, `black`,
  `pre-commit`, `mkdocs-material`.

## Testing strategy

There are three levels.

1. Unit tests on every metric, every estimator, and every preprocessing step.
2. Property tests via `hypothesis` on the metric module, for example DP diff
   staying in `[0, 1]` and permutation invariance of group labels.
3. Reference-oracle tests where the in-house metric is checked against
   Fairlearn or scikit-learn on a fixed seed.

Coverage targets: 90% line coverage on `src/tinyaudit/fairness` and
`src/tinyaudit/uncertainty`, 70% overall. Keep the full suite under five
minutes. Mark known-flaky tests with `@pytest.mark.flaky` and open a tracking
issue.

## Reproducibility

- Every experiment runs from a Hydra config in `experiments/configs`.
- Every config is pinned to a git SHA.
- Every output CSV is stamped with the config hash and the library versions.
- `PYTHONHASHSEED`, torch, and numpy seeds all come from a single `seed` flag.
- Determinism is documented and tested on Linux/x86. Minor float drift on other
  platforms is acknowledged in the README, not papered over.

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

Everything else is internal. The CLI is a thin wrapper over the same function
so an audit reproduces from a YAML config. Widening the public API means
updating this document in the same change.

## Documentation

mkdocs-material, hosted on GitHub Pages, auto-built on tag. Sections:
Quickstart, Concepts (point versus uncertainty fairness), Metrics reference,
API reference (from docstrings via mkdocstrings), and Reproducing the paper.

## CI configuration

GitHub Actions, three jobs running in parallel on every push:

- lint: ruff, black `--check`, mypy.
- test: pytest with coverage on Python 3.11 and 3.12.
- build: package builds plus an import smoke test.

A nightly cron runs the full experiment grid on a single seed and posts the
summary CSV to a results-bot issue.

## Release process

Tag `v0.1.0` the day the paper is submitted. PyPI publish via Trusted
Publisher. Zenodo automatically issues a DOI on tagged releases, wired up
early. The DOI is cited in the paper.
