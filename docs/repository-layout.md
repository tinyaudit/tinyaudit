# Repository Layout

One mono-repo. The goal is for it to be boring and easy to find your way
around.

```
tinyaudit/
├── README.md                     # 5-minute pitch + install + quickstart
├── LICENSE                       # Apache-2.0
├── CITATION.cff                  # auto-recognized by GitHub
├── pyproject.toml                # PEP 517 packaging (no setup.py)
├── .pre-commit-config.yaml       # black + ruff + isort + nbstripout
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                # pytest + lint on every push
│   │   ├── docs.yml              # mkdocs build on tag
│   │   └── release.yml           # PyPI publish on tag
│   ├── ISSUE_TEMPLATE.md
│   └── PULL_REQUEST_TEMPLATE.md
├── src/
│   └── tinyaudit/
│       ├── __init__.py           # public API
│       ├── data/                 # dataset loaders
│       │   ├── adult.py
│       │   ├── folktables.py
│       │   └── compas.py
│       ├── models/               # model wrappers
│       │   ├── base.py           # AuditedModel protocol
│       │   ├── sklearn.py
│       │   └── torch.py
│       ├── fairness/             # point-prediction metrics
│       │   ├── parity.py         # DP, EO, DI
│       │   └── _frames.py        # MetricFrame-equivalent
│       ├── uncertainty/          # the novel module
│       │   ├── mc_dropout.py
│       │   ├── ensemble.py
│       │   ├── early_exit.py     # QUTE-inspired
│       │   └── metrics.py        # group entropy, ECE/group, selective fairness
│       ├── xai/                  # explainability
│       │   ├── shap_wrap.py
│       │   ├── lime_wrap.py
│       │   └── occlusion.py      # lightweight alternative
│       ├── compress/             # quantization + pruning
│       │   ├── quantize.py
│       │   └── prune.py
│       ├── profile/              # memory + FLOPs
│       │   └── footprint.py
│       ├── card/                 # audit card renderer
│       │   ├── schema.py         # pydantic models
│       │   ├── render.py         # Jinja2, HTML, PDF
│       │   └── templates/
│       │       └── audit_card.html.j2
│       └── cli.py                # `tinyaudit run ...`
├── tests/                        # pytest, mirrors src/ layout
│   ├── fairness/
│   ├── uncertainty/
│   └── fixtures/                 # tiny deterministic datasets
├── notebooks/                    # exploratory; not the source of truth
│   ├── 01_eda_adult.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_kuzucu_replication.ipynb
│   └── 04_compression_sweep.ipynb
├── experiments/                  # reproducible scripts
│   ├── configs/                  # hydra / yaml configs
│   ├── run_baselines.py
│   ├── run_compression_sweep.py
│   └── results/                  # CSVs (git-tracked), figures (built)
├── paper/                        # LaTeX source
│   ├── main.tex
│   ├── sections/
│   ├── figures/                  # auto-built from experiments/results
│   └── references.bib
└── docs/                         # mkdocs site, deployed via Pages
```

## Account and visibility

Public organization on GitHub named `tinyaudit`. Two members, plus a mentor
invited as a collaborator once confirmed. License: Apache-2.0. The archival
fairness venue prefers a permissive license, and the workshop is flexible.

## Branching model

`main` is always green and tagged. Feature work happens on short-lived
`feature/<topic>` branches. Each PR needs CI green plus one human review. The
authors review each other, and the mentor reviews the big merges. Squash on
merge. No long-running dev branch.

## Commit hygiene

Conventional Commits format (`feat:`, `fix:`, `test:`, `docs:`, `exp:`). One
logical change per commit. Co-author trailers on every paired commit, so
authorship is unambiguous if reviewers check the git log.

## Issue and project tracking

A GitHub Projects (v2) board with three columns: Backlog, In Progress, Done.
Every week of the plan becomes a milestone. Every deliverable becomes an issue.
Risk-register items become open issues labeled `risk`.

## Decision records

Architecture decisions live as dated ADRs in `docs/decisions/`. For example,
`0000-di-direction.md` pins the disparate-impact direction convention and
`0001-default-estimator.md` records which uncertainty estimator is the public
default and why, based on the cost versus quality table. ADRs are append-only.
Supersede, do not rewrite.
