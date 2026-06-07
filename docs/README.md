# TinyAudit Documentation

These docs are the build blueprint for the package. They say what to build,
how the pieces fit, and the standards every module ships with. They are also
the content for the mkdocs-material site deployed to GitHub Pages.

## Start here

Read these in order before writing code:

1. [Overview](overview.md). What the package is, the core idea, and how it is
   structured.
2. [Architecture and module spec](architecture.md). The build blueprint: the
   public API, the model protocol, every module's responsibility, and the key
   signatures to implement against.
3. [Pipeline](pipeline.md). The seven stages of a single audit run and the
   runtime dataflow between them.

## Reference

- [Repository layout](repository-layout.md). The mono-repo tree, the branching
  model, commit hygiene, and issue tracking.
- [Datasets](datasets.md). UCI Adult, Folktables (ACSIncome), and COMPAS. The
  loader contract, the preprocessing, and the validation checks.
- [Metrics reference](metrics.md). The six audit metrics, the three uncertainty
  estimators, and the explainers, with definitions and conventions.
- [Conventions and standards](conventions.md). Python version, style,
  dependencies, the testing strategy, reproducibility, the public API
  contract, documentation, CI, and the release process.

## Constraints

- [Engineering constraints](scope-and-limitations.md). What the numbers can and
  cannot claim, and the correctness traps to build around.
- [Implementation risks](risks.md). The engineering risks, their likelihood and
  impact, and the mitigation for each.

## Editing these docs

The docs are the source of truth. Notebooks are exploratory and never
authoritative. If a notebook and a doc disagree, either the doc wins or the doc
is wrong, so fix one of them.

Architecture and policy decisions are recorded as dated ADRs in
`docs/decisions/`. Do not rewrite an ADR. Supersede it with a new one.

If you change the public API or a cross-module signature, update
[architecture.md](architecture.md) in the same change. The blueprint and the
code do not get to drift apart.
