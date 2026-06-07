# Overview

## What TinyAudit is

TinyAudit is a Python package that audits a small trained model for fairness,
explainability, and uncertainty, and produces a one-page audit card. It runs
inside a memory and compute budget sized for the same microcontroller-class
hardware the audited model would run on.

It is built and benchmarked on three datasets: UCI Adult, Folktables
(ACSIncome subset), and COMPAS. The sensitive attributes are `sex` and `race`.

## The core idea

Three things are usually measured in isolation, by three different tool stacks:

- Point-prediction fairness (demographic parity, equalized odds, disparate
  impact).
- Predictive uncertainty (entropy, variance, mutual information).
- Feature-level explainability (SHAP, LIME, occlusion).

TinyAudit measures all three in one pipeline, per sensitive group, and re-runs
them under compression. The reason this is worth doing: a compressed model can
pass demographic parity while being systematically more confident on
majority-group inputs and hesitant on minority-group inputs. A point-only audit
will not catch that. Adding the uncertainty-aware metrics will.

## What it produces

You give it a trained model, a dataset, and a sensitive attribute name. It
returns an `AuditCard` and writes a JSON manifest plus per-figure CSVs. Every
reported number reproduces from a pinned config: parameter count, peak RAM,
FLOPs, demographic-parity diff, equalized-odds diff, ECE per group, and
selective-fairness AUC.

## How it is structured

A single run executes seven stages in order: ingest, profile, point fairness,
uncertainty, uncertainty-aware fairness, explainability, render. The runtime
dataflow is in [pipeline.md](pipeline.md). The module-by-module build
blueprint, including the public API, the model protocol, and every key
signature, is in [architecture.md](architecture.md).

The package keeps its public surface to two names, `audit` and `AuditCard`.
Everything else is internal. The CLI is a thin wrapper over `audit()` so any
card reproduces from a YAML config.

## Status

Scoped. Baseline experiments not started. Most of the source tree does not
exist yet. Build it against the contracts in
[architecture.md](architecture.md), bottom-up, with the testing and CI
standards in [conventions.md](conventions.md). The engineering constraints and
known correctness traps are in
[scope-and-limitations.md](scope-and-limitations.md) and
[risks.md](risks.md).
