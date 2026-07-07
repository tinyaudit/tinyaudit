# ADR 0000 — The core contribution is the uncertainty pillar

- Status: **Accepted** (per `plan.pdf`; confirm in writing before submission)
- Date: 2026-05-18
- Deciders: Atharva Doke, Kasyap Tumuluri

## Context

TinyAudit fuses three things — point-prediction fairness, predictive
uncertainty, and explainability — under a microcontroller-class memory and
FLOPs budget. We need one of these to be *the* contribution, the thing a
reviewer remembers, so the paper has a spine.

The literature gap analysis (see the lit review) is specific:

- AIF360 / Fairlearn / Aequitas assume cloud compute and test-time group
  labels; they do not address on-device constraints (Lee & Singh 2021; Han et
  al. 2024).
- Kuzucu et al. (2023) introduce uncertainty-based fairness measures but
  evaluate on full-size models.
- QUTE (Ghanathe & Wilton 2024) delivers on-device uncertainty but never
  evaluates through a fairness lens.

The intersection — on-device uncertainty estimation *as a fairness auditing
signal* — is empty.

## Decision

The **uncertainty pillar is the core contribution**: group-conditional
predictive uncertainty (group entropy, ECE per group, selective-fairness AUC)
measured within an edge-feasible footprint, with a QUTE-style early-exit
ensemble as the on-device feasibility proof point. Point fairness and
explainability are supporting evidence.

The differentiator that must survive scrutiny is **edge feasibility under
compression**: "does the audit, and the fairness picture it paints, survive
int8 quantization and magnitude pruning?"

## Consequences

- Week 5 (uncertainty-aware metrics, Kuzucu replication) is the schedule's
  keystone; if it slips, cut the lightweight XAI alternative first (per the
  risk register), never the uncertainty work.
- If FairlyUncertain (Rosenblatt & Witter 2024) turns out to already cover the
  gap, pivot the headline to **fairness drift under compression** — the
  edge-feasibility angle still stands.
- The novelty claim itself (what is new, why it matters) must be written by the
  authors from their reading of the literature, not auto-generated.
