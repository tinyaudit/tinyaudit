# ADR 0001 — Target venue

- Status: **Proposed** (deadlines must be verified on the official sites before
  this is Accepted)
- Date: 2026-05-18
- Deciders: Atharva Doke, Kasyap Tumuluri

## Context

A 10-week build by two high-school authors needs a venue whose format and bar
match the artifact: a working repo plus a short paper, with high-school
authorship welcomed.

> Deadlines change and are outside the assistant's knowledge cutoff. **Verify
> every date on the venue's official website before committing.** The dates
> below are the plan's working assumptions, not confirmed facts.

## Decision

- **Primary: NeurIPS Responsible AI workshop, October 2026.** ~4-page short
  paper, non-archival, fast turnaround, high-school authorship welcomed. This
  is the right shape for the 10-week build. An accepted workshop paper is
  citable while a fuller version goes through a main conference.
- **Reach: ACM FAccT 2027** (deadline ~Jan–Feb 2027). Archival main-conference
  venue; needs a mentor as senior author. The workshop paper feeds into it:
  expand 4 → 8–10 pages, add an ablation and a human-readability study of the
  audit card (the latter requires IRB).

## Alternatives considered

- **tinyML Research Symposium (Mar–Apr 2027)** — engineering-leaning sibling;
  reframe around "audit-aware compression." Same artifacts, hardware audience.
- **AIES (Feb–Mar 2027)** — backup if FAccT does not work out.
- **MLSys On-Device ML workshops** — worth checking depending on co-locations.

## Consequences

- Submit to the workshop as student authors even if no mentor materializes by
  end of Week 2 (workshops are forgiving); re-engage on the FAccT extension
  once the workshop paper exists.
- License is Apache-2.0 (FAccT prefers permissive; NeurIPS RAI is flexible) —
  see the repository `LICENSE`.
