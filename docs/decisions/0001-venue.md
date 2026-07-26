# ADR 0001 — Target venue

- Status: **Accepted** (dates verified against official sites 2026-07-26)
- Date: 2026-05-18, revised 2026-07-26
- Deciders: Atharva Doke, Kasyap Tumuluri

## Context

A 10-week build by two high-school authors needs a venue whose format and bar
match the artifact: a working repo plus a short paper, with high-school
authorship welcomed.

> Dates below were checked against the official venue sites on 2026-07-26.
> Re-verify before each submission; workshop CFPs move.

## Decision

- **Primary: a NeurIPS 2026 workshop.** NeurIPS 2026 workshops run **Dec 11–12
  (Sydney)** and **Dec 12–13 (Paris, Atlanta)**. The relevant deadline is the
  **suggested submission date for workshop contributions, 2026-08-29 AoE**,
  with mandatory accept/reject notification by **2026-09-29**. Workshop
  proposals were decided 2026-07-11, so the accepted-workshop list is settling
  now and individual workshop CFPs are appearing on their own sites. Pick the
  specific workshop from that list; typical format is 4 pages, non-archival.
  - Correction to the earlier draft of this ADR: there is no "October 2026"
    NeurIPS RAI workshop. October was a working assumption and was wrong in
    both month and framing. The real constraint is the late-August
    contribution deadline.
- **Reach: ACM FAccT 2027.** Abstracts **2026-10-27**, full papers
  **2026-11-03**, initial decisions 2026-12-22, rebuttal 2027-01-28, final
  notification 2027-03-23, conference **2027-06-21–24**. Up to 14 pages
  excluding references, with both archival and non-archival tracks and a
  revision round. No student or short-paper track. The workshop paper feeds
  into it: expand 4 → 14 pages, add an ablation and a human-readability study
  of the audit card (the latter requires IRB).
  - The FAccT timeline sits *after* the NeurIPS workshop deadline and before
    the workshop itself, so the intended sequence works: submit the short
    paper in August, submit the extended version in early November, present
    at the workshop in December.

## Alternatives considered

- **ICLR 2027 fairness workshops.** The ICLR 2026 edition (AFAA, "Algorithmic
  Fairness Across Alignment Procedures and Agentic Systems") ran a **3-page
  tiny-paper track** alongside a 6–9 page main track, non-archival with opt-in
  PMLR proceedings, and explicitly accepted work under review elsewhere. Its
  2026 deadline (Feb 5) has passed, but a 2027 edition would be a strong fit
  and the tiny-paper track matches this artifact well. Watch for the CFP.
- **AIES (Feb–Mar 2027)** — backup if FAccT does not work out.
- **tinyML Research Symposium** — engineering-leaning sibling; reframe around
  "audit-aware compression." Caveat found 2026-07-26: no 2027 CFP is posted
  and the series shows no activity after 2025, so treat it as dormant rather
  than a plannable target.
- **MLSys On-Device ML workshops** — worth checking depending on co-locations.

## Consequences

- Submit to the workshop as student authors even if no mentor materializes by
  end of Week 2 (workshops are forgiving); re-engage on the FAccT extension
  once the workshop paper exists.
- License is Apache-2.0 (FAccT prefers permissive; NeurIPS RAI is flexible) —
  see the repository `LICENSE`.
