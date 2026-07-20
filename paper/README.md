# Paper source

Workshop paper draft for TinyAudit. Target venue: NeurIPS Responsible AI
workshop 2026 (reach: ACM FAccT 2027).

## Build

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## What is here

- `main.tex` — the full draft. All nine sections are written, every results
  table is filled with **real mean ± std numbers over seeds**, and both findings
  plus the on-device feasibility claim are prose-complete. The single remaining
  `\todo{...}` (renders in red) is the one figure export — see `figures/` below.
- `refs.bib` — 26 references, all cited, matching the project literature review.
  Several carry a `\todo{verify venue}` note: confirm the exact venue/pages
  against the canonical source before camera-ready.
- `figures/` — export chart PDFs here (`compression.pdf`, the decoupling slope
  charts). The web one-pager (`web/index.html`) renders the same charts.

## Numbers are not hand-entered

Every table value comes from `experiments/results/*_multiseed.csv`. To change a
number, regenerate the aggregates and update the table:

```bash
python experiments/run_multiseed.py         # writes *_multiseed.csv (seeds 0-9)
python experiments/run_audit_footprint.py   # writes audit_footprint*_.csv
```

The feasibility table (Section 6) comes from `run_audit_footprint.py`, which
profiles the **audit pipeline itself** (not the audited model) with
`tracemalloc` as a function of streaming batch size. The headline is that the
audit's peak RAM is ≈ 0.6 MB + 2.3 KB/sample and constant in test-set size.

## What is left

1. Export the crossing demographic-parity / ECE chart to
   `figures/compression.pdf` and uncomment the `\includegraphics` in Section 5
   (the last `\todo`). The web one-pager renders the same chart.
2. Verify the `\todo{verify venue}` bib entries against canonical sources.
3. Swap `\documentclass` for the venue style file at submission time.
