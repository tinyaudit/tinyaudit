# Paper source

Workshop paper draft for TinyAudit. Target venue: NeurIPS Responsible AI
workshop 2026 (reach: ACM FAccT 2027).

## Build

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## What is here

- `main.tex` — the scaffold. Section skeleton, drafted abstract, and every
  results table pre-filled with **real mean ± std numbers over 10 seeds**. Prose
  still to write is marked `\todo{...}` (renders in red).
- `refs.bib` — reference stubs to fill in.
- `figures/` — export chart PDFs here (`compression.pdf`, the decoupling slope
  charts). The web one-pager (`web/index.html`) renders the same charts.

## Numbers are not hand-entered

Every table value comes from `experiments/results/*_multiseed.csv`. To change a
number, regenerate the aggregates and update the table:

```bash
python experiments/run_multiseed.py   # writes *_multiseed.csv (seeds 0-9)
```

## Draft order (biggest wins first)

1. `\todo` in Section 6 (feasibility): measure the **audit pipeline's** own peak
   RAM / FLOPs, not just the model's. This is the novelty claim.
2. Intro (Section 1): the two-trend collision and the contributions.
3. Related work (Section 7): position against uncertainty-aware fairness,
   fairness-under-compression, and on-device auditing.
4. Export the figures and drop them into `figures/`.
