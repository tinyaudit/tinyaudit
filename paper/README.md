# Paper source

- `tinyaudit_workshop.tex` is the short paper. 5 pages plus references. This is
  the one to submit.
- `main.tex` is the full length draft, used as the basis for a longer version.
- `neurips_2026.sty` is the official NeurIPS 2026 style. `reference_template.tex`
  is the unmodified upstream example.

## Build

```bash
pdflatex tinyaudit_workshop.tex
pdflatex tinyaudit_workshop.tex     # twice, for references and figure refs
```

No bibtex step. References are a hand written `thebibliography` block in the
file. `main.tex` uses `refs.bib` and does need the full cycle:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Figures come from `python experiments/make_paper_figures.py`. Table values come
from `experiments/results/*_multiseed.csv`, so regenerate those rather than
editing numbers by hand.

## Submission and camera-ready modes

Change only the `\usepackage` line in `tinyaudit_workshop.tex`. Each of these
was verified by building it:

| Options | Line numbers | Authors | Footer |
|---------|--------------|---------|--------|
| `[dblblindworkshop]` | on | anonymous | Submitted to ... Do not distribute |
| `[dblblindworkshop,nonanonymous]` | on | shown | Submitted to ... Do not distribute |
| `[dblblindworkshop,final]` | off | shown | workshop track line |

Use `[dblblindworkshop]` to submit and `[dblblindworkshop,final]` for camera
ready. Do not use `nonanonymous` for camera ready. It reveals the names but
leaves the line numbers and the "Do not distribute" footer in place.

The line numbers in the submission build are not a bug. The style loads `lineno`
on purpose so reviewers can cite a specific line. Set the venue string with
`\workshoptitle{...}`, which only appears in `final` mode.

Submission mode blanks `\ack` automatically, but it cannot blank a repository
URL, a dataset path, or a self citation. Scrub those by hand.

## Venues

Dates checked 2026-07-26. Re-verify on the venue site before relying on them.

| Venue | Deadline | Length | Notes |
|-------|----------|--------|-------|
| [ODI](https://odi2026.github.io/), NeurIPS 2026 | 2026-08-29 | 5 pages excl. refs | On-Device Intelligence. Non archival, double blind. Sydney, Dec 11 to 12. Closest fit. |
| [LIGHT](https://almaai-disi-unibo.github.io/neurips2026-light-smallModels/), NeurIPS 2026 | see site | see site | Deployable Small Foundation Models. Paris, Dec 12 to 13. Strong topical match, details were not posted as of 2026-07-26. |
| [AI4GOOD](https://trustworthy-ai-for-good.github.io/), NeurIPS 2026 | TBD | 2 to 8 pages | Trustworthy AI for Good. Non archival, double blind. Paris. Highlighted track is multi agent safety. |
| [FAccT 2027](https://facctconference.org/2027/cfp.html) | 2026-10-27 abstract, 2026-11-03 paper | 14 pages excl. refs | Archival and non archival tracks. Conference 2027-06-21 to 24. |

There is no NeurIPS Responsible AI workshop in 2026.

Worth watching: ICLR fairness workshops, whose 2026 edition (AFAA) had a 3 page
tiny paper track and accepted work under review elsewhere. The tinyML Research
Symposium has no 2027 call for papers and no activity after 2025.
