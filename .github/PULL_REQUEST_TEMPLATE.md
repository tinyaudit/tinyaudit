<!-- Keep PRs to one logical change. Squash on merge. -->

## What this changes

<!-- One paragraph. Link the issue it closes: "Closes #NN". -->

## Why

<!-- The motivation, not just the diff. -->

## Checklist

- [ ] One logical change; commit messages follow Conventional Commits
      (`feat:`, `fix:`, `test:`, `docs:`, `exp:`)
- [ ] `pre-commit run --all-files` passes (black + ruff + isort + mypy)
- [ ] `pytest` passes locally; new code has tests
- [ ] Coverage holds (90% on `fairness/` and `uncertainty/`, 70% overall)
- [ ] Docs / docstrings updated if behavior or the public API changed
- [ ] If a metric or result changed, the relevant CSV / figure is regenerated

## Notes for the reviewer

<!-- Anything that needs a careful human eye: a metric convention, an edge
case, a number that moved. -->
