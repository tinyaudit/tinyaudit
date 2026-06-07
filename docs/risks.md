# Implementation Risks

These are the engineering traps in building this package. Each one is tracked
as an open issue labeled `risk`. The mitigation is the agreed response, not a
hope.

| # | Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| 1 | Disparate-impact direction shipped backwards | Medium | High | Pin the convention in the docstring and a dedicated test. Reference-oracle test against Fairlearn on a fixed seed. A backwards ratio is the single easiest silent bug here |
| 2 | int8 quantization wrecks calibration, so ECE-per-group is meaningless | Medium | Medium | Treat it as a finding, not a failure. Add optional temperature scaling on the audit set as a recovery step and re-run. Record both results |
| 3 | Cross-machine non-determinism makes numbers unreproducible | Medium | High | Single `seed` flag drives `PYTHONHASHSEED`, torch, and numpy. Pin deps in a lockfile. Stamp every CSV with the config hash and versions. Guarantee determinism only on Linux/x86 and document the caveat |
| 4 | Multi-valued sensitive attributes mishandled (race is not binary) | Medium | High | The loader returns raw sensitive columns. Metrics take the grouping explicitly. Property test: permutation invariance of group labels. No implicit binary assumption anywhere |
| 5 | Calibration edge cases crash or silently skew (empty bins, single-class groups) | Medium | Medium | Explicit handling and unit tests for empty bins and single-class groups. Edge cases are tested, not just executed |
| 6 | Footprint numbers do not reflect the claimed envelope | Low | Medium | `profile_model()` measures peak RAM via tracemalloc and FLOPs via thop or fvcore for every model and every compression setting, including the estimators themselves |
| 7 | The uncertainty module balloons and blocks everything downstream | High | Medium | It is the one module nothing else can proceed without. Keep its interface (`UncertaintyOutput`) frozen early so metrics and the card can be built against it in parallel |
| 8 | CI flakiness blocks merges | Medium | Low | Keep the full suite under five minutes. Mark known-flaky tests with `@pytest.mark.flaky` and a tracking issue |

## What to watch

The two high-impact correctness risks are 1, 3, and 4. They share a shape: the
code runs, produces a plausible number, and is wrong. A run that executes is
not evidence of a correct metric. The defenses are the same in each case: a
pinned convention, a reference oracle, a property test, and a sanity value the
output is checked against (Adult DP diff around 0.15, COMPAS DI well below
0.8).

Risk 7 is a sequencing risk, not a correctness one. The uncertainty estimators
and the uncertainty-aware metrics are the part of the system nothing else can
substitute for, so freeze the `UncertaintyOutput` dataclass before building
outward from it. Once that interface is stable, the metrics, the card schema,
and the renderer can all be built and tested against it without waiting for the
estimators to be finished.
