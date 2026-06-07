# Engineering Constraints

What the package can and cannot claim about the numbers it produces. Build to
these constraints so the audit card never overstates what was measured.

## Edge is simulated, not deployed

There is no physical microcontroller in the loop. Nothing is flashed to an
STM32 or ESP32, and no real on-device latency or power draw is measured.
Everything edge-related is software-simulated.

Footprint numbers (parameter count, FLOPs, peak RAM) are reported as
consistency evidence against a stated budget, roughly 32 to 512 KB SRAM and 1
to 4 MB flash, not as measured hardware results. The renderer should label them
that way. The supported claim is "sized for MCU-class hardware," not "deployed
on an MCU."

## Reproducibility is platform-bound

Floating-point determinism varies by platform. Seeds and library versions are
pinned, and determinism is tested on Linux/x86. Exact numbers may shift on a
different machine. The single `seed` flag drives `PYTHONHASHSEED`, torch, and
numpy. Every output CSV is stamped with the config hash and the library
versions. This caveat belongs in the README and should not be silently
dropped.

## A run that executes is not a correct metric

The subtle metric definitions are where bugs hide and still produce
plausible-looking output:

- The disparate-impact direction convention. Numerator, denominator, and what
  counts as the favored outcome. Fixed in the docstring and a test.
- Multi-class and multi-valued sensitive attributes. `race` is not binary.
  No implicit binary assumption anywhere.
- Calibration-error edge cases. Empty bins and single-class groups are tested,
  not just executed.

Every metric ships with a unit test, a hypothesis property test, and a
reference-oracle test against Fairlearn or scikit-learn on a fixed seed. The
output is also checked against a known sanity value: Adult DP diff around 0.15,
COMPAS DI well below 0.8. A result far from the sanity value is a signal to
investigate, not to commit.

## What this means for the card

The renderer reads every footprint number as simulated under a stated budget,
every fairness number as reproducible from a pinned config on Linux/x86, and
every per-group result as derived from the raw sensitive columns the loader
returned. The card states the budget it was sized against and the platform the
numbers are deterministic on. It does not imply a hardware deployment result.
