"""Central RNG seeding for reproducible audit runs.

A single ``seed`` flag has to make an entire ``audit()`` run reproducible.
:func:`seed_everything` is the one place that seeds every global RNG the
pipeline touches -- ``random``, ``numpy``, and (when importable) ``torch`` --
plus ``PYTHONHASHSEED`` for child processes. Estimators that take an explicit
``seed=`` argument (e.g. :class:`~tinyaudit.uncertainty.ensemble.DeepEnsemble`)
keep threading it; this function covers everything that reads the *global* RNG
state instead.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed every global RNG the pipeline relies on from one ``seed``.

    This sets, from the single ``seed``:

    * ``os.environ["PYTHONHASHSEED"]`` -- note this only affects the hash
      randomization of *child* processes / interpreters started after this
      point. The already-running interpreter fixed its hash seed at startup, so
      setting it here does not retroactively change this process; it is still
      worth setting so any subprocess-based reproduction is deterministic.
    * ``random.seed`` and ``np.random.seed`` -- the stdlib and NumPy legacy
      global RNGs.
    * ``torch.manual_seed`` / ``torch.cuda.manual_seed_all`` -- only if torch is
      importable. torch is imported lazily inside a ``try/except ImportError``
      so this function works unchanged when torch is absent. The safe cuDNN
      determinism flags are set (guarded so they never raise);
      ``torch.use_deterministic_algorithms`` is deliberately *not* forced, since
      it raises on ops without a deterministic implementation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001 - never let a determinism flag break a run
        pass
