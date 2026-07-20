"""Cost measurement.

``Footprint`` is the frozen contract. ``profile_model`` is implemented in
the first build wave; the field names and types here do not change.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
from pydantic import BaseModel

from tinyaudit.models.base import AuditedModel


class Footprint(BaseModel):
    """Resource cost of running a model on one dataset.

    These numbers back the "sized for microcontroller-class hardware" claim,
    so they are recorded for every model and every compression setting,
    including the uncertainty estimators themselves.
    """

    n_params: int
    peak_ram_bytes: int
    flops: int
    wall_clock_s_per_sample: float


def _as_array(X: Any) -> np.ndarray:
    """Coerce a numpy array or a pandas frame/series into a numpy array.

    Pandas is detected duck-typed via ``.to_numpy`` so this stays import-free.
    """
    if isinstance(X, np.ndarray):
        return X
    to_numpy = getattr(X, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    return np.asarray(X)


def profile_model(model: AuditedModel, X: np.ndarray) -> Footprint:
    """Measure ``model`` running over ``X``.

    - ``n_params`` is read straight off the model.
    - ``peak_ram_bytes`` wraps one ``predict`` pass in ``tracemalloc`` and
      reports the traced peak in bytes.
    - ``wall_clock_s_per_sample`` times one ``predict`` pass with
      ``time.perf_counter`` divided by the sample count (0 when empty).
    - ``flops`` is an analytic proxy for the scikit-learn vertical slice:
      ``n_params * n_samples`` approximates one linear pass over the inputs.
      The exact thop/fvcore torch path is added in a later wave once the
      torch adapter lands; thop is not installed here.

    ``X`` may be a numpy array or a pandas frame/series; it is coerced.
    """
    arr = _as_array(X)
    n_samples = int(arr.shape[0]) if arr.ndim > 0 else 0

    n_params = int(model.n_params)

    # Peak RAM of a single predict pass. Nesting-safe: if a caller is already
    # tracing (e.g. experiments/run_audit_footprint.py profiling the whole
    # pipeline), do not start/stop tracemalloc here, since stopping it would
    # discard the caller's measurement. Measure a delta against the outer peak
    # instead.
    outer = tracemalloc.is_tracing()
    if not outer:
        tracemalloc.start()
    try:
        _, before = tracemalloc.get_traced_memory()
        model.predict(arr)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not outer:
            tracemalloc.stop()
    peak_ram_bytes = int(peak - before) if outer else int(peak)

    # Wall clock of a single predict pass, per sample.
    if n_samples == 0:
        wall_clock_s_per_sample = 0.0
    else:
        start = time.perf_counter()
        model.predict(arr)
        elapsed = time.perf_counter() - start
        wall_clock_s_per_sample = elapsed / n_samples

    # Analytic FLOPs proxy: one linear pass over the inputs. The thop/fvcore
    # torch path replaces this for the torch slice in a later wave.
    flops = int(n_params * n_samples)

    return Footprint(
        n_params=n_params,
        peak_ram_bytes=peak_ram_bytes,
        flops=flops,
        wall_clock_s_per_sample=wall_clock_s_per_sample,
    )
