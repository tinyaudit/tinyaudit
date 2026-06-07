"""Model compression: int8 quantization and magnitude pruning."""

from __future__ import annotations

from tinyaudit.compress.prune import magnitude_prune
from tinyaudit.compress.quantize import quantize_int8

__all__ = ["quantize_int8", "magnitude_prune"]
