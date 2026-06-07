"""Point-prediction fairness metrics."""

from __future__ import annotations

from tinyaudit.fairness._frames import MetricFrame
from tinyaudit.fairness.parity import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
)

__all__ = [
    "MetricFrame",
    "demographic_parity_difference",
    "disparate_impact_ratio",
    "equalized_odds_difference",
]
