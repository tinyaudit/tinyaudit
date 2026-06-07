"""Uncertainty estimation and uncertainty-aware fairness metrics."""

from __future__ import annotations

from tinyaudit.uncertainty.early_exit import EarlyExitEnsemble
from tinyaudit.uncertainty.ensemble import DeepEnsemble
from tinyaudit.uncertainty.mc_dropout import MCDropout, aggregate_samples
from tinyaudit.uncertainty.metrics import (
    ece_per_group,
    group_predictive_entropy,
    selective_fairness_auc,
)
from tinyaudit.uncertainty.types import UncertaintyEstimator, UncertaintyOutput

__all__ = [
    "UncertaintyOutput",
    "UncertaintyEstimator",
    "MCDropout",
    "DeepEnsemble",
    "EarlyExitEnsemble",
    "aggregate_samples",
    "group_predictive_entropy",
    "ece_per_group",
    "selective_fairness_auc",
]
