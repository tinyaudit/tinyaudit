"""Explainability (XAI) layer.

Three explainers that all return the same ``(n_samples, n_features)`` array
shape so the card renderer is uniform, plus the per-group helpers the audit
uses to flag importance flips across sensitive groups.

``occlusion_attributions`` and the per-group helpers are imported eagerly
because they depend only on numpy and the model protocol. ``shap_attributions``
and ``lime_attributions`` are loaded lazily on first attribute access so a
shap-/lime-free install can still use occlusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tinyaudit.xai.occlusion import (
    importance_flips,
    occlusion_attributions,
    per_group_importance,
)

if TYPE_CHECKING:  # pragma: no cover - import-time only for static checkers
    from tinyaudit.xai.lime_wrap import lime_attributions
    from tinyaudit.xai.shap_wrap import shap_attributions

__all__ = [
    "shap_attributions",
    "lime_attributions",
    "occlusion_attributions",
    "per_group_importance",
    "importance_flips",
]


def __getattr__(name: str) -> Any:
    if name == "shap_attributions":
        from tinyaudit.xai.shap_wrap import shap_attributions

        return shap_attributions
    if name == "lime_attributions":
        from tinyaudit.xai.lime_wrap import lime_attributions

        return lime_attributions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
