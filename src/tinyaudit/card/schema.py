"""The audit card data model.

Built from the run manifest, never hand-edited. This is the contract
between the pipeline and the renderer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from tinyaudit.profile.footprint import Footprint

Band = Literal["green", "amber", "red"]


class MetricValue(BaseModel):
    """One metric, its value, and its traffic-light band."""

    name: str
    value: float
    band: Band


class FairnessBlock(BaseModel):
    """Point-prediction fairness: DP, EO, DI."""

    metrics: list[MetricValue]
    per_group: dict[str, dict[str, float]] = {}


class UncertaintyBlock(BaseModel):
    """Uncertainty-aware fairness: group entropy, ECE per group, selective AUC."""

    metrics: list[MetricValue]
    per_group: dict[str, dict[str, float]] = {}


class XaiBlock(BaseModel):
    """Explainer summary and per-group importance."""

    top_features: list[str]
    per_group_importance: dict[str, dict[str, float]] = {}
    importance_flips: list[str] = []


class AuditCard(BaseModel):
    """One run, one page. ``uncertainty`` and ``explainability`` are optional
    so a fairness-only audit still produces a valid card."""

    dataset: str
    model: str
    compression: str | None = None
    footprint: Footprint
    fairness: FairnessBlock
    uncertainty: UncertaintyBlock | None = None
    explainability: XaiBlock | None = None
    manifest_path: str

    def to_html(self) -> str:
        """Render the card to a standalone HTML string."""
        from tinyaudit.card.render import render_html

        return render_html(self)

    def to_pdf(self, path: str) -> None:
        """Render the card to a one-page PDF at ``path``."""
        from tinyaudit.card.render import render_pdf

        render_pdf(self, path)
