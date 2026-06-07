"""Renderer tests: HTML from the schema, PDF skipped without weasyprint."""

from __future__ import annotations

import pytest

from tinyaudit.card.render import render_html
from tinyaudit.card.schema import (
    AuditCard,
    FairnessBlock,
    MetricValue,
    UncertaintyBlock,
    XaiBlock,
)
from tinyaudit.profile.footprint import Footprint


def _footprint() -> Footprint:
    return Footprint(
        n_params=1234,
        peak_ram_bytes=56789,
        flops=987654,
        wall_clock_s_per_sample=0.00042,
    )


def _fairness() -> FairnessBlock:
    return FairnessBlock(
        metrics=[
            MetricValue(name="Demographic parity", value=0.1234, band="green"),
            MetricValue(name="Equalized odds", value=0.4567, band="amber"),
            MetricValue(name="Disparate impact", value=0.6500, band="red"),
        ]
    )


def _card(
    *,
    uncertainty: UncertaintyBlock | None = None,
    explainability: XaiBlock | None = None,
) -> AuditCard:
    return AuditCard(
        dataset="UCI-Adult",
        model="LogisticRegression",
        compression="int8",
        footprint=_footprint(),
        fairness=_fairness(),
        uncertainty=uncertainty,
        explainability=explainability,
        manifest_path="experiments/results/run-xyz/manifest.json",
    )


def test_render_html_returns_str_with_core_content() -> None:
    html = render_html(_card())

    assert isinstance(html, str)
    assert html.strip().startswith("<!DOCTYPE html>")
    # Dataset / model / compression surfaced.
    assert "UCI-Adult" in html
    assert "LogisticRegression" in html
    assert "int8" in html
    # Footprint numbers surfaced (thousands-formatted).
    assert "1,234" in html
    # Metric names and formatted values surfaced.
    assert "Demographic parity" in html
    assert "0.1234" in html
    assert "0.4567" in html
    assert "0.6500" in html
    # Traffic-light CSS classes driven by band.
    assert "band-green" in html
    assert "band-amber" in html
    assert "band-red" in html
    # Manifest path in footer.
    assert "experiments/results/run-xyz/manifest.json" in html


def test_optional_sections_absent_when_none() -> None:
    html = render_html(_card(uncertainty=None, explainability=None))

    assert "Uncertainty-aware fairness" not in html
    assert "Explainability" not in html


def test_optional_sections_present_when_populated() -> None:
    uncertainty = UncertaintyBlock(
        metrics=[
            MetricValue(name="Group entropy gap", value=0.2200, band="amber"),
            MetricValue(name="Selective fairness AUC", value=0.8800, band="green"),
        ]
    )
    explainability = XaiBlock(
        top_features=["age", "education", "hours_per_week"],
        importance_flips=["capital_gain"],
    )
    html = render_html(_card(uncertainty=uncertainty, explainability=explainability))

    assert "Uncertainty-aware fairness" in html
    assert "Group entropy gap" in html
    assert "0.2200" in html
    assert "Explainability" in html
    assert "age" in html
    assert "education" in html
    assert "capital_gain" in html


def test_html_round_trips_through_card_method() -> None:
    card = _card()
    assert card.to_html() == render_html(card)


def test_render_pdf_basic() -> None:
    weasyprint = pytest.importorskip("weasyprint")
    assert weasyprint is not None

    import tempfile
    from pathlib import Path

    card = _card()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "card.pdf"
        card.to_pdf(str(out))
        assert out.exists()
        assert out.stat().st_size > 0
