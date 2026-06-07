"""Card rendering: pydantic schema -> Jinja2 -> HTML -> PDF.

The HTML path has no heavy dependencies. The PDF path imports WeasyPrint
lazily so a fairness-only HTML render works without the full stack.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from tinyaudit.card.schema import AuditCard

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "audit_card.html.j2"


def render_html(card: AuditCard) -> str:
    """Render ``card`` to a standalone, self-contained HTML string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2", "html.j2"], default=True),
    )
    template = env.get_template(_TEMPLATE_NAME)
    # jinja2 is untyped in this environment, so ``render`` is ``Any``;
    # coerce to a concrete ``str`` to stay strict-clean.
    return str(template.render(card=card))


def render_pdf(card: AuditCard, path: str) -> None:
    """Render ``card`` to a one-page PDF at ``path``.

    WeasyPrint is imported here, not at module top, so the HTML path works
    without it. A missing install raises a clear, actionable error.
    """
    try:
        import weasyprint
    except ImportError as exc:  # pragma: no cover - exercised only without weasyprint
        raise RuntimeError("PDF rendering requires weasyprint; install the full stack") from exc

    html = render_html(card)
    weasyprint.HTML(string=html).write_pdf(target=path)
