"""TinyAudit: audit small models for fairness, explainability, and uncertainty."""

from tinyaudit.card.schema import AuditCard
from tinyaudit.pipeline import audit

__all__ = ["audit", "AuditCard"]
__version__ = "0.1.0"
