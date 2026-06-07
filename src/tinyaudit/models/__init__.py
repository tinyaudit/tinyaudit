"""Model adapters conforming to the :class:`AuditedModel` protocol.

``SklearnModel`` and the ``AuditedModel`` protocol are imported eagerly.
``TorchModel`` is loaded lazily on first attribute access so a sklearn-only
install does not pull in torch at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tinyaudit.models.base import AuditedModel
from tinyaudit.models.sklearn import SklearnModel

if TYPE_CHECKING:  # pragma: no cover - import-time only for static checkers
    from tinyaudit.models.torch import TorchModel

__all__ = ["AuditedModel", "SklearnModel", "TorchModel"]


def __getattr__(name: str) -> Any:
    if name == "TorchModel":
        from tinyaudit.models.torch import TorchModel

        return TorchModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
