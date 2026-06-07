"""Dataset loaders.

Each loader is deterministic and owns its preprocessing and schema validation.
"""

from __future__ import annotations

from tinyaudit.data.adult import load_adult
from tinyaudit.data.compas import load_compas
from tinyaudit.data.folktables import load_folktables

__all__ = ["load_adult", "load_folktables", "load_compas"]
