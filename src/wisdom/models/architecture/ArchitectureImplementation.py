"""ArchitectureImplementation vocabulary; scientific values are unchanged."""

from enum import Enum


class ArchitectureImplementation(str, Enum):
    """Preserve archived scalar/vector behavior or use the paired, audited V8 contract."""

    LEGACY     = "legacy"
    CONTROLLED = "controlled"
