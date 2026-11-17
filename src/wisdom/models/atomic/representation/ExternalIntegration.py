"""ExternalIntegration vocabulary; scientific values are unchanged."""

from enum import Enum


class ExternalIntegration(str, Enum):
    """Separate static initialization from contextual information and residue-table removal."""

    NONE = "none"
    STATIC = "static"
    CONTEXT = "context"
    BOTH = "both"
    REPLACE = "replace"
