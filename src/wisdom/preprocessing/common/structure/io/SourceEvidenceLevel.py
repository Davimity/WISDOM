"""SourceEvidenceLevel vocabulary; scientific values are unchanged."""

from enum import Enum


class SourceEvidenceLevel(str, Enum):
    """Distinguish original bytes, verified republication and absent source evidence."""

    EXACT = "exact"
    EQUIVALENT = "equivalent"
    UNAVAILABLE = "unavailable"
