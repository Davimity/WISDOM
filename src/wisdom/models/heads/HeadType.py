"""HeadType vocabulary; scientific values are unchanged."""

from enum import Enum


class HeadType(str, Enum):
    """Relationship between protein classification and surface localization heads."""

    SINGLE         = "single"
    DUAL           = "dual"
    GLOBAL_CONTEXT = "global_context"
    FILM           = "film"
