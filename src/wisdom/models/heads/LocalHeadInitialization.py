"""LocalHeadInitialization vocabulary; scientific values are unchanged."""

from enum import Enum


class LocalHeadInitialization(str, Enum):
    """Independent conditioned classifier or explicit copy of the shared local classifier."""

    INDEPENDENT = "independent"
    COPY = "copy"
