"""ExternalModelKind vocabulary; scientific values are unchanged."""

from enum import Enum


class ExternalModelKind(str, Enum):
    """Limit this infrastructure to the three explicitly requested checkpoint families."""

    ESM2 = "esm2"
    ESMC = "esmc"
    SAPROT = "saprot"
