"""DiffusionReadout vocabulary; scientific values are unchanged."""

from enum import Enum


class DiffusionReadout(str, Enum):
    """Distinguish additional scales from a capacity-matched duplicated-scale control."""

    SINGLE = "single"
    CONCAT = "concat"
    MIXTURE = "mixture"
    DUPLICATE = "duplicate"
