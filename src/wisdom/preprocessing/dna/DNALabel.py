"""Scientific labels used while curating the DNA-binding benchmark."""

from enum import Enum


class DNALabel(str, Enum):
    """Represent the four evidence states before a benchmark row is accepted."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN  = "unknown"
    CONFLICT = "conflict"
