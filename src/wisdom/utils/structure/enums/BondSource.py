"""Evidence used to establish a covalent bond."""

from enum import IntEnum


class BondSource(IntEnum):
    """Encode the highest-priority evidence used to establish a persisted bond."""

    NONE = 0
    EXPLICIT = 1
    TEMPLATE = 2
    PEPTIDE = 3
    DISULFIDE = 4
    GEOMETRIC = 5
