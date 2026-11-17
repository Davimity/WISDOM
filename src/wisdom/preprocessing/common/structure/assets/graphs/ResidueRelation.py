"""ResidueRelation vocabulary; scientific values are unchanged."""

from enum import IntFlag


class ResidueRelation(IntFlag):
    """Retain all simultaneously true residue-edge semantics in a compact bit mask."""

    PEPTIDE = 1
    SPATIAL = 2
    DISULFIDE = 4
    OTHER_COVALENT = 8
