"""Typed structural connection semantics read from PDB or PDBx/mmCIF."""

from enum import IntEnum


class ConnectionType(IntEnum):
    """Distinguish source connection semantics before covalent-graph reconstruction."""

    UNKNOWN = 0
    COVALENT = 1
    DISULFIDE = 2
    METAL_COORDINATION = 3
    HYDROGEN_BOND = 4
    OTHER = 5
