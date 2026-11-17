"""Bounded molecular communication topologies; never implicit recurrent inference."""

from enum import Enum


class ResidueTopology(str, Enum):
    """Name the single additional residue stage and its receiving representation."""

    DESCENDING = "descending"
    ROUNDTRIP = "roundtrip"
    PARALLEL = "parallel"
