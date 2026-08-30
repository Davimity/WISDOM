"""Persisted coarse atom roles."""

from enum import IntEnum


class AtomRole(IntEnum):
    """Encode mutually exclusive coarse atomic roles persisted as ``uint8`` categories."""

    UNKNOWN = 0
    BACKBONE = 1
    SIDECHAIN = 2
    HYDROGEN = 3
    METAL = 4
    WATER = 5
    NONPOLYMER = 6
