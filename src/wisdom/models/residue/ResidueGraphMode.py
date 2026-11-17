"""Closed scientific controls for physical residue communication."""

from enum import Enum


class ResidueGraphMode(str, Enum):
    """Separate learned per-node capacity, null messages, sequence and spatial information."""

    POINTWISE = "pointwise"
    NULL = "null"
    SEQUENCE = "sequence"
    CONTACT = "contact"
    HYBRID = "hybrid"
