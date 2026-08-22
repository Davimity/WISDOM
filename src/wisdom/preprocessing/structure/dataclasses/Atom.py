"""Normalized atom domain value."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Atom:
    """Represent one parser-independent atom in normalized Cartesian space.

    Attributes:
        index: Consecutive zero-based index in hierarchy traversal order.
        name: Structure atom name after surrounding whitespace removal.
        atomic_number: Chemical element atomic number supplied by Gemmi.
        position: Cartesian coordinates ``(x, y, z)`` in ångströms, optionally centered.
        formal_charge: Integral formal charge read from the source structure.
    """

    index         : int
    name          : str
    atomic_number : int
    position      : tuple[float, float, float]
    formal_charge : int                        = 0
