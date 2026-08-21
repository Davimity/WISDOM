"""Persisted chemical bond types."""

from enum import IntEnum


class BondType(IntEnum):
    """Encode closed chemical bond categories persisted on covalent graph relations."""

    NONE = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    PEPTIDE = 5
    DISULFIDE = 6
    COORDINATION = 7

    @property
    def order(self) -> float:
        """Map the semantic category to its conventional scalar bond order.

        Returns:
            ``1``, ``2``, ``3``, or ``1.5`` for ordinary/aromatic bonds. Peptide and disulfide
            categories have order one. ``NONE`` and ``COORDINATION`` return zero because WISDOM does
            not assign them a conventional covalent multiplicity.
        """
        return {
            BondType.SINGLE: 1.0,
            BondType.DOUBLE: 2.0,
            BondType.TRIPLE: 3.0,
            BondType.AROMATIC: 1.5,
            BondType.PEPTIDE: 1.0,
            BondType.DISULFIDE: 1.0,
        }.get(self, 0.0)
