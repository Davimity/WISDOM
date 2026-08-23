"""Normalized chain domain value."""

from dataclasses import dataclass

from wisdom.preprocessing.structure.dataclasses.Residue import Residue


@dataclass(frozen=True, slots=True)
class Chain:
    """Represent one selected structural chain and exclusively own its residues.

    Attributes:
        id: Author/model chain identifier preserved from the coordinate file.
        residues: Retained residues in deterministic source-model order.
    """

    id       : str
    residues : tuple[Residue, ...]
