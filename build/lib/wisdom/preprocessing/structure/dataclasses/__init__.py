"""Domain and provenance dataclasses used by preprocessing."""

from wisdom.preprocessing.structure.dataclasses.Atom import Atom
from wisdom.preprocessing.structure.dataclasses.Chain import Chain
from wisdom.preprocessing.structure.dataclasses.Protein import Protein
from wisdom.preprocessing.structure.dataclasses.ProteinMetadata import ProteinMetadata
from wisdom.preprocessing.structure.dataclasses.Residue import Residue
from wisdom.preprocessing.structure.dataclasses.StructureSource import StructureSource

__all__ = ["Atom", "Chain", "Protein", "ProteinMetadata", "Residue", "StructureSource"]
