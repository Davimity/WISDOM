"""Domain and provenance dataclasses used by preprocessing."""

from preprocess.dataclasses.Atom import Atom
from preprocess.dataclasses.Chain import Chain
from preprocess.dataclasses.Protein import Protein
from preprocess.dataclasses.ProteinMetadata import ProteinMetadata
from preprocess.dataclasses.Residue import Residue
from preprocess.dataclasses.StructureSource import StructureSource

__all__ = ["Atom", "Chain", "Protein", "ProteinMetadata", "Residue", "StructureSource"]
