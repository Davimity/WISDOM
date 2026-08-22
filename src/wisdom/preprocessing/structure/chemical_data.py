"""Centralized static chemistry used by the structural preprocessor."""

from wisdom.preprocessing.structure.enums.AtomRole import AtomRole
from wisdom.preprocessing.structure.enums.BondSource import BondSource
from wisdom.preprocessing.structure.enums.BondType import BondType
from wisdom.preprocessing.structure.enums.Relation import Relation

# Canonical protein amino-acid residue names in a stable persisted order.
# Used to encode residue identity compactly (zero is reserved for unknown).
# This collection does not imply that non-canonical residues are invalid.
AMINO_ACIDS = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)

# Mapping from canonical residue name to its persisted integer identifier.
# Used in atom feature arrays and versioned through the preprocessing schema.
# The IDs are categorical labels, not biochemical similarity scores.
AMINO_ACID_TO_ID = {name: index + 1 for index, name in enumerate(AMINO_ACIDS)}

# Atom names treated as protein backbone atoms.
# Used only to assign a coarse atom role.
# This list does not define covalent connectivity.
BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

# Residue names treated as crystallographic water.
# Used only for filtering and coarse atom-role assignment.
# This does not describe solvent chemistry.
WATER_RESIDUES = frozenset({"HOH", "WAT", "DOD"})

# Chemical element symbols classified as metals by the filtering policy.
# Used as a stable fallback when Gemmi cannot classify an element.
# This set does not describe coordination chemistry or oxidation state.
METAL_ELEMENTS = frozenset(
    {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "FR",
        "BE",
        "MG",
        "CA",
        "SR",
        "BA",
        "RA",
        "SC",
        "TI",
        "V",
        "CR",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "Y",
        "ZR",
        "NB",
        "MO",
        "TC",
        "RU",
        "RH",
        "PD",
        "AG",
        "CD",
        "HF",
        "TA",
        "W",
        "RE",
        "OS",
        "IR",
        "PT",
        "AU",
        "HG",
        "AL",
        "GA",
        "IN",
        "TL",
        "SN",
        "PB",
        "BI",
        "LA",
        "CE",
        "PR",
        "ND",
        "PM",
        "SM",
        "EU",
        "GD",
        "TB",
        "DY",
        "HO",
        "ER",
        "TM",
        "YB",
        "LU",
        "AC",
        "TH",
        "PA",
        "U",
        "NP",
        "PU",
        "AM",
        "CM",
    }
)

# Coarse atom-role identifiers persisted as compact categorical features.
# Used to distinguish backbone, side-chain and broad chemical roles.
# These IDs are not learned embeddings or complete atom typing.
ATOM_ROLE_UNKNOWN    = AtomRole.UNKNOWN
ATOM_ROLE_BACKBONE   = AtomRole.BACKBONE
ATOM_ROLE_SIDECHAIN  = AtomRole.SIDECHAIN
ATOM_ROLE_HYDROGEN   = AtomRole.HYDROGEN
ATOM_ROLE_METAL      = AtomRole.METAL
ATOM_ROLE_WATER      = AtomRole.WATER
ATOM_ROLE_NONPOLYMER = AtomRole.NONPOLYMER

# Bond-type identifiers persisted on covalent relations.
# Used to retain known chemical semantics alongside numeric bond order.
# A type of NONE means the union edge is spatial-only, not an unknown bond.
BOND_TYPE_NONE         = BondType.NONE
BOND_TYPE_SINGLE       = BondType.SINGLE
BOND_TYPE_DOUBLE       = BondType.DOUBLE
BOND_TYPE_TRIPLE       = BondType.TRIPLE
BOND_TYPE_AROMATIC     = BondType.AROMATIC
BOND_TYPE_PEPTIDE      = BondType.PEPTIDE
BOND_TYPE_DISULFIDE    = BondType.DISULFIDE
BOND_TYPE_COORDINATION = BondType.COORDINATION

# Provenance identifiers for reconstructed covalent bonds.
# Used to audit the evidence and priority behind each chemical relation.
# They are confidence categories, not experimental measurement methods.
BOND_SOURCE_NONE       = BondSource.NONE
BOND_SOURCE_EXPLICIT   = BondSource.EXPLICIT
BOND_SOURCE_TEMPLATE   = BondSource.TEMPLATE
BOND_SOURCE_PEPTIDE    = BondSource.PEPTIDE
BOND_SOURCE_DISULFIDE = BondSource.DISULFIDE
BOND_SOURCE_GEOMETRIC = BondSource.GEOMETRIC

# Bit flags for the single persisted atomic topology.
# Used to distinguish radius-neighbor and covalent relationships on one pair.
# Sharing an edge does not make the two relation semantics interchangeable.
RELATION_SPATIAL  = Relation.SPATIAL
RELATION_COVALENT = Relation.COVALENT

# Covalent bonds common to every canonical amino-acid backbone.
# Used as the base of deterministic residue-template reconstruction.
# This list excludes peptide bonds between residues.
BACKBONE_BONDS = (
    ("N", "CA", BOND_TYPE_SINGLE, 1.0),
    ("CA", "C", BOND_TYPE_SINGLE, 1.0),
    ("C", "O", BOND_TYPE_DOUBLE, 2.0),
    ("C", "OXT", BOND_TYPE_SINGLE, 1.0),
)

# Side-chain covalent templates for canonical amino acids.
# Used only when both named heavy atoms exist in the normalized residue.
# Missing atoms are never invented and protonation/aromatic resonance is not inferred.
STANDARD_RESIDUE_BONDS = {
    "ALA": (("CA", "CB", BOND_TYPE_SINGLE, 1.0),),
    "ARG": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD", BOND_TYPE_SINGLE, 1.0),
        ("CD", "NE", BOND_TYPE_SINGLE, 1.0),
        ("NE", "CZ", BOND_TYPE_SINGLE, 1.0),
        ("CZ", "NH1", BOND_TYPE_DOUBLE, 2.0),
        ("CZ", "NH2", BOND_TYPE_SINGLE, 1.0),
    ),
    "ASN": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "OD1", BOND_TYPE_DOUBLE, 2.0),
        ("CG", "ND2", BOND_TYPE_SINGLE, 1.0),
    ),
    "ASP": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "OD1", BOND_TYPE_DOUBLE, 2.0),
        ("CG", "OD2", BOND_TYPE_SINGLE, 1.0),
    ),
    "CYS": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "SG", BOND_TYPE_SINGLE, 1.0),
    ),
    "GLN": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD", BOND_TYPE_SINGLE, 1.0),
        ("CD", "OE1", BOND_TYPE_DOUBLE, 2.0),
        ("CD", "NE2", BOND_TYPE_SINGLE, 1.0),
    ),
    "GLU": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD", BOND_TYPE_SINGLE, 1.0),
        ("CD", "OE1", BOND_TYPE_DOUBLE, 2.0),
        ("CD", "OE2", BOND_TYPE_SINGLE, 1.0),
    ),
    "GLY": (),
    "HIS": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "ND1", BOND_TYPE_AROMATIC, 1.5),
        ("ND1", "CE1", BOND_TYPE_AROMATIC, 1.5),
        ("CE1", "NE2", BOND_TYPE_AROMATIC, 1.5),
        ("NE2", "CD2", BOND_TYPE_AROMATIC, 1.5),
        ("CD2", "CG", BOND_TYPE_AROMATIC, 1.5),
    ),
    "ILE": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG1", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG2", BOND_TYPE_SINGLE, 1.0),
        ("CG1", "CD1", BOND_TYPE_SINGLE, 1.0),
    ),
    "LEU": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD1", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD2", BOND_TYPE_SINGLE, 1.0),
    ),
    "LYS": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD", BOND_TYPE_SINGLE, 1.0),
        ("CD", "CE", BOND_TYPE_SINGLE, 1.0),
        ("CE", "NZ", BOND_TYPE_SINGLE, 1.0),
    ),
    "MET": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "SD", BOND_TYPE_SINGLE, 1.0),
        ("SD", "CE", BOND_TYPE_SINGLE, 1.0),
    ),
    "PHE": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD1", BOND_TYPE_AROMATIC, 1.5),
        ("CD1", "CE1", BOND_TYPE_AROMATIC, 1.5),
        ("CE1", "CZ", BOND_TYPE_AROMATIC, 1.5),
        ("CZ", "CE2", BOND_TYPE_AROMATIC, 1.5),
        ("CE2", "CD2", BOND_TYPE_AROMATIC, 1.5),
        ("CD2", "CG", BOND_TYPE_AROMATIC, 1.5),
    ),
    "PRO": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD", BOND_TYPE_SINGLE, 1.0),
        ("CD", "N", BOND_TYPE_SINGLE, 1.0),
    ),
    "SER": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "OG", BOND_TYPE_SINGLE, 1.0),
    ),
    "THR": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "OG1", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG2", BOND_TYPE_SINGLE, 1.0),
    ),
    "TRP": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD1", BOND_TYPE_AROMATIC, 1.5),
        ("CD1", "NE1", BOND_TYPE_AROMATIC, 1.5),
        ("NE1", "CE2", BOND_TYPE_AROMATIC, 1.5),
        ("CE2", "CD2", BOND_TYPE_AROMATIC, 1.5),
        ("CD2", "CG", BOND_TYPE_AROMATIC, 1.5),
        ("CD2", "CE3", BOND_TYPE_AROMATIC, 1.5),
        ("CE3", "CZ3", BOND_TYPE_AROMATIC, 1.5),
        ("CZ3", "CH2", BOND_TYPE_AROMATIC, 1.5),
        ("CH2", "CZ2", BOND_TYPE_AROMATIC, 1.5),
        ("CZ2", "CE2", BOND_TYPE_AROMATIC, 1.5),
    ),
    "TYR": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG", BOND_TYPE_SINGLE, 1.0),
        ("CG", "CD1", BOND_TYPE_AROMATIC, 1.5),
        ("CD1", "CE1", BOND_TYPE_AROMATIC, 1.5),
        ("CE1", "CZ", BOND_TYPE_AROMATIC, 1.5),
        ("CZ", "CE2", BOND_TYPE_AROMATIC, 1.5),
        ("CE2", "CD2", BOND_TYPE_AROMATIC, 1.5),
        ("CD2", "CG", BOND_TYPE_AROMATIC, 1.5),
        ("CZ", "OH", BOND_TYPE_SINGLE, 1.0),
    ),
    "VAL": (
        ("CA", "CB", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG1", BOND_TYPE_SINGLE, 1.0),
        ("CB", "CG2", BOND_TYPE_SINGLE, 1.0),
    ),
}
