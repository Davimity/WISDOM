"""Version-one canonical heavy-atom nomenclature; never fitted to a dataset."""

from wisdom.preprocessing.common.structure.chemical_data import STANDARD_RESIDUE_BONDS

# Named paths follow the IUPAC/IUB protein torsion convention. A terminal symmetric
# atom keeps its exact name; no protonation-dependent equivalence is assumed.

CHI_PATHS = {
    "ARG": ("N CA CB CG", "CA CB CG CD", "CB CG CD NE", "CG CD NE CZ"),
    "ASN": ("N CA CB CG", "CA CB CG OD1"),
    "ASP": ("N CA CB CG", "CA CB CG OD1"),
    "CYS": ("N CA CB SG",),
    "GLN": ("N CA CB CG", "CA CB CG CD", "CB CG CD OE1"),
    "GLU": ("N CA CB CG", "CA CB CG CD", "CB CG CD OE1"),
    "HIS": ("N CA CB CG", "CA CB CG ND1"),
    "ILE": ("N CA CB CG1", "CA CB CG1 CD1"),
    "LEU": ("N CA CB CG", "CA CB CG CD1"),
    "LYS": ("N CA CB CG", "CA CB CG CD", "CB CG CD CE", "CG CD CE NZ"),
    "MET": ("N CA CB CG", "CA CB CG SD", "CB CG SD CE"),
    "PHE": ("N CA CB CG", "CA CB CG CD1"),
    "PRO": ("N CA CB CG", "CA CB CG CD"),
    "SER": ("N CA CB OG",),
    "THR": ("N CA CB OG1",),
    "TRP": ("N CA CB CG", "CA CB CG CD1"),
    "TYR": ("N CA CB CG", "CA CB CG CD1"),
    "VAL": ("N CA CB CG1",),
}

EXACT_ATOMS = tuple(sorted({(residue, name) for residue, bonds in STANDARD_RESIDUE_BONDS.items()
                           for name in {"N", "CA", "C", "O", "OXT"}
                           | {str(bond[i]) for bond in bonds for i in (0, 1)}}))
ATOM_NAMES = tuple(sorted({name for _, name in EXACT_ATOMS}))
EXACT_IDS = {pair: i + 1 for i, pair in enumerate(EXACT_ATOMS)}
NAME_IDS = {name: i + 1 for i, name in enumerate(ATOM_NAMES)}
POSITIONS = ("UNK", "backbone_N", "backbone_CA", "backbone_C", "backbone_O",
             "terminal_OXT", "sidechain_beta", "sidechain_gamma", "sidechain_delta",
             "sidechain_epsilon", "sidechain_zeta", "sidechain_eta")
POSITION_IDS = {"N": 1, "CA": 2, "C": 3, "O": 4, "OXT": 5}
SIDECHAIN_POSITION_IDS = {"B": 6, "G": 7, "D": 8, "E": 9, "Z": 10, "H": 11}

# Bit positions identify overlapping motifs, not mutually exclusive chemical types.
FUNCTIONAL_GROUPS = (
    "backbone_amide", "carbonyl", "carboxyl", "guanidino", "amine", "hydroxyl",
    "thiol", "thioether", "aromatic", "imidazole", "disulfide",
)
FUNCTIONAL_ATOMS = {
    "carboxyl": {"ASP": "CG OD1 OD2", "GLU": "CD OE1 OE2"},
    "guanidino": {"ARG": "NE CZ NH1 NH2"},
    "amine": {"LYS": "NZ"},
    "hydroxyl": {"SER": "OG", "THR": "OG1", "TYR": "OH"},
    "thiol": {"CYS": "SG"},
    "thioether": {"MET": "SD"},
    "imidazole": {"HIS": "CG ND1 CD2 CE1 NE2"},
    "aromatic": {"HIS": "CG ND1 CD2 CE1 NE2", "PHE": "CG CD1 CD2 CE1 CE2 CZ",
                 "TYR": "CG CD1 CD2 CE1 CE2 CZ", "TRP": "CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2"},
}
