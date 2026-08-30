"""Gemmi-backed structure normalization."""

from __future__ import annotations

from dataclasses import replace

import gemmi
import numpy as np

from wisdom.utils.structure.models.Atom import Atom
from wisdom.utils.structure.models.Chain import Chain
from wisdom.utils.structure.models.Protein import Protein
from wisdom.utils.structure.models.Residue import Residue
from wisdom.utils.structure.enums.BondType import BondType
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.structure.StructureSource import StructureSource
from wisdom.utils.structure.enums.ConnectionType import ConnectionType
from wisdom.preprocessing.structure.chemical_data import METAL_ELEMENTS
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessingProvenance import PreprocessingProvenance


class ProteinReader:
    """Convert one resolved coordinate source into parser-independent domain objects.

    Gemmi remains an implementation boundary: this class consumes its structural model and emits
    only immutable WISDOM dataclasses plus separate provenance metadata.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """Bind structural selection and normalization policies.

        Args:
            config: Model, chain, entity filter, hydrogen, metal, water, and centering settings.
        """
        self.config = config

    def read(self, source: StructureSource) -> tuple[Protein, PreprocessingProvenance]:
        """Parse, filter, normalize, center, and detach one coordinate structure from Gemmi.

        Alternate conformations are reduced to one atom per name by occupancy and a deterministic
        altLoc tie-break. Retained atoms receive consecutive indices, coordinates are optionally
        transformed as ``x' = x - mean(x)``, and explicit source connections are rebound to the
        resulting immutable atoms.

        Args:
            source: Validated local/cached structure path, chain selector, format, and source hash.

        Returns:
            A normalized ``Protein`` hierarchy and separate processing provenance.

        Raises:
            ValueError: If Gemmi cannot parse the structure, the model/chains are invalid, a residue
                lacks a numeric sequence ID, coordinates are non-finite, or filtering removes every
                atom/heavy atom.
        """
        # The shared deposition object owns Gemmi parsing, entity setup, assemblies, sequences,
        # and experimental attributes. This reader only applies preprocessing-specific filters.
        try:
            structure = ProteinStructure(source.path).structure
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Gemmi could not parse {source.path}: {error}") from error
        if self.config.model_index >= len(structure):
            raise ValueError(
                f"model_index {self.config.model_index} does not exist; "
                f"structure has {len(structure)} models"
            )

        # Resolve model and chain selection before creating any normalized objects.
        structure.add_entity_types()

        model     = structure[self.config.model_index]
        requested = source.chains or tuple(self.config.chains)
        available = tuple(chain.name for chain in model)
        missing   = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"requested chains not found: {missing}; available: {list(available)}")

        selected = requested or available

        # Traverse the single ownership hierarchy while building addresses for source connections.
        chains     : list[Chain] = []
        atom_index : int         = 0

        address_to_index: dict[tuple[str, int, str, str], int] = {}
        for gemmi_chain in model:
            if gemmi_chain.name not in selected:
                continue
            residues: list[Residue] = []
            for gemmi_residue in gemmi_chain:
                # Entity and element policies are applied before alternate-conformation selection.
                is_water   = bool(gemmi_residue.is_water())
                is_polymer = gemmi_residue.entity_type == gemmi.EntityType.Polymer
                is_metal   = any(
                    atom.element.is_metal or atom.element.name in METAL_ELEMENTS
                    for atom in gemmi_residue
                )
                if is_water and not self.config.include_waters:
                    continue
                if is_metal and not self.config.include_metals:
                    continue
                if (
                    not is_water
                    and not is_metal
                    and not is_polymer
                    and not self.config.include_nonpolymer
                ):
                    continue
                if gemmi_residue.seqid.num is None:
                    raise ValueError(f"residue has no numeric sequence ID: {gemmi_residue.seqid}")

                # Keep one atom per name: occupancy first, then blank/A altLoc, then alphabetically.
                chosen: dict[str, tuple[tuple[float, bool, int], gemmi.Atom]] = {}
                for gemmi_atom in gemmi_residue:
                    element = gemmi_atom.element
                    if element.is_hydrogen and not self.config.include_hydrogens:
                        continue
                    if (
                        element.is_metal or element.name in METAL_ELEMENTS
                    ) and not self.config.include_metals:
                        continue
                    name   = gemmi_atom.name.strip()
                    altloc = self._clean_code(gemmi_atom.altloc)
                    rank   = (
                        float(gemmi_atom.occ),
                        altloc in ("", "A"),
                        -ord(altloc[:1] or " "),
                    )
                    if name not in chosen or rank > chosen[name][0]:
                        chosen[name] = (rank, gemmi_atom)
                if not chosen:
                    continue

                # Sorted atom names make indices independent from parser iteration subtleties.
                insertion_code : str        = self._clean_code(gemmi_residue.seqid.icode)
                atoms          : list[Atom] = []
                for name in sorted(chosen):
                    gemmi_atom = chosen[name][1]
                    position = (
                        float(gemmi_atom.pos.x),
                        float(gemmi_atom.pos.y),
                        float(gemmi_atom.pos.z),
                    )
                    if not np.isfinite(position).all():
                        raise ValueError(
                            f"non-finite coordinates at "
                            f"{gemmi_chain.name}/{gemmi_residue.seqid}/{name}"
                        )
                    atom = Atom(
                        index=atom_index,
                        name=name,
                        atomic_number=gemmi_atom.element.atomic_number,
                        position=position,
                        formal_charge=int(gemmi_atom.charge),
                    )
                    atoms.append(atom)
                    address_to_index[
                        (
                            gemmi_chain.name,
                            gemmi_residue.seqid.num,
                            insertion_code,
                            name,
                        )
                    ] = atom_index
                    atom_index += 1
                residues.append(
                    Residue(
                        name=gemmi_residue.name.strip().upper(),
                        number=gemmi_residue.seqid.num,
                        insertion_code=insertion_code,
                        is_polymer=is_polymer,
                        atoms=tuple(atoms),
                    )
                )
            if residues:
                chains.append(Chain(gemmi_chain.name, tuple(residues)))

        # The filtered representation must contain matter meaningful to structural learning.
        atoms = [atom for chain in chains for residue in chain.residues for atom in residue.atoms]
        if not atoms:
            raise ValueError("no atoms remain after filtering")
        if not any(atom.atomic_number > 1 for atom in atoms):
            raise ValueError("no heavy atom remains after filtering")

        # Center once in float64 and rebuild the immutable hierarchy without duplicating ownership.
        coordinates = np.asarray([atom.position for atom in atoms], dtype=np.float64)
        origin      = (
            coordinates.mean(axis=0) if self.config.center_coordinates else np.zeros(3)
        )
        centered = {
            atom.index: replace(
                atom,
                position=(
                    float(coordinates[row, 0] - origin[0]),
                    float(coordinates[row, 1] - origin[1]),
                    float(coordinates[row, 2] - origin[2]),
                ),
            )
            for row, atom in enumerate(atoms)
        }
        normalized_chains = tuple(
            Chain(
                chain.id,
                tuple(
                    replace(
                        residue,
                        atoms=tuple(centered[atom.index] for atom in residue.atoms),
                    )
                    for residue in chain.residues
                ),
            )
            for chain in chains
        )
        # A temporary index lookup binds source connections to the final centered Atom instances.
        normalized_atoms = {
            atom.index: atom
            for chain in normalized_chains
            for residue in chain.residues
            for atom in residue.atoms
        }
        protein = Protein(
            id=source.protein_id,
            chains=normalized_chains,
            explicit_connections=self._explicit_connections(
                structure, source, address_to_index, normalized_atoms
            ),
        )
        provenance = PreprocessingProvenance(
            source_identifier=source.identifier,
            source_path=str(source.path),
            source_hash=source.sha256,
            source_format=source.format,
            selected_chains=tuple(selected),
            coordinate_origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        )
        return protein, provenance

    def _explicit_connections(
        self,
        structure        : gemmi.Structure,
        source           : StructureSource,
        address_to_index : dict[tuple[str, int, str, str], int],
        atoms            : dict[int, Atom],
    ) -> tuple[tuple[Atom, Atom, ConnectionType, BondType], ...]:
        """Resolve source-declared connections onto final normalized atoms.

        Gemmi's generic connection list provides disulfide, metal, hydrogen, and covalent semantics.
        For PDBx/mmCIF, the method additionally reads ``_struct_conn.pdbx_value_order`` so explicit
        single, double, triple, and aromatic multiplicity is not collapsed. Atom addresses use
        ``(chain, residue number, insertion code, atom name)`` and unresolved/filtered partners are
        skipped.

        Args:
            structure: Parsed Gemmi structure containing source connection records.
            source: Resolved source whose format decides whether mmCIF value-order columns exist.
            address_to_index: Mapping from source atom addresses to normalized consecutive indices.
            atoms: Mapping from normalized indices to final centered immutable atoms.

        Returns:
            Deterministically sorted unique atom-reference connections with semantic enums.
        """
        connections: dict[tuple[int, int], tuple[ConnectionType, BondType]] = {}

        # Normalize Gemmi connections into one undirected pair dictionary.
        for connection in structure.connections:
            indices: list[int] = []
            for address in (connection.partner1, connection.partner2):
                if address.res_id.seqid.num is None:
                    break
                key = (
                    address.chain_name,
                    address.res_id.seqid.num,
                    self._clean_code(address.res_id.seqid.icode),
                    address.atom_name.strip(),
                )
                if key not in address_to_index:
                    break
                indices.append(address_to_index[key])
            if len(indices) != 2 or indices[0] == indices[1]:
                continue
            pair = tuple(sorted(indices))
            if connection.type == gemmi.ConnectionType.Disulf:
                value = (ConnectionType.DISULFIDE, BondType.DISULFIDE)
            elif connection.type == gemmi.ConnectionType.MetalC:
                value = (ConnectionType.METAL_COORDINATION, BondType.COORDINATION)
            elif connection.type == gemmi.ConnectionType.Hydrog:
                value = (ConnectionType.HYDROGEN_BOND, BondType.NONE)
            else:
                value = (ConnectionType.COVALENT, BondType.SINGLE)
            connections[(pair[0], pair[1])] = value

        # mmCIF value-order fields preserve multiplicity unavailable in generic connection typing.
        if source.format == "mmcif":
            block = gemmi.cif.read(str(source.path)).sole_block()
            columns = {
                name: list(block.find_values(f"_struct_conn.{name}"))
                for name in (
                    "conn_type_id",
                    "ptnr1_auth_asym_id",
                    "ptnr1_auth_seq_id",
                    "pdbx_ptnr1_PDB_ins_code",
                    "ptnr1_label_atom_id",
                    "ptnr2_auth_asym_id",
                    "ptnr2_auth_seq_id",
                    "pdbx_ptnr2_PDB_ins_code",
                    "ptnr2_label_atom_id",
                    "pdbx_value_order",
                )
            }
            orders = {
                "sing": BondType.SINGLE,
                "doub": BondType.DOUBLE,
                "trip": BondType.TRIPLE,
                "arom": BondType.AROMATIC,
            }
            for row, connection_name in enumerate(columns["conn_type_id"]):
                try:
                    left = (
                        columns["ptnr1_auth_asym_id"][row],
                        int(columns["ptnr1_auth_seq_id"][row]),
                        self._clean_code(columns["pdbx_ptnr1_PDB_ins_code"][row]),
                        columns["ptnr1_label_atom_id"][row],
                    )
                    right = (
                        columns["ptnr2_auth_asym_id"][row],
                        int(columns["ptnr2_auth_seq_id"][row]),
                        self._clean_code(columns["pdbx_ptnr2_PDB_ins_code"][row]),
                        columns["ptnr2_label_atom_id"][row],
                    )
                except (IndexError, ValueError):
                    continue
                if left not in address_to_index or right not in address_to_index:
                    continue
                pair = tuple(sorted((address_to_index[left], address_to_index[right])))
                order_name = (
                    columns["pdbx_value_order"][row].lower()
                    if row < len(columns["pdbx_value_order"])
                    else "sing"
                )
                bond_type       = orders.get(order_name, BondType.SINGLE)
                connection_type = ConnectionType.COVALENT
                if "disulf" in connection_name.lower():
                    connection_type, bond_type = ConnectionType.DISULFIDE, BondType.DISULFIDE
                elif "metal" in connection_name.lower():
                    connection_type, bond_type = (
                        ConnectionType.METAL_COORDINATION,
                        BondType.COORDINATION,
                    )
                connections[(pair[0], pair[1])] = (connection_type, bond_type)

        return tuple(
            (atoms[src], atoms[dst], connection_type, bond_type)
            for (src, dst), (connection_type, bond_type) in sorted(connections.items())
        )

    @staticmethod
    def _clean_code(value: str) -> str:
        """Canonicalize optional PDB/mmCIF character codes.

        Args:
            value: Raw insertion or alternate-location code supplied by Gemmi/mmCIF.

        Returns:
            A stripped unquoted code, or the empty string for NUL, blank, and ``?`` missing markers.
        """
        cleaned = value.strip().strip("'\"")
        return "" if cleaned in ("\x00", "", "?") else cleaned
