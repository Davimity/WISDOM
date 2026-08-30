"""Shared Gemmi representation of one deposited protein structure."""

import gemmi
import hashlib
import math

from pathlib import Path
from wisdom.utils.structure.BiologicalAssembly import BiologicalAssembly


class ProteinStructure:
    """Own one coordinate deposition and provide its shared structural operations."""

    def __init__(self, path: Path) -> None:
        """Read one PDB or mmCIF coordinate file and prepare entity metadata.

        Args:
            path: Local coordinate file supported by Gemmi. Compressed files are accepted
                when Gemmi can read them directly.

        Raises:
            OSError: If the coordinate file cannot be opened.
            RuntimeError: If Gemmi cannot parse the coordinate syntax.
            ValueError: If the file contains no coordinate model.
        """
        self.path      = path
        self.structure = gemmi.read_structure(str(path))

        if not self.structure:
            raise ValueError(f"structure has no coordinate model: {path}")

        self.structure.setup_entities()
        self.structure.assign_label_seq_id()

        # Experimental metadata belongs to the deposited structure itself. PDB files may not
        # expose every value; unavailable attributes remain explicit instead of being guessed.

        resolution = float(self.structure.resolution)
        method     = "unavailable"
        dates      : tuple[str | None, ...] = ()

        name = path.name.lower()
        if name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
            block  = gemmi.cif.read(str(path)).sole_block()
            method = str(block.find_value("_exptl.method") or "unavailable").strip("'\"")
            dates  = (
                block.find_value("_pdbx_database_status.recvd_initial_deposition_date"),
                block.find_value("_database_PDB_rev.date_original"),
                block.find_value("_pdbx_audit_revision_history.revision_date"),
            )

        self.resolution          = (
            resolution if math.isfinite(resolution) and resolution > 0.0 else None
        )
        self.release_year        = next(
            (
                int(value[:4])
                for value in dates
                if value and len(value) >= 4 and value[:4].isdigit()
            ),
            None,
        )
        self.experimental_method = method or "unavailable"

    def assembly(self, assembly_id: str) -> BiologicalAssembly:
        """Generate one named biological assembly.

        Gemmi applies the transformations declared in the mmCIF assembly tables and names
        copied chains with its deterministic ``Dup`` policy.

        Args:
            assembly_id: Exact assembly name declared by the coordinate deposition.

        Returns:
            Cohesive assembly object exposing protein copies and DNA atoms.

        Raises:
            ValueError: If the requested assembly is absent.
        """
        assembly = next(
            (
                candidate
                for candidate in self.structure.assemblies
                if str(candidate.name) == assembly_id
            ),
            None,
        )
        if assembly is None:
            raise ValueError(f"assembly {assembly_id!r} is absent from {self.path.name}")

        model = gemmi.make_assembly(
            assembly,
            self.structure[0],
            gemmi.HowToNameCopiedChain.Dup,
        )
        return BiologicalAssembly(self.structure, model, assembly_id)

    def sequence(self, chain: gemmi.Chain) -> str:
        """Return the complete entity sequence of one deposited protein chain.

        Args:
            chain: Deposited chain belonging to this structure's first model.

        Returns:
            Complete uppercase one-letter entity sequence.

        Raises:
            ValueError: If the chain has no complete entity sequence.
        """
        entity = self.structure.get_entity_of(chain.get_polymer())
        if entity is None or not entity.full_sequence:
            raise ValueError(f"chain {chain.name!r} has no complete entity sequence")
        return gemmi.one_letter_code(entity.full_sequence).upper()

    def sha256(self) -> str:
        """Return the SHA-256 digest of the exact coordinate-file bytes.

        Returns:
            Lowercase hexadecimal SHA-256 computed without loading the entire file into memory.

        Raises:
            OSError: If the source file cannot be read.
        """
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
