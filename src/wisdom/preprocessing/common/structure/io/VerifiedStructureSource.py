"""Recover exact or equivalent published sources without ambiguous structural alignment."""

import gzip
import gemmi
import numpy as np

from typing import Any
from pathlib import Path
from dataclasses import replace
from collections.abc import Mapping
from wisdom.utils.structure.models.Protein import Protein
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.common.structure.io.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.io.StructureSource import StructureSource
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive


class VerifiedStructureSource:
    """Bind a source hierarchy to every base atom before exposing author identities."""

    @staticmethod
    def selection(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
        """Read universal assembly fields without interpreting any task evidence or labels.

        Args:
            metadata: Member provenance, possibly containing a historical nested structural record.

        Returns:
            Only assembly_id, protein_chain and protein_copy when an unambiguous complete
            selection exists. All functional keys are ignored; their container name is irrelevant.

        Raises:
            ValueError: Two structural provenance records declare contradictory selections.
        """
        fields = ("assembly_id", "protein_chain", "protein_copy")
        candidates = [metadata, *(value for value in metadata.values()
                                  if isinstance(value, Mapping))]
        selections = [{key: candidate[key] for key in fields}
                      for candidate in candidates if all(key in candidate for key in fields)]
        if not selections:
            return None
        if any(candidate != selections[0] for candidate in selections[1:]):
            raise ValueError("member declares conflicting structural assembly/copy provenance")
        return selections[0]

    @staticmethod
    def recover(
        base: Path, published: Path | None = None, published_hash: str | None = None,
        quality: bool = True,
        assembly: Mapping[str, Any] | None = None,
    ) -> tuple[Protein | None, Path | None, str]:
        """Recover a portable source using selection plus known centroid normalization only.

        Args:
            base: Immutable schema-3 archive, including filters, source digest and chain selection.
            published: Optional native member `source_structure` asset, not a live PDB revision.
            published_hash: Optional immutable asset checksum, with or without `sha256:` prefix.
            quality: Read experimental quality only for exact original bytes. Equivalent sources
                provide author identity but never invent original occupancy/B/altLoc observations.
            assembly: Optional universal assembly_id/protein_chain/protein_copy provenance. These
                explicit operations may be tried only if the deposited frame fails exact joining.

        Returns:
            Verified hierarchy, selected file and quality level: `exact`, `equivalent`, or
            `unavailable`. Equivalent coordinates must agree within 2e-5 Å after the original
            filtering/centering policy; they are canonicalized to base float32 for exact joins.
            No rotation, sequence-only alignment or PDB-ID equivalence is attempted.

        Raises:
            ValueError: A supplied published asset has wrong bytes or atom/parent correspondence.
        """
        metadata = ProteinArchive(PreprocessConfig()).read_metadata(base)
        if metadata is None:
            raise ValueError("source recovery requires readable universal metadata")
        historical = Path(str(metadata.get("source_path", "")))
        exact = historical.is_file() and ProteinArchive.sha256(historical) == metadata[
            "source_hash"]
        path = historical if exact else published
        if path is None:
            return None, None, "unavailable"
        digest = ProteinArchive.sha256(path)
        if path == published and published_hash and digest != published_hash.removeprefix(
            "sha256:"
        ):
            raise ValueError("published source_structure checksum mismatch")
        exact = digest == metadata["source_hash"]
        suffixes = set(path.suffixes)
        source = StructureSource(str(metadata["source_identifier"]), str(metadata["protein_id"]),
            tuple(metadata["selected_chains"]), path, digest,
            "mmcif" if suffixes & {".cif", ".mmcif"} else str(metadata["source_format"]), True)
        payload = path.read_bytes()
        if payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        raw = gemmi.read_structure_string(payload)
        source = replace(source, format="mmcif" if raw.input_format == gemmi.CoorFormat.Mmcif
                         else "pdb")
        deposition = (ProteinStructure(path, mmcif_bytes=payload)
                      if source.format == "mmcif" else None)
        raw.setup_entities()
        reader = ProteinReader(PreprocessConfig(**metadata["config"]), quality and exact)
        protein, provenance = reader.read(source, raw, payload)
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in ("atom_positions", "atomic_numbers", "atom_names",
                "residue_names", "residue_indices", "chain_indices")}
        if not VerifiedStructureSource.matches(protein, arrays, metadata["selected_chains"]) \
                and assembly and deposition is not None:
            _, copy = deposition.assembly(str(assembly["assembly_id"])).protein_copy(
                str(assembly["protein_chain"]), int(assembly["protein_copy"]))
            selected = deposition.structure.clone()
            while len(selected):
                del selected[0]
            model = gemmi.Model(1)
            model.add_chain(copy.clone())
            selected.add_model(model)
            protein, provenance = reader.read(source, selected, payload)
        if not VerifiedStructureSource.matches(protein, arrays, metadata["selected_chains"]):
            raise ValueError("published source_structure mismatches base atom identity/geometry")
        if not np.allclose(provenance.coordinate_origin, metadata["coordinate_origin"],
                           atol=2e-5, rtol=0):
            raise ValueError("published source_structure uses an unverified coordinate frame")

        # Float32 quantization is the only authorized numerical transformation. Parent ownership
        # remains unchanged and no atom is matched by approximate spatial proximity.

        chains = tuple(replace(chain, residues=tuple(replace(residue, atoms=tuple(
            replace(atom, position=(float(arrays["atom_positions"][atom.index, 0]),
                float(arrays["atom_positions"][atom.index, 1]),
                float(arrays["atom_positions"][atom.index, 2])))
            for atom in residue.atoms)) for residue in chain.residues)) for chain in protein.chains)
        return replace(protein, chains=chains), path, "exact" if exact else "equivalent"

    @staticmethod
    def matches(
        protein: Protein, arrays: Mapping[str, np.ndarray], expected_chains: list[str],
    ) -> bool:
        """Compare the entire ordered identity hierarchy and known normalized coordinate frame.

        Args:
            protein: Domain hierarchy from exact bytes or a declared Gemmi assembly operation.
            arrays: Base names, elements, positions and ordinal parent indices.
            expected_chains: Actual chain order retained by the original reader.

        Returns:
            True only for a complete unambiguous join within 2e-5 Å float32 tolerance. No nearest
            atom assignment, rigid fitting, residue-letter alignment or partial acceptance occurs.
        """
        atoms = [atom for chain in protein.chains for residue in chain.residues
                 for atom in residue.atoms]
        residues = [residue for chain in protein.chains for residue in chain.residues]
        positions = np.asarray([atom.position for atom in atoms], dtype=np.float64)
        return all((
            tuple(chain.id for chain in protein.chains) == tuple(expected_chains),
            np.array_equal([a.name for a in atoms], arrays["atom_names"]),
            np.array_equal([a.atomic_number for a in atoms], arrays["atomic_numbers"]),
            np.array_equal([r.name for r in residues for _ in r.atoms], arrays["residue_names"]),
            np.array_equal([i for i, r in enumerate(residues) for _ in r.atoms],
                           arrays["residue_indices"]),
            np.array_equal([i for i, c in enumerate(protein.chains)
                            for r in c.residues for _ in r.atoms], arrays["chain_indices"]),
            positions.shape == arrays["atom_positions"].shape,
        )) and bool(np.allclose(positions, arrays["atom_positions"], atol=2e-5, rtol=0))
