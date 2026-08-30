"""Project DNA-interface ground truth onto fixed WISDOM surface points."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.spatial import cKDTree

from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.dna.DNAAnnotationSink import DNAAnnotationSink


class DNAAnnotationTransform:
    """Create label sidecar arrays without modifying the universal base archive."""

    SCHEMA_VERSION = "1.3"

    def __init__(
        self,
        positive_gap     : float = 1.4,
        negative_gap     : float = 3.0,
        sensitivity_gaps : tuple[float, ...] = (1.0, 1.4, 2.0),
    ) -> None:
        """Configure the physical interface and ambiguity band in ångströms.

        For surface point ``s_i`` and DNA atom ``j`` with van der Waals radius ``r_j``, the stored
        separation is ``d_i = min_j (||s_i-x_j|| - r_j)``. Because ``s_i`` already lies on the
        protein solvent-excluded envelope, this is a DNA-to-protein-surface gap rather than an
        atom-centre distance. ``d_i <= positive_gap`` is a hard interface point; values at or above
        ``negative_gap`` are hard non-interface points; the interval between them is excluded by
        ``surface_valid_mask``. Soft targets use a cosine transition across that interval.

        Args:
            positive_gap: Upper surface gap for a confident positive point in Å. The default 1.4 Å
                is the radius of a conventional water probe.
            negative_gap: Lower surface gap for a confident negative point in Å.
            sensitivity_gaps: Additional hard cutoffs stored for threshold-sensitivity analysis.

        Raises:
            ValueError: If thresholds are non-positive, unordered, or sensitivity values empty.
        """
        if positive_gap <= 0.0 or negative_gap <= positive_gap:
            raise ValueError("annotation gaps require 0 < positive_gap < negative_gap")
        if not sensitivity_gaps or any(value <= 0.0 for value in sensitivity_gaps):
            raise ValueError("sensitivity_gaps must contain positive distances")
        self.positive_gap     = float(positive_gap)
        self.negative_gap     = float(negative_gap)
        self.sensitivity_gaps = tuple(float(value) for value in sensitivity_gaps)

    def transform(
        self,
        record                 : Mapping[str, Any],
        resolved_structure_root: Path,
    ) -> dict[str, Any]:
        """Compute aligned target arrays and cryptographic base-geometry provenance.

        Positive rows load DNA coordinates from the exact curated assembly. Base surface points are
        translated back to source coordinates with ``source = centered + coordinate_origin`` before
        the sparse nearest-neighbour query. Curated negatives receive hard zero targets at every
        valid surface point; their DNA distance is not physically computable and is represented by
        NaN together with a false ``surface_distance_valid`` mask.

        Args:
            record: Joined manifest/base-NPZ record prepared by the preprocessing flow.
            resolved_structure_root: Directory receiving verified uncompressed structures.

        Returns:
            Record containing small sidecar arrays, metadata, and a safe output filename.

        Raises:
            TypeError: If the source value is not a mapping.
            ValueError: If base arrays, metadata, label, DNA chains, or coordinate lengths disagree.
            OSError: If the base archive or source assembly cannot be read.
        """
        key   = str(record["key"])
        value = record.get("value")
        if not isinstance(value, Mapping):
            raise TypeError("DNA annotation record must be a mapping")
        value = dict(value)
        label = int(value["label"])
        if label not in {0, 1}:
            raise ValueError("DNA annotation requires a curated binary protein label")

        # Materialize each content-addressed coordinate file once. Every member retains the source
        # structure as auditable dataset evidence, including negatives, but an existing verified
        # file avoids repeating gzip decompression when several chains share one PDB deposition.
        archive_value = value.get("structure_archive_path")
        if archive_value:
            archive_path  = Path(str(archive_value))
            expected_hash = str(value["structure_sha256"])
            structure_path = resolved_structure_root / f"{expected_hash}.cif"
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            if structure_path.is_file():
                observed_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()
                if observed_hash != expected_hash:
                    raise ValueError(
                        f"resolved structure bytes disagree with design for {key}"
                    )
            else:
                content = gzip.decompress(archive_path.read_bytes())
                if hashlib.sha256(content).hexdigest() != expected_hash:
                    raise ValueError(
                        f"geometry structure bytes disagree with design for {key}"
                    )
                self._atomic_write(structure_path, content)
            value["structure_path"] = str(structure_path.resolve())

        base_path = Path(str(value["base_npz"]))
        base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
        with np.load(base_path, allow_pickle=False) as archive:
            required = {
                "metadata_json",
                "residue_indices",
                "surface_area_weights",
                "surface_atom_neighbors",
                "surface_atom_mask",
                "surface_neighbors",
                "surface_neighbor_mask",
                "surface_positions",
            }
            if not required.issubset(archive.files):
                missing = sorted(required - set(archive.files))
                raise ValueError(f"{base_path.name} lacks annotation inputs: {missing}")
            residue_indices  = archive["residue_indices"].astype(np.int64)
            surface_weights  = archive["surface_area_weights"].astype(np.float64)
            surface_atoms    = archive["surface_atom_neighbors"].astype(np.int64)
            surface_atom_mask = archive["surface_atom_mask"].astype(np.bool_)
            surface_neighbors = archive["surface_neighbors"].astype(np.int64)
            surface_neighbor_mask = archive["surface_neighbor_mask"].astype(np.bool_)
            surface_positions = archive["surface_positions"].astype(np.float64)
            metadata          = json.loads(str(archive["metadata_json"].item()))
        if surface_positions.ndim != 2 or surface_positions.shape[1] != 3:
            raise ValueError("base surface_positions must have shape [M,3]")
        origin = np.asarray(metadata.get("coordinate_origin"), dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("base metadata requires a finite coordinate_origin [3]")
        source_surface = surface_positions + origin
        surface_count  = len(source_surface)

        local_method   = str(value.get("local_gt_method", ""))
        local_expected = bool(value.get("local_gt_expected", False))
        local_reason   = "available"
        if label == 1 and local_expected and local_method == "dna_distance":
            structure_path = Path(str(value["structure_path"]))
            structure_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()
            if structure_hash != str(value["structure_sha256"]):
                raise ValueError("DNA assembly bytes do not match curated source provenance")
            rotation    = np.asarray(value["assembly_rotation"], dtype=np.float64)
            translation = np.asarray(value["assembly_translation"], dtype=np.float64)
            if rotation.shape != (3, 3) or translation.shape != (3,):
                raise ValueError("dataset design assembly transform must have shapes [3,3] and [3]")
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                raise ValueError("dataset design assembly transform must be finite")
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
                raise ValueError("dataset design assembly rotation is not orthonormal")
            assembly_surface = source_surface @ rotation.T + translation
            protein_structure = ProteinStructure(structure_path)
            assembled         = protein_structure.assembly(str(value["assembly_id"]))
            assembled.protein_copy(
                str(value["protein_chain"]),
                int(value["protein_copy"]),
            )
            dna_positions, dna_radii, _ = assembled.dna_atoms()
            if not len(dna_positions):
                raise ValueError("positive DNA row has no heavy atoms in its declared assembly")
            center_distance, nearest = cKDTree(dna_positions).query(assembly_surface, k=1)
            distance = center_distance - dna_radii[nearest]
            hard     = (distance <= self.positive_gap).astype(np.uint8)
            valid    = np.logical_or(
                distance <= self.positive_gap,
                distance >= self.negative_gap,
            )
            phase = np.clip(
                (distance - self.positive_gap) / (self.negative_gap - self.positive_gap),
                0.0,
                1.0,
            )
            soft           = 0.5 * (1.0 + np.cos(np.pi * phase))
            distance_valid = np.ones(surface_count, dtype=np.bool_)
            sensitivity    = np.stack(
                [(distance <= cutoff).astype(np.uint8) for cutoff in self.sensitivity_gaps],
                axis=1,
            )
        elif label == 1 and local_expected and local_method == "binding_residue_mask":
            binding_residues = np.asarray(value["binding_residue_indices"], dtype=np.int64)
            if binding_residues.ndim != 1 or not len(binding_residues):
                raise ValueError("binding_residue_mask requires non-empty residue indices")
            if surface_atoms.ndim != 2 or surface_atoms.shape[0] != surface_count:
                raise ValueError("surface_atom_neighbors must have shape [M,Jmax]")

            # Each surface sample inherits the label of its nearest represented protein atom. The
            # atom's flattened residue index is aligned to the design binding-residue evidence.
            if np.any(~surface_atom_mask[:, 0]):
                raise ValueError("every surface point requires a nearest atom in column zero")
            nearest_atoms = surface_atoms[:, 0]
            hard = np.isin(residue_indices[nearest_atoms], binding_residues).astype(np.uint8)

            distance       = np.full(surface_count, np.nan, dtype=np.float64)
            valid          = np.ones(surface_count, dtype=np.bool_)
            soft           = hard.astype(np.float64)
            distance_valid = np.zeros(surface_count, dtype=np.bool_)
            sensitivity    = np.repeat(hard[:, None], len(self.sensitivity_gaps), axis=1)
        elif label == 1:
            distance       = np.full(surface_count, np.nan, dtype=np.float64)
            hard           = np.zeros(surface_count, dtype=np.uint8)
            valid          = np.zeros(surface_count, dtype=np.bool_)
            soft           = np.zeros(surface_count, dtype=np.float64)
            distance_valid = np.zeros(surface_count, dtype=np.bool_)
            sensitivity    = np.zeros((surface_count, len(self.sensitivity_gaps)), dtype=np.uint8)
            local_reason   = "source_has_no_reliable_local_ground_truth"
        else:
            distance       = np.full(surface_count, np.nan, dtype=np.float64)
            hard           = np.zeros(surface_count, dtype=np.uint8)
            valid          = np.ones(surface_count, dtype=np.bool_)
            soft           = np.zeros(surface_count, dtype=np.float64)
            distance_valid = np.zeros(surface_count, dtype=np.bool_)
            sensitivity    = np.zeros((surface_count, len(self.sensitivity_gaps)), dtype=np.uint8)

        # A globally positive sample with no projected positive point remains globally valid, but
        # every local mask entry is disabled so it cannot become a false all-negative surface.
        local_available = label == 0 or bool(np.any(hard == 1))
        if label == 1 and not local_available:
            valid[:] = False
            if local_expected:
                local_reason = "zero_positive_surface_points"
        positive_weight = float(np.sum(surface_weights[hard == 1]))
        total_weight    = float(np.sum(surface_weights))
        region_count = (
            self._region_count(hard, surface_neighbors, surface_neighbor_mask)
            if local_available
            else 0
        )

        annotation_metadata: dict[str, Any] = {
            "annotation_schema_version": self.SCHEMA_VERSION,
            "base_identifier": key,
            "base_npz_sha256": base_hash,
            "base_npz_path": str(base_path.resolve()),
            "base_surface_count": surface_count,
            "source_structure_sha256": str(value["structure_sha256"]),
            "source_structure_path": str(value.get("structure_path", "")),
            "assembly_id": str(value.get("assembly_id", "")),
            "protein_chain": str(value.get("protein_chain", "")),
            "protein_copy": int(value.get("protein_copy", 0)),
            "assembly_rotation": value.get("assembly_rotation"),
            "assembly_translation": value.get("assembly_translation"),
            "protein_label": label,
            "local_gt_expected": local_expected,
            "local_gt_available": local_available,
            "local_gt_reason": local_reason,
            "local_gt_method": local_method,
            "split": str(value.get("split", "")),
            "tier": str(value.get("tier", "")),
            "positive_gap_angstrom": self.positive_gap,
            "negative_gap_angstrom": self.negative_gap,
            "sensitivity_gaps_angstrom": list(self.sensitivity_gaps),
            "distance_definition": "nearest_DNA_atom_center_distance_minus_DNA_vdw_radius",
            "positive_surface_weight": positive_weight,
            "total_surface_weight": total_weight,
            "interface_fraction": positive_weight / total_weight if total_weight > 0.0 else 0.0,
            "number_of_positive_regions": region_count,
        }
        arrays = {
            "surface_target_hard": hard,
            "surface_valid_mask": valid,
            "surface_target_soft": soft.astype(np.float32),
            "surface_distance_to_dna": distance.astype(np.float32),
            "surface_distance_valid": distance_valid,
            "surface_target_hard_sensitivity": sensitivity,
            "local_gt_available": np.asarray(local_available, dtype=np.bool_),
            "sensitivity_gaps": np.asarray(self.sensitivity_gaps, dtype=np.float32),
            "base_npz_sha256": np.asarray(base_hash),
            "annotation_metadata_json": np.asarray(
                json.dumps(annotation_metadata, sort_keys=True, separators=(",", ":"))
            ),
        }
        return {
            "key": key,
            "value": {
                "arrays": arrays,
                "metadata": annotation_metadata,
                "output_name": f"{base_path.stem}.dna.npz",
            },
        }

    def process(
        self,
        record                 : Mapping[str, Any],
        output_root            : Path,
        resolved_structure_root: Path,
    ) -> dict[str, Any]:
        """Compute and atomically persist one sidecar for LambdaForge ``Work.map``.

        Arrays are written inside the worker so the process returns only a compact JSON-compatible
        audit row instead of transferring large arrays to the coordinator. Scientific resume is
        performed here by the sink before transformation; the surrounding framework map remains
        stateless and therefore cannot bypass archive validation.

        Args:
            record: Catalog/base-geometry join for one logical protein.
            output_root: Directory receiving DNA sidecars.
            resolved_structure_root: Directory receiving verified uncompressed structures.

        Returns:
            Record containing the JSON-compatible sidecar audit row.

        Raises:
            TypeError: If annotation inputs or arrays violate their schema.
            ValueError: If provenance, point alignment, or local targets are inconsistent.
            OSError: If structures, base archives, or sidecars cannot be read or written.
        """
        sink    = DNAAnnotationSink()
        resumed = sink.resume(
            record,
            output_root,
            self.positive_gap,
            self.negative_gap,
            self.sensitivity_gaps,
        )
        if resumed is not None:
            return resumed

        transformed = self.transform(record, resolved_structure_root)
        sink.write(transformed, output_root)

        key = str(record["key"])
        return {"key": key, "value": sink.records[key]}

    @staticmethod
    def _region_count(
        hard     : np.ndarray,
        neighbors: np.ndarray,
        mask     : np.ndarray,
    ) -> int:
        """Count positive regions through the persisted bounded surface neighborhood.

        Args:
            hard: Binary point targets with shape ``[M]``.
            neighbors: Nearest-surface IDs with shape ``[M,Ksmax]``.
            mask: Boolean validity table with the same shape.

        Returns:
            Number of connected components induced by positive surface points.

        Raises:
            ValueError: If the sparse graph has an invalid shape or endpoint.
        """
        if neighbors.ndim != 2 or neighbors.shape != mask.shape or len(neighbors) != len(hard):
            raise ValueError("surface neighbor IDs and mask must have shape [M,Ksmax]")
        if np.any(neighbors[mask] < 0) or np.any(neighbors[mask] >= len(hard)):
            raise ValueError("surface neighbor table contains an invalid endpoint")
        positive = set(np.flatnonzero(hard).tolist())
        adjacency: dict[int, list[int]] = {index: [] for index in positive}
        for first in positive:
            for second_value in neighbors[first, mask[first]]:
                second = int(second_value)
                if second in positive:
                    adjacency[first].append(second)
                    adjacency[second].append(first)

        regions = 0
        while positive:
            regions += 1
            pending = [positive.pop()]
            while pending:
                current = pending.pop()
                for neighbor in adjacency[current]:
                    if neighbor in positive:
                        positive.remove(neighbor)
                        pending.append(neighbor)
        return regions

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        """Publish immutable uncompressed structure bytes atomically.

        Args:
            path: Final checkpoint-owned structure path.
            content: Complete verified uncompressed mmCIF bytes.

        Returns:
            ``None`` after ``fsync`` and atomic replacement.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
