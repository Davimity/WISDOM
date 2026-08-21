"""Validated, reproducible and atomic NPZ storage."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any, cast
from uuid import uuid4
from zipfile import BadZipFile

import lambdaforge
import numpy as np
import scipy

from preprocess import __version__
from preprocess.dataclasses.Protein import Protein
from preprocess.dataclasses.ProteinMetadata import ProteinMetadata
from preprocess.enums.Relation import Relation
from preprocess.PreprocessConfig import PreprocessConfig


class StorageManager:
    """Own the WISDOM NPZ schema, provenance, resume checks, and atomic persistence."""

    SCHEMA_VERSION = "2.1"
    METADATA_NAME  = "metadata_json"
    ARRAY_NAMES    = (
        "atom_positions",
        "atomic_numbers",
        "residue_type_ids",
        "atom_role_ids",
        "residue_indices",
        "chain_indices",
        "formal_charges",
        "vdw_radii",
        "covalent_radii",
        "atom_names",
        "residue_names",
        "atom_edge_index",
        "atom_edge_distance",
        "atom_edge_relation_mask",
        "atom_edge_bond_type",
        "atom_edge_bond_order",
        "atom_edge_bond_source",
        "atom_edge_bond_confidence",
        "atom_edge_same_residue",
        "atom_edge_same_chain",
        "atom_edge_residue_separation",
        "surface_positions",
        "surface_normals",
        "surface_curvatures",
        "surface_area_weights",
        "surface_component_ids",
        "surface_edge_index",
        "surface_edge_distance",
        "surface_atom_edge_index",
        "surface_atom_distance",
    )

    def __init__(self, config: PreprocessConfig) -> None:
        """Bind schema identity and scientific configuration to storage operations.

        Args:
            config: Effective configuration used for metadata and compatibility hashes.
        """
        self.config = config

    @property
    def config_hash(self) -> str:
        """Return the SHA-256 identity of settings that change scientific arrays.

        Returns:
            Hexadecimal SHA-256 over canonical compact JSON from ``scientific_dict``.
        """
        return self._canonical_hash(self.config.scientific_dict())

    def make_metadata(
        self,
        protein          : Protein,
        protein_metadata : ProteinMetadata,
        arrays           : Mapping[str, np.ndarray],
        warnings         : list[str],
    ) -> dict[str, Any]:
        """Build sufficient scientific and software provenance for one representation.

        Args:
            protein: Normalized hierarchy used to compute atom/residue counts.
            protein_metadata: Source digest/path/format, selected chains, and coordinate origin.
            arrays: Complete representation used to compute graph and surface counts.
            warnings: Deterministic diagnostics produced during surface construction.

        Returns:
            JSON-compatible source, configuration, version, count, origin, and warning metadata.
        """
        # Scientific settings are serialized once and hashed from the identical payload.
        scientific_config = self.config.scientific_dict()
        return {
            "protein_id": protein.id,
            "source_identifier": protein_metadata.source_identifier,
            "source_path": protein_metadata.source_path,
            "source_hash": protein_metadata.source_hash,
            "source_format": protein_metadata.source_format,
            "selected_chains": list(protein_metadata.selected_chains),
            "model_index": self.config.model_index,
            "coordinate_origin": list(protein_metadata.coordinate_origin),
            "atom_count": sum(
                len(residue.atoms) for chain in protein.chains for residue in chain.residues
            ),
            "residue_count": sum(len(chain.residues) for chain in protein.chains),
            "atom_edge_count": int(arrays["atom_edge_index"].shape[1]),
            "surface_point_count": int(arrays["surface_positions"].shape[0]),
            "surface_edge_count": int(arrays["surface_edge_index"].shape[1]),
            "surface_atom_edge_count": int(arrays["surface_atom_edge_index"].shape[1]),
            "preprocessing_schema_version": self.SCHEMA_VERSION,
            "preprocessing_code_version": __version__,
            "config_hash": self._canonical_hash(scientific_config),
            "config": scientific_config,
            "lambdaforge_version": lambdaforge.__version__,
            "gemmi_version": metadata.version("gemmi"),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "warnings": warnings,
        }

    def validate(self, arrays: Mapping[str, np.ndarray]) -> dict[str, float | int]:
        """Validate every persisted array and cross-array invariant before publication.

        Args:
            arrays: Complete atom, atomic-edge, surface, surface-edge, and bipartite array mapping.

        Returns:
            Numerical surface diagnostics quantifying envelope error, outward-normal agreement,
            dimensionless curvature magnitude, and graph connectivity.

        Raises:
            ValueError: If arrays are missing, object-typed, incorrectly shaped, non-finite,
                out-of-range, duplicated, non-unit, unnormalized, or geometrically inconsistent.
        """
        # Schema completeness and pickle safety are checked before indexing individual arrays.
        required = set(self.ARRAY_NAMES)
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"missing required arrays: {sorted(missing)}")
        if any(array.dtype == object for array in arrays.values()):
            raise ValueError("object arrays are forbidden")

        # Atom coordinates establish N and the valid domain of every atom-level array/index.
        atom_positions = arrays["atom_positions"]
        if atom_positions.ndim != 2 or atom_positions.shape[1] != 3 or len(atom_positions) == 0:
            raise ValueError("atom_positions must have non-empty shape [N,3]")
        if not np.isfinite(atom_positions).all():
            raise ValueError("atom_positions contains non-finite values")
        atom_count    = len(atom_positions)
        atom_features = (
            "atomic_numbers",
            "residue_type_ids",
            "atom_role_ids",
            "residue_indices",
            "chain_indices",
            "formal_charges",
            "vdw_radii",
            "covalent_radii",
            "atom_names",
            "residue_names",
        )
        for name in atom_features:
            if arrays[name].shape != (atom_count,):
                raise ValueError(f"{name} must have shape [N]")
        if np.any(arrays["atomic_numbers"] == 0) or np.any(arrays["residue_indices"] < 0):
            raise ValueError("atomic numbers and residue indices must be valid")

        # Delegate cohesive edge/surface invariants after basic atom feature validation.
        self._validate_undirected_edges(
            "atom_edge_index",
            arrays["atom_edge_index"],
            atom_positions,
            arrays["atom_edge_distance"],
        )
        self._validate_atomic_edge_features(arrays)
        self._validate_surface(arrays)
        self._validate_bipartite(arrays, atom_positions)
        return self.surface_diagnostics(arrays)

    def surface_diagnostics(
        self,
        arrays: Mapping[str, np.ndarray],
    ) -> dict[str, float | int]:
        """Measure and enforce molecular-envelope, normal, curvature, and isolation quality.

        For surface point ``p`` and nearby atom ``i``, the signed expanded-sphere gap is
        ``g_i(p)=||p-c_i||-(r_i+probe_radius)``. A published union-boundary point must satisfy
        ``min_i g_i(p)`` within ``max(5e-4, 0.025h)`` ångströms of zero, where ``h`` is surface
        resolution. A negative value beyond that tolerance means the point is inside an expanded
        sphere; a positive value means it is floating away from every sphere.

        The expected envelope normal is rebuilt from the same soft minimum of active sphere
        gradients used during generation. Its cosine with the stored normal must be at least 0.99.
        Curvature triplets must satisfy ``C² = 2H²-K`` and the scale-normalized curvedness ``C*r``
        must not exceed 25, a conservative guard against numerically singular local fits rather
        than a physical curvature cutoff.

        Args:
            arrays: Already shape-validated WISDOM arrays containing atom/surface geometry and the
                complete surface-to-atom incidence within ``atom_surface_radius``.

        Returns:
            JSON-compatible scalar diagnostics including signed-gap extrema, normal cosine,
            curvature bound, component counts, isolated points, and maximum graph-neighbor gap.

        Raises:
            ValueError: If a point is inside/floating away from the molecular envelope, a normal is
                inconsistent, a curvature triplet is algebraically or numerically unstable, or the
                bipartite neighborhood cannot reconstruct every point.
        """
        atom_positions = arrays["atom_positions"].astype(np.float64, copy=False)
        expanded_radii = (
            arrays["vdw_radii"].astype(np.float64, copy=False) + self.config.probe_radius
        )

        surface_positions = arrays["surface_positions"].astype(np.float64, copy=False)
        surface_normals   = arrays["surface_normals"].astype(np.float64, copy=False)

        edge_index  = arrays["surface_atom_edge_index"]
        surface_ids = edge_index[0]
        atom_ids    = edge_index[1]

        # Recompute geometry instead of trusting stored edge distances already checked elsewhere.
        edge_offsets = surface_positions[surface_ids] - atom_positions[atom_ids]
        distances    = np.linalg.norm(edge_offsets, axis=1)

        # The signed union gap detects both points buried inside atoms and points floating outside.
        gaps         = distances - expanded_radii[atom_ids]
        minimum_gaps = np.full(len(surface_positions), np.inf, dtype=np.float64)
        np.minimum.at(minimum_gaps, surface_ids, gaps)
        if not np.isfinite(minimum_gaps).all():
            raise ValueError("surface envelope cannot be reconstructed for every point")

        gap_tolerance = max(5.0e-4, 0.025 * self.config.surface_resolution)
        if float(minimum_gaps.min()) < -gap_tolerance:
            raise ValueError("a surface point lies inside the expanded-sphere molecular envelope")
        if float(minimum_gaps.max()) > gap_tolerance:
            raise ValueError(
                "a surface point floats outside the expanded-sphere molecular envelope"
            )

        # Reconstruct the outward soft-min gradient and compare it with every persisted normal.
        radial = edge_offsets / np.maximum(distances[:, None], 1.0e-12)

        smoothness = max(0.25 * self.config.surface_resolution, 1.0e-3)
        active      = gaps <= minimum_gaps[surface_ids] + 2.5 * smoothness
        weights     = np.exp(-(gaps[active] - minimum_gaps[surface_ids[active]]) / smoothness)

        expected_normals = np.zeros_like(surface_positions)
        np.add.at(expected_normals, surface_ids[active], weights[:, None] * radial[active])
        expected_lengths = np.linalg.norm(expected_normals, axis=1, keepdims=True)
        expected_normals /= np.maximum(expected_lengths, 1.0e-12)

        normal_cosines = np.sum(expected_normals * surface_normals, axis=1)
        if float(normal_cosines.min()) < 0.99:
            raise ValueError("a surface normal disagrees with the outward molecular envelope")

        # H, K and C derive from two real principal curvatures and obey exact identities.
        curvatures = arrays["surface_curvatures"].astype(np.float64, copy=False)
        mean       = curvatures[:, :, 0]
        gaussian   = curvatures[:, :, 1]
        curvedness = curvatures[:, :, 2]
        if not np.allclose(
            curvedness * curvedness,
            2.0 * mean * mean - gaussian,
            rtol=2.0e-4,
            atol=2.0e-6,
        ):
            raise ValueError("surface curvature channels violate C^2 = 2H^2 - K")

        radii = self.config.surface_resolution * np.asarray(self.config.curvature_scales)
        dimensionless_curvature = curvedness * radii[None, :]
        if float(dimensionless_curvature.max()) > 25.0:
            raise ValueError("surface curvature contains a numerically unstable local fit")

        # Connectivity is diagnostic rather than a validity failure because separate chains or
        # cavities may legitimately create more than one local surface component.
        component_ids   = arrays["surface_component_ids"]
        component_sizes = np.bincount(component_ids)
        graph_edges     = arrays["surface_edge_index"]
        graph_distances = arrays["surface_edge_distance"]

        degree = np.zeros(len(surface_positions), dtype=np.int64)
        np.add.at(degree, graph_edges[0], 1)
        np.add.at(degree, graph_edges[1], 1)
        isolated_count = int(np.count_nonzero(degree == 0))

        return {
            "minimum_signed_gap": float(minimum_gaps.min()),
            "maximum_signed_gap": float(minimum_gaps.max()),
            "maximum_absolute_gap": float(np.abs(minimum_gaps).max()),
            "minimum_normal_cosine": float(normal_cosines.min()),
            "mean_normal_cosine": float(normal_cosines.mean()),
            "maximum_dimensionless_curvature": float(dimensionless_curvature.max()),
            "surface_components": len(component_sizes),
            "isolated_surface_points": isolated_count,
            "largest_component_fraction": float(component_sizes.max() / len(surface_positions)),
            "maximum_surface_edge_distance": (
                float(graph_distances.max()) if len(graph_distances) else 0.0
            ),
        }

    def read_metadata(self, path: Path) -> dict[str, Any] | None:
        """Read pickle-free scalar JSON metadata from an existing NPZ defensively.

        Args:
            path: Candidate per-protein NPZ path.

        Returns:
            Parsed metadata mapping, or ``None`` when the file/array/JSON is absent or invalid.
        """
        try:
            with np.load(path, allow_pickle=False) as archive:
                value = json.loads(str(archive[self.METADATA_NAME].item()))
                return cast(dict[str, Any], value)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, BadZipFile, EOFError):
            return None

    def can_resume(
        self,
        path        : Path,
        source_hash : str,
    ) -> bool:
        """Decide whether an existing representation is scientifically reusable.

        Args:
            path: Existing or prospective NPZ path.
            source_hash: SHA-256 of the exact currently resolved source bytes.

        Returns:
            ``True`` only when source hash, scientific configuration hash, and schema version equal
            current values and the exact pickle-free archive passes every schema and numerical
            invariant. Corrupt or partially written archives always return ``False``.
        """
        value = self.read_metadata(path)
        compatible = bool(
            value is not None
            and value.get("source_hash") == source_hash
            and value.get("config_hash") == self.config_hash
            and value.get("preprocessing_schema_version") == self.SCHEMA_VERSION
        )
        if not compatible:
            return False

        # Metadata identity is necessary but insufficient: validate the exact reusable arrays too.
        try:
            with np.load(path, allow_pickle=False) as archive:
                expected_names = {*self.ARRAY_NAMES, self.METADATA_NAME}
                if set(archive.files) != expected_names:
                    return False
                arrays = {name: archive[name] for name in self.ARRAY_NAMES}
            self.validate(arrays)
        except (OSError, ValueError, KeyError, BadZipFile, EOFError):
            return False
        return True

    def write(
        self,
        path           : Path,
        arrays         : Mapping[str, np.ndarray],
        metadata_value : Mapping[str, Any],
    ) -> None:
        """Validate, verify, and atomically publish one compressed pickle-free NPZ.

        The method writes a PID/UUID temporary file, flushes and synchronizes it, reopens it with
        ``allow_pickle=False``, revalidates arrays/JSON, and finally publishes with ``os.replace``.

        Args:
            path: Final per-protein NPZ destination.
            arrays: Complete validated scientific representation.
            metadata_value: JSON-compatible provenance mapping stored as a scalar Unicode array.

        Raises:
            ValueError: If pre-write or reopened representation validation fails.
            OSError: If directory creation, writing, synchronization, reopening, or rename fails.
        """
        self.validate(arrays)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Metadata is encoded as Unicode JSON so the archive remains object-free and pickle-free.
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        payload   = dict(arrays)
        payload[self.METADATA_NAME] = np.asarray(
            json.dumps(metadata_value, sort_keys=True, separators=(",", ":"))
        )

        # Publish only after a full close/reopen validation of the exact temporary bytes.
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)  # type: ignore[arg-type]
                handle.flush()
                os.fsync(handle.fileno())
            with np.load(temporary, allow_pickle=False) as archive:
                self.validate({name: archive[name] for name in arrays})
                json.loads(str(archive[self.METADATA_NAME].item()))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def write_report(
        path  : Path,
        value : Mapping[str, Any],
    ) -> None:
        """Atomically publish the global human-readable JSON run report.

        Args:
            path: Final report path inside the LambdaForge run directory.
            value: JSON-compatible aggregate counts and ordered per-record results.

        Raises:
            OSError: If directory creation, writing, synchronization, or publication fails.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def write_text(
        path  : Path,
        value : str,
    ) -> None:
        """Atomically publish a UTF-8 human-readable text artifact.

        Args:
            path: Final text path inside the LambdaForge run directory.
            value: Complete report text; a trailing newline is added when absent.

        Raises:
            OSError: If directory creation, writing, synchronization, or publication fails.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(value)
                if not value.endswith("\n"):
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_atomic_edge_features(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Validate relation masks and one-dimensional feature lengths for atomic edges.

        Args:
            arrays: Representation containing ``atom_edge_index`` and every atomic edge feature.

        Raises:
            ValueError: If relation masks are outside ``{SPATIAL, COVALENT, BOTH}`` or an edge
                feature length differs from the edge count.
        """
        edge_count  = arrays["atom_edge_index"].shape[1]
        relation    = arrays["atom_edge_relation_mask"]
        valid_masks = [
            Relation.SPATIAL,
            Relation.COVALENT,
            Relation.SPATIAL | Relation.COVALENT,
        ]
        if relation.shape != (edge_count,) or not np.all(np.isin(relation, valid_masks)):
            raise ValueError("atom relation masks are invalid")
        names = (
            "atom_edge_bond_type",
            "atom_edge_bond_order",
            "atom_edge_bond_source",
            "atom_edge_bond_confidence",
            "atom_edge_same_residue",
            "atom_edge_same_chain",
            "atom_edge_residue_separation",
        )
        for name in names:
            if arrays[name].shape != (edge_count,):
                raise ValueError(f"{name} must have shape [E]")

    def _validate_surface(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Validate surface geometry, normalized area, components, and undirected topology.

        Args:
            arrays: Representation containing all surface point and surface graph arrays.

        Raises:
            ValueError: If shapes, finiteness, normal lengths, curvature values, positive normalized
                area weights, component IDs, or graph invariants are invalid.
        """
        positions = arrays["surface_positions"]
        normals   = arrays["surface_normals"]
        count     = len(positions)
        if positions.shape != (count, 3) or count == 0:
            raise ValueError("surface_positions must have non-empty shape [M,3]")
        if (
            normals.shape != (count, 3)
            or not np.isfinite(positions).all()
            or not np.isfinite(normals).all()
        ):
            raise ValueError("surface positions or normals are invalid")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("surface normals are not unit length")

        curvatures = arrays["surface_curvatures"]
        expected_curvature_shape = (count, len(self.config.curvature_scales), 3)
        if curvatures.shape != expected_curvature_shape or not np.isfinite(curvatures).all():
            raise ValueError("surface curvatures are invalid")

        weights = arrays["surface_area_weights"]
        if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("surface area weights must be finite and positive")
        if not np.isclose(weights.sum(), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("surface area weights must sum to one")

        components = arrays["surface_component_ids"]
        if components.shape != (count,) or np.any(components < 0):
            raise ValueError("surface component IDs are invalid")
        self._validate_undirected_edges(
            "surface_edge_index",
            arrays["surface_edge_index"],
            positions,
            arrays["surface_edge_distance"],
        )

    def _validate_bipartite(
        self,
        arrays         : Mapping[str, np.ndarray],
        atom_positions : np.ndarray,
    ) -> None:
        """Validate surface-to-atom incidence, distances, and complete surface coverage.

        Args:
            arrays: Representation containing surface positions and bipartite edge arrays.
            atom_positions: Validated ``[N,3]`` atom coordinates defining atom endpoint bounds.

        Raises:
            ValueError: If dtype/shape/endpoints/distances are invalid or a surface point has no
                incident atom edge.
        """
        surface_positions = arrays["surface_positions"]
        edge_index        = arrays["surface_atom_edge_index"]
        distances         = arrays["surface_atom_distance"]
        if edge_index.dtype != np.int32 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("surface_atom_edge_index must have shape [2,E] and dtype int32")
        if distances.shape != (edge_index.shape[1],) or not np.isfinite(distances).all():
            raise ValueError("surface-atom distances are invalid")
        if edge_index.size:
            if np.any(edge_index[0] < 0) or np.any(edge_index[0] >= len(surface_positions)):
                raise ValueError("surface indices are out of range")
            if np.any(edge_index[1] < 0) or np.any(edge_index[1] >= len(atom_positions)):
                raise ValueError("atom indices are out of range")
            expected = np.linalg.norm(
                surface_positions[edge_index[0]] - atom_positions[edge_index[1]], axis=1
            )
            if not np.allclose(distances, expected, rtol=2e-5, atol=2e-5):
                raise ValueError("surface-atom distances disagree with positions")
        if len(np.unique(edge_index[0])) != len(surface_positions):
            raise ValueError("every surface point must be related to at least one atom")

    @staticmethod
    def _validate_undirected_edges(
        name       : str,
        edge_index : np.ndarray,
        positions  : np.ndarray,
        distances  : np.ndarray,
    ) -> None:
        """Validate one sparse undirected Euclidean graph against its node coordinates.

        Args:
            name: Edge-index array name used in precise validation messages.
            edge_index: Expected ``int32 [2,E]`` unique endpoints satisfying ``src < dst``.
            positions: Finite node coordinates with shape ``[V,3]``.
            distances: Persisted Euclidean distance for each edge with shape ``[E]``.

        Raises:
            ValueError: If shape/dtype/range/order/uniqueness fails or stored distances disagree
                with ``||positions[src]-positions[dst]||_2`` within float32 tolerance.
        """
        if edge_index.dtype != np.int32 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"{name} must have shape [2,E] and dtype int32")
        if distances.shape != (edge_index.shape[1],) or not np.isfinite(distances).all():
            raise ValueError(f"{name} distances have invalid shape or values")
        if not edge_index.size:
            return
        if edge_index.min() < 0 or edge_index.max() >= len(positions):
            raise ValueError(f"{name} indices are out of range")
        if not np.all(edge_index[0] < edge_index[1]):
            raise ValueError(f"{name} must satisfy src < dst")
        pairs = edge_index.T
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError(f"{name} contains duplicate edges")
        # Recompute every persisted distance from final compact coordinates.
        expected = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
        if not np.allclose(distances, expected, rtol=2e-5, atol=2e-5):
            raise ValueError(f"{name} distances disagree with positions")

    @staticmethod
    def _canonical_hash(value: Mapping[str, Any]) -> str:
        """Hash a mapping through deterministic compact ASCII JSON serialization.

        Args:
            value: JSON-compatible mapping whose key order must not affect identity.

        Returns:
            Hexadecimal SHA-256 digest of sorted, separator-normalized, ASCII JSON bytes.
        """
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode()).hexdigest()
