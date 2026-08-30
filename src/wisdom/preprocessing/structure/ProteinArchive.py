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
import scipy.sparse

from wisdom.utils.structure.models.Protein import Protein
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessingProvenance import PreprocessingProvenance
from wisdom.preprocessing.structure.SurfaceAtomNeighborhoodBuilder import (
    SurfaceAtomNeighborhoodBuilder,
)


class ProteinArchive:
    """Own the WISDOM NPZ schema, provenance, resume checks, and atomic persistence."""

    SCHEMA_VERSION = "3.0"
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
        "atom_edge_is_covalent",
        "atom_edge_spatial_rank",
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
        "surface_atom_neighbors",
        "surface_atom_distances",
        "surface_atom_normal_offsets",
        "surface_atom_tangential_distances",
        "surface_atom_mask",
        "surface_neighbors",
        "surface_neighbor_distances",
        "surface_neighbor_mask",
        "diffusion_mass",
        "diffusion_eigenvalues",
        "diffusion_eigenvectors",
        "diffusion_gradient_index",
        "diffusion_gradient_x",
        "diffusion_gradient_y",
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
        protein  : Protein,
        provenance: PreprocessingProvenance,
        arrays   : Mapping[str, np.ndarray],
        warnings : list[str],
    ) -> dict[str, Any]:
        """Build sufficient scientific and software provenance for one representation.

        Args:
            protein: Normalized hierarchy used to compute atom/residue counts.
            provenance: Source digest/path/format, selected chains, and coordinate origin.
            arrays: Complete representation used to compute graph and surface counts.
            warnings: Deterministic diagnostics produced during surface construction.

        Returns:
            JSON-compatible source, configuration, version, count, origin, and warning metadata.
        """
        # Scientific settings are serialized once and hashed from the identical payload.
        scientific_config = self.config.scientific_dict()
        try:
            robust_laplacian_version = metadata.version("robust_laplacian")
        except metadata.PackageNotFoundError:
            robust_laplacian_version = "unavailable-distribution-metadata"

        return {
            "protein_id": protein.id,
            "source_identifier": provenance.source_identifier,
            "source_path": provenance.source_path,
            "source_hash": provenance.source_hash,
            "source_format": provenance.source_format,
            "selected_chains": list(provenance.selected_chains),
            "model_index": self.config.model_index,
            "coordinate_origin": list(provenance.coordinate_origin),
            "atom_count": sum(
                len(residue.atoms) for chain in protein.chains for residue in chain.residues
            ),
            "residue_count": sum(len(chain.residues) for chain in protein.chains),
            "atom_edge_count": int(arrays["atom_edge_index"].shape[1]),
            "surface_point_count": int(arrays["surface_positions"].shape[0]),
            "atom_spatial_candidate_count": int(
                np.count_nonzero(arrays["atom_edge_spatial_rank"])
            ),
            "surface_atom_neighbor_count": int(arrays["surface_atom_mask"].sum()),
            "diffusion_spectral_modes": len(arrays["diffusion_eigenvalues"]),
            "diffusion_gradient_entries": len(arrays["diffusion_gradient_x"]),
            "preprocessing_schema_version": self.SCHEMA_VERSION,
            "preprocessing_code_version": metadata.version("wisdom"),
            "config_hash": self._canonical_hash(scientific_config),
            "config": scientific_config,
            "lambdaforge_version": lambdaforge.__version__,
            "gemmi_version": metadata.version("gemmi"),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "robust_laplacian_version": robust_laplacian_version,
            "warnings": warnings,
        }

    def validate(self, arrays: Mapping[str, np.ndarray]) -> dict[str, float | int]:
        """Validate every persisted array and cross-array invariant before publication.

        Args:
            arrays: Complete atom, bounded-neighbour, surface, and differential-operator mapping.

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

        # Delegate cohesive bounded-topology and intrinsic-operator invariants.
        self._validate_undirected_edges(
            "atom_edge_index",
            arrays["atom_edge_index"],
            atom_positions,
            arrays["atom_edge_distance"],
        )
        self._validate_atomic_edge_features(arrays)
        self._validate_surface(arrays)
        self._validate_surface_atom_neighbors(arrays, atom_positions)
        self._validate_diffusion(arrays)
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
            arrays: Already shape-validated WISDOM arrays containing atom/surface geometry,
                bounded nearest-atom tables, and bounded surface neighbors.

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

        neighbor_ids = arrays["surface_atom_neighbors"]
        neighbor_mask = arrays["surface_atom_mask"]
        surface_ids = np.repeat(
            np.arange(len(surface_positions), dtype=np.int32),
            neighbor_mask.sum(axis=1),
        )
        atom_ids = neighbor_ids[neighbor_mask]

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
        surface_neighbor_mask = arrays["surface_neighbor_mask"]
        surface_neighbors     = arrays["surface_neighbors"]
        surface_distances     = arrays["surface_neighbor_distances"]
        row = np.repeat(
            np.arange(len(surface_positions), dtype=np.int32),
            surface_neighbor_mask.sum(axis=1),
        )
        column = surface_neighbors[surface_neighbor_mask]
        adjacency = scipy.sparse.coo_matrix(
            (np.ones(len(row)), (row, column)),
            shape=(len(surface_positions), len(surface_positions)),
        )
        component_count, component_ids = scipy.sparse.csgraph.connected_components(
            adjacency,
            directed=False,
        )
        component_sizes = np.bincount(component_ids, minlength=component_count)
        degree          = surface_neighbor_mask.sum(axis=1)
        isolated_count  = int(np.count_nonzero(degree == 0))

        return {
            "minimum_signed_gap": float(minimum_gaps.min()),
            "maximum_signed_gap": float(minimum_gaps.max()),
            "maximum_absolute_gap": float(np.abs(minimum_gaps).max()),
            "minimum_normal_cosine": float(normal_cosines.min()),
            "mean_normal_cosine": float(normal_cosines.mean()),
            "maximum_dimensionless_curvature": float(dimensionless_curvature.max()),
            "surface_components": int(component_count),
            "isolated_surface_points": isolated_count,
            "largest_component_fraction": float(component_sizes.max() / len(surface_positions)),
            "maximum_surface_neighbor_distance": (
                float(surface_distances[surface_neighbor_mask].max())
                if surface_neighbor_mask.any()
                else 0.0
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
        """Validate covalent flags, spatial ranks, and atomic edge feature lengths.

        Args:
            arrays: Representation containing ``atom_edge_index`` and every atomic edge feature.

        Raises:
            ValueError: If an edge is neither covalent nor a ranked spatial candidate, a spatial
                rank exceeds ``Kmax``, or an edge feature length differs from the edge count.
        """
        edge_count   = arrays["atom_edge_index"].shape[1]
        is_covalent  = arrays["atom_edge_is_covalent"]
        spatial_rank = arrays["atom_edge_spatial_rank"]
        if is_covalent.shape != (edge_count,) or is_covalent.dtype != np.bool_:
            raise ValueError("atom_edge_is_covalent must be Boolean with shape [E]")
        if spatial_rank.shape != (edge_count,) or spatial_rank.dtype.kind not in "iu":
            raise ValueError("atom_edge_spatial_rank must be integer with shape [E]")
        if np.any(spatial_rank > self.config.atom_spatial_k_max):
            raise ValueError("atomic spatial rank exceeds configured Kmax")
        if np.any(~is_covalent & (spatial_rank == 0)):
            raise ValueError("every atomic edge must be covalent or a spatial candidate")
        if np.any(
            (spatial_rank > 0)
            & (arrays["atom_edge_distance"] > self.config.atom_spatial_radius + 2.0e-5)
        ):
            raise ValueError("ranked atomic spatial edge exceeds configured radius")
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
        """Validate fixed surface geometry and normalized represented-area weights.

        Args:
            arrays: Representation containing surface geometry and intrinsic operator arrays.

        Raises:
            ValueError: If shapes, finiteness, normal lengths, curvature values, positive normalized
                area weights are invalid.
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

    def _validate_surface_atom_neighbors(
        self,
        arrays         : Mapping[str, np.ndarray],
        atom_positions : np.ndarray,
    ) -> None:
        """Validate compact surface-to-atom tables and all persisted invariant geometry.

        Args:
            arrays: Representation containing surface positions and bounded nearest-atom arrays.
            atom_positions: Validated ``[N,3]`` atom coordinates defining atom endpoint bounds.

        Raises:
            ValueError: If shape, sentinel, ordering, endpoints, distances, masks, local geometry,
                or complete point coverage is invalid.
        """
        surface_positions = arrays["surface_positions"]
        normals           = arrays["surface_normals"]
        neighbors         = arrays["surface_atom_neighbors"]
        distances         = arrays["surface_atom_distances"]
        normal_offsets    = arrays["surface_atom_normal_offsets"]
        tangential        = arrays["surface_atom_tangential_distances"]
        mask              = arrays["surface_atom_mask"]

        expected_shape = (len(surface_positions), self.config.surface_atom_k_max)
        for name in (
            "surface_atom_neighbors",
            "surface_atom_distances",
            "surface_atom_normal_offsets",
            "surface_atom_tangential_distances",
            "surface_atom_mask",
        ):
            if arrays[name].shape != expected_shape:
                raise ValueError(f"{name} must have shape [M,Jmax]")
        if neighbors.dtype != np.int32 or mask.dtype != np.bool_:
            raise ValueError("surface atom IDs/masks must use int32/bool dtypes")
        if np.any(mask.sum(axis=1) == 0):
            raise ValueError("every surface point must retain at least one atom")
        if np.any(neighbors[~mask] != -1) or np.any(neighbors[mask] < 0):
            raise ValueError("surface atom sentinels disagree with the validity mask")
        if np.any(neighbors[mask] >= len(atom_positions)):
            raise ValueError("surface atom neighbor ID is out of range")
        if not all(
            np.isfinite(value[mask]).all()
            for value in (distances, normal_offsets, tangential)
        ):
            raise ValueError("surface atom geometry contains non-finite valid values")

        for point_index in range(len(surface_positions)):
            valid_ids = neighbors[point_index, mask[point_index]]
            valid_distances = distances[point_index, mask[point_index]]
            if len(np.unique(valid_ids)) != len(valid_ids):
                raise ValueError("a surface atom neighborhood contains duplicate atom IDs")
            order = np.lexsort((valid_ids, valid_distances))
            if not np.array_equal(order, np.arange(len(valid_ids))):
                raise ValueError("surface atom neighbors must be sorted by distance then ID")

        surface_ids = np.repeat(
            np.arange(len(surface_positions), dtype=np.int32),
            mask.sum(axis=1),
        )
        atom_ids = neighbors[mask]
        offsets  = (
            atom_positions[atom_ids].astype(np.float64)
            - surface_positions[surface_ids].astype(np.float64)
        )
        expected_distances, expected_normal, expected_tangent = (
            SurfaceAtomNeighborhoodBuilder.invariant_geometry(
                offsets,
                normals[surface_ids],
            )
        )
        if np.any(expected_distances > self.config.surface_atom_radius + 2.0e-5):
            raise ValueError("surface atom neighbor exceeds configured radius")
        if not np.allclose(distances[mask], expected_distances, rtol=2e-5, atol=2e-5):
            raise ValueError("surface atom distances disagree with coordinates")
        if not np.allclose(normal_offsets[mask], expected_normal, rtol=2e-5, atol=2e-5):
            raise ValueError("surface atom normal offsets disagree with coordinates")
        if not np.allclose(tangential[mask], expected_tangent, rtol=2e-5, atol=2e-5):
            raise ValueError("surface atom tangential distances disagree with coordinates")

    def _validate_diffusion(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Validate bounded surface neighbors, low spectrum, and sparse tangent gradients.

        Args:
            arrays: Representation containing all DiffusionNet operator arrays.

        Raises:
            ValueError: If operator dimensions, sparse indices, finite values, positive mass,
                eigenvalue ordering, mass orthonormality, or constant-gradient behavior fails.
        """
        point_count = len(arrays["surface_positions"])
        neighbor_shape = (point_count, self.config.surface_neighbor_k_max)
        neighbors = arrays["surface_neighbors"]
        distances = arrays["surface_neighbor_distances"]
        mask      = arrays["surface_neighbor_mask"]
        if neighbors.shape != neighbor_shape or distances.shape != neighbor_shape:
            raise ValueError("surface neighbor tables must have shape [M,Ksmax]")
        if mask.shape != neighbor_shape or mask.dtype != np.bool_ or neighbors.dtype != np.int32:
            raise ValueError("surface neighbor masks/IDs have invalid shape or dtype")
        if np.any(mask.sum(axis=1) == 0) or np.any(neighbors[~mask] != -1):
            raise ValueError("surface neighbor masks and sentinels are inconsistent")
        if np.any(neighbors[mask] < 0) or np.any(neighbors[mask] >= point_count):
            raise ValueError("surface neighbor ID is out of range")
        if not np.isfinite(distances[mask]).all() or np.any(distances[mask] <= 0.0):
            raise ValueError("surface neighbor distances must be finite and positive")

        mass         = arrays["diffusion_mass"]
        eigenvalues  = arrays["diffusion_eigenvalues"]
        eigenvectors = arrays["diffusion_eigenvectors"]
        if mass.shape != (point_count,) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
            raise ValueError("diffusion mass must be finite, positive, and have shape [M]")
        mode_count = len(eigenvalues)
        if (
            mode_count < 1
            or mode_count > self.config.diffusion_spectral_modes_max
            or eigenvectors.shape != (point_count, mode_count)
            or not np.isfinite(eigenvalues).all()
            or not np.isfinite(eigenvectors).all()
        ):
            raise ValueError("diffusion eigenpair dimensions or values are invalid")
        if np.any(eigenvalues < -2.0e-5) or np.any(np.diff(eigenvalues) < -2.0e-5):
            raise ValueError("diffusion eigenvalues must be nonnegative and sorted")
        gram = eigenvectors.T @ (mass[:, None] * eigenvectors)
        if not np.allclose(gram, np.eye(mode_count), rtol=2.0e-3, atol=2.0e-3):
            raise ValueError("diffusion eigenvectors are not mass-orthonormal")

        gradient_index = arrays["diffusion_gradient_index"]
        gradient_x     = arrays["diffusion_gradient_x"]
        gradient_y     = arrays["diffusion_gradient_y"]
        if (
            gradient_index.dtype != np.int32
            or gradient_index.ndim != 2
            or gradient_index.shape[0] != 2
            or gradient_x.shape != (gradient_index.shape[1],)
            or gradient_y.shape != (gradient_index.shape[1],)
        ):
            raise ValueError("diffusion gradient COO arrays have invalid dimensions")
        if gradient_index.size and (
            gradient_index.min() < 0 or gradient_index.max() >= point_count
        ):
            raise ValueError("diffusion gradient COO index is out of range")
        if not np.isfinite(gradient_x).all() or not np.isfinite(gradient_y).all():
            raise ValueError("diffusion gradient values must be finite")
        row_sum_x = np.bincount(
            gradient_index[0], weights=gradient_x, minlength=point_count
        )
        row_sum_y = np.bincount(
            gradient_index[0], weights=gradient_y, minlength=point_count
        )
        if not np.allclose(row_sum_x, 0.0, atol=2.0e-5) or not np.allclose(
            row_sum_y, 0.0, atol=2.0e-5
        ):
            raise ValueError("diffusion gradient of a constant field is not zero")

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
