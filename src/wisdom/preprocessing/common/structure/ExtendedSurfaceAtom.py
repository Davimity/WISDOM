"""Optional, base-bound CSR neighborhoods; the schema-3 transfer table is untouched."""

import json
import numpy as np

from typing import Any
from pathlib import Path
from itertools import pairwise
from scipy.spatial import cKDTree
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.SurfaceAtomNeighborhoodBuilder import (
    SurfaceAtomNeighborhoodBuilder,
)


class ExtendedSurfaceAtom:
    """Build and verify sparse surface candidates without padded nonexistent atoms."""

    VERSION = "1.0"

    def __init__(self, radius: float = 12.0, maximum: int = 96) -> None:
        """Choose an explicit physical cutoff and audited per-point storage cap.

        Args:
            radius: Positive atom-center cutoff in ångströms, inclusive.
            maximum: Positive stored candidate cap per point; excess is counted, never hidden.
        """
        if not np.isfinite(radius) or radius <= 0 or maximum < 1:
            raise ValueError("extended radius and maximum must be finite and positive")
        self.radius  = radius
        self.maximum = maximum

    def build(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Construct deterministic CSR rows in bounded query blocks.

        Args:
            arrays: Base atom/surface positions [N,3]/[M,3] and normals [M,3], in Å.

        Returns:
            int64 point_ptr [M+1], int32 atom_index [P], float32 geometry [P], and
            int32 candidate_count/discarded_count [M]. Empty rows are legal: a CSR row
            of length zero has no candidate, not an invented atom zero.
        """
        atoms   = np.asarray(arrays["atom_positions"], dtype=np.float64)
        points  = np.asarray(arrays["surface_positions"], dtype=np.float64)
        normals = np.asarray(arrays["surface_normals"], dtype=np.float64)
        tree    = cKDTree(atoms)

        counts = np.zeros(len(points), dtype=np.int32)
        rows: list[np.ndarray] = []
        geometry: list[np.ndarray] = []

        # Query blocks bound temporary Python neighbor lists even on large surfaces. Sorting
        # rounded distances reproduces the historic writer's persisted-distance/atom-ID rule.

        for start in range(0, len(points), 1024):
            candidates = tree.query_ball_point(points[start:start + 1024], self.radius, workers=1)
            for local, ids in enumerate(candidates):
                point = start + local
                indices = np.asarray(ids, dtype=np.int32)
                counts[point] = len(indices)
                d, z, rho = SurfaceAtomNeighborhoodBuilder.invariant_geometry(
                    atoms[indices] - points[point], normals[point])
                distance = d.astype(np.float32)
                order = np.lexsort((indices, distance))[:self.maximum]
                rows.append(indices[order])
                geometry.append(np.column_stack((distance[order], z[order], rho[order])).astype(
                    np.float32))

        lengths = np.minimum(counts, self.maximum)
        pointer = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        values  = np.concatenate(geometry) if geometry else np.empty((0, 3), np.float32)
        return {
            "point_ptr":       pointer,
            "atom_index":      np.concatenate(rows) if rows else np.empty(0, np.int32),
            "distance":        values[:, 0],
            "normal_offset":   values[:, 1],
            "tangential":      values[:, 2],
            "candidate_count": counts,
            "discarded_count": counts - lengths,
        }

    def write(self, base: Path, output: Path) -> None:
        """Write a pickle-free independently versioned sidecar, without modifying base bytes.

        Args:
            base: Immutable schema-3 universal NPZ.
            output: Temporary destination owned by LambdaForge's managed file API.

        Raises:
            ValueError: Invalid coordinates, normals, or a non-schema-3 input.
        """
        with np.load(base, allow_pickle=False) as archive:
            original = json.loads(str(archive["metadata_json"].item()))
            if original["preprocessing_schema_version"] != "3.0":
                raise ValueError("extended candidates require structural schema 3.0")
            arrays = {name: archive[name] for name in (
                "atom_positions", "surface_positions", "surface_normals")}
        values = self.build(arrays)
        metadata = {
            "schema": self.VERSION, "algorithm": "csr-persisted-distance-id-v1",
            "radius_angstrom": self.radius, "maximum": self.maximum,
            "units": "angstrom", "offset": "atom-minus-point; outward normalized normal",
            "source_hash": original.get("source_hash"), "config_hash": original.get("config_hash"),
            "numpy_version": np.__version__, "coverage": self.coverage(values["candidate_count"]),
            "atom_count": len(arrays["atom_positions"]),
            "point_count": len(arrays["surface_positions"]),
        }
        with output.open("wb") as stream:
            np.savez_compressed(stream, **values,  # type: ignore[arg-type]
                base_npz_sha256=np.asarray(ProteinArchive.sha256(base)),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))

    def validate(self, path: Path, base: Path) -> bool:
        """Check fingerprint, exact schema, CSR order, truncation and recomputed geometry.

        Args:
            path: Candidate managed sidecar, opened without pickle.
            base: Exact universal NPZ defining atom/point indices and coordinate frame.

        Returns:
            True only after all structural and numerical checks pass.

        Raises:
            ValueError: Wrong fingerprint, schema, dtypes, coverage, sorting or geometry.
        """
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        metadata = json.loads(str(values.pop("metadata_json").item()))
        digest   = str(values.pop("base_npz_sha256").item())
        if digest != ProteinArchive.sha256(base):
            raise ValueError("extended surface-atom base NPZ hash mismatch")
        if (metadata["schema"], metadata["radius_angstrom"], metadata["maximum"]) != (
            self.VERSION, self.radius, self.maximum):
            raise ValueError("extended surface-atom schema/configuration mismatch")
        if metadata.get("algorithm") != "csr-persisted-distance-id-v1" or metadata.get(
            "units") != "angstrom":
            raise ValueError("extended surface-atom algorithm/units mismatch")
        with np.load(base, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in (
                "atom_positions", "surface_positions", "surface_normals")}
            original = json.loads(str(archive["metadata_json"].item()))
        if any(metadata.get(key) != original.get(key) for key in ("source_hash", "config_hash")):
            raise ValueError("extended surface-atom source/config provenance mismatch")

        # Full deterministic reconstruction also detects a missing in-radius atom or a wrong
        # pre-truncation count; geometry-only checks would not establish candidate completeness.

        expected = self.build(arrays)
        if metadata.get("atom_count") != len(arrays["atom_positions"]) or metadata.get(
            "point_count") != len(arrays["surface_positions"]):
            raise ValueError("extended surface-atom support counts mismatch")
        if set(values) != set(expected):
            raise ValueError("extended surface-atom array schema mismatch")
        for key, actual in values.items():
            wanted = expected[key]
            if actual.dtype.kind == "f" and not np.isfinite(actual).all():
                raise ValueError(f"non-finite extended surface-atom array: {key}")
            if actual.dtype != wanted.dtype or not np.array_equal(actual, wanted):
                raise ValueError(f"invalid extended surface-atom array: {key}")
        if metadata["coverage"] != self.coverage(values["candidate_count"]):
            raise ValueError("extended surface-atom coverage report mismatch")
        return True

    @staticmethod
    def coverage(counts: np.ndarray) -> dict[str, Any]:
        """Summarize actual pre-truncation candidate support, including empty neighborhoods.

        Args:
            counts: Integer candidate totals [M] before applying any storage cap.

        Returns:
            JSON min/mean/median/p90/p95/p99/max and fractions exceeding 32/64/96/128.
        """
        return {
            "minimum": int(counts.min()) if len(counts) else 0,
            "mean": float(counts.mean()) if len(counts) else 0.0,
            "quantiles": np.quantile(counts, [.5, .9, .95, .99]).tolist() if len(counts) else [],
            "maximum": int(counts.max()) if len(counts) else 0,
            "above": {str(k): float(np.mean(counts > k)) if len(counts) else 0.0
                      for k in (32, 64, 96, 128)},
        }

    @staticmethod
    def dense(values: Mapping[str, np.ndarray], maximum: int) -> dict[str, np.ndarray]:
        """Recover a padded prefix for an explicitly requested future consumer.

        Args:
            values: Validated CSR arrays for one protein, never concatenated across proteins.
            maximum: Positive width of the returned prefix, not larger than stored support.

        Returns:
            Historical-style [M,J] tables: ID -1, false mask and zero geometry for padding.
            Consumers must additionally filter distance when requesting a smaller radius.
        """
        ptr = values["point_ptr"]
        result = {"surface_atom_neighbors": np.full((len(ptr) - 1, maximum), -1, np.int32),
                  "surface_atom_mask": np.zeros((len(ptr) - 1, maximum), np.bool_)}
        names = {"distance": "distances", "normal_offset": "normal_offsets",
                 "tangential": "tangential_distances"}
        result.update({f"surface_atom_{name}": np.zeros((len(ptr) - 1, maximum), np.float32)
                       for name in names.values()})
        for row, (start, stop) in enumerate(pairwise(ptr)):
            stop = min(stop, start + maximum)
            length = stop - start
            result["surface_atom_neighbors"][row, :length] = values["atom_index"][start:stop]
            result["surface_atom_mask"][row, :length] = True
            for source, target in names.items():
                result[f"surface_atom_{target}"][row, :length] = values[source][start:stop]
        return result

    @staticmethod
    def batch(
        proteins: Sequence[Mapping[str, np.ndarray]], atom_counts: Sequence[int],
    ) -> dict[str, np.ndarray]:
        """Concatenate validated CSR candidates into explicitly disjoint CPU index namespaces.

        Args:
            proteins: One validated CSR mapping per protein, in caller-selected order.
            atom_counts: Exact base atom count per protein, including atoms absent from its
                surface neighborhoods; never infer offsets from the largest candidate ID.

        Returns:
            Concatenated CSR arrays with int64 global atom indices/pointers, plus atom_ptr and
            surface_ptr protein boundaries. No historical collator or model calls this opt-in
            method automatically. Geometry and support counts are not modified.

        Raises:
            ValueError: Empty/mismatched input or a candidate outside its own protein.
        """
        if not proteins or len(proteins) != len(atom_counts):
            raise ValueError("CSR batch requires matching protein mappings and base atom counts")
        atom_ptr = np.concatenate(([0], np.cumsum(atom_counts, dtype=np.int64)))
        surface_ptr = np.concatenate(([0], np.cumsum(
            [len(value["point_ptr"]) - 1 for value in proteins], dtype=np.int64)))
        pointers = [np.asarray([0], np.int64)]
        indices = []
        pair_offset = 0
        for i, value in enumerate(proteins):
            local = value["atom_index"]
            if np.any(local < 0) or np.any(local >= atom_counts[i]):
                raise ValueError("CSR candidate crosses its protein's atom namespace")
            indices.append(local.astype(np.int64) + atom_ptr[i])
            pointers.append(value["point_ptr"][1:] + pair_offset)
            pair_offset += len(local)
        return {
            "atom_ptr": atom_ptr, "surface_ptr": surface_ptr,
            "point_ptr": np.concatenate(pointers), "atom_index": np.concatenate(indices),
            **{name: np.concatenate([value[name] for value in proteins]) for name in (
                "distance", "normal_offset", "tangential", "candidate_count", "discarded_count")},
        }
