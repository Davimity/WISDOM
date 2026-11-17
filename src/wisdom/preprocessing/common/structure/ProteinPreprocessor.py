"""Per-protein construction of universal WISDOM geometry."""

from __future__ import annotations

import os
import time
import numpy as np

from typing import Any
from pathlib import Path
from collections.abc import Mapping
from wisdom.utils.structure.AtomicDescriptors import AtomicDescriptors
from wisdom.preprocessing.common.structure.io.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.storage.ProteinSink import ProteinSink
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.geometry.AtomicStructureBuilder import (
    AtomicStructureBuilder,
)
from wisdom.preprocessing.common.structure.geometry.DiffusionOperatorBuilder import (
    DiffusionOperatorBuilder,
)
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.geometry.SurfaceBuilder import SurfaceBuilder
from wisdom.preprocessing.common.structure.io.StructureResolver import StructureResolver
from wisdom.preprocessing.common.structure.geometry.SurfaceAtomNeighborhoodBuilder import (
    SurfaceAtomNeighborhoodBuilder,
)


class ProteinPreprocessor:
    """Convert one source record into a validated universal NPZ."""

    def __init__(self, config: PreprocessConfig) -> None:
        """Store the scientific configuration used by every protein.

        Args:
            config: Atom selection, graph, surface, and curvature settings.
        """
        self.config = config

    def transform(
        self,
        record        : Mapping[str, Any],
        manifest      : Path,
        structure_root: Path,
    ) -> dict[str, Any]:
        """Build all universal arrays for one manifest record without writing them.

        Args:
            record: JSON-compatible source mapping created by ``ProteinSource``.
            manifest: TXT input used to resolve relative local structure paths.
            structure_root: Directory containing exact Selection ``.cif.gz`` snapshots.

        Returns:
            Mapping carrying arrays, provenance, output name, and a concise scientific report.

        Raises:
            TypeError: If the source record lacks its identifier or output filename.
            ValueError: If parsing, graph construction, or surface construction fails.
            OSError: If coordinate files cannot be read.
        """
        key         = str(record["key"])
        identifier  = record.get("identifier")
        output_name = record.get("output_name")
        if not isinstance(identifier, str) or not isinstance(output_name, str):
            raise TypeError("protein source records require identifier and output_name")

        # Native numerical libraries must not create nested pools inside each process worker.

        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"

        started = time.perf_counter()
        timings: dict[str, float] = {}
        stage = time.perf_counter()
        source  = StructureResolver(structure_root).resolve(identifier, manifest.parent)
        timings["source_resolution"] = time.perf_counter() - stage

        # The scientific path deliberately reads as a direct sequence of domain transformations.

        stage = time.perf_counter()
        protein, provenance = ProteinReader(self.config).read(source)
        timings["read_filter_normalize"] = time.perf_counter() - stage

        stage = time.perf_counter()
        arrays = AtomicStructureBuilder(
            self.config.atom_spatial_radius,
            self.config.atom_spatial_k_max,
        ).build(protein)
        timings["atomic_graph"] = time.perf_counter() - stage

        # Derive generic chemistry once during preprocessing. Persisting these fixed descriptors
        # avoids repeating graph/name analysis every time HPO opens the same protein in a new epoch.

        stage = time.perf_counter()
        descriptors = AtomicDescriptors.derive(
            arrays["atomic_numbers"],
            arrays["atom_names"],
            arrays["residue_names"],
            arrays["formal_charges"],
            arrays["atom_edge_index"],
            arrays["atom_edge_bond_order"],
            arrays["atom_edge_is_covalent"],
        )
        arrays.update(
            {
                name: values
                for name, values in descriptors.items()
                if name != "formal_charges"
            }
        )
        timings["chemical_descriptors"] = time.perf_counter() - stage

        surface = SurfaceBuilder(
            resolution       = self.config.surface_resolution,
            probe_radius     = self.config.probe_radius,
            curvature_scales = self.config.curvature_scales,
        )
        surface_arrays, warnings = surface.build(arrays["atom_positions"], arrays["vdw_radii"])
        timings.update(surface.timings)

        duplicate_names = arrays.keys() & surface_arrays.keys()
        if duplicate_names:
            raise ValueError(f"duplicate array names: {sorted(duplicate_names)}")
        arrays.update(surface_arrays)

        # Store one bounded nearest-atom table whose prefixes realize every runtime J choice.

        stage = time.perf_counter()
        transfer_arrays = SurfaceAtomNeighborhoodBuilder(
            radius        = self.config.surface_atom_radius,
            max_neighbors = self.config.surface_atom_k_max,
        ).build(
            arrays["atom_positions"],
            arrays["surface_positions"],
            arrays["surface_normals"],
        )
        arrays.update(transfer_arrays)
        timings["surface_atom_neighbors"] = time.perf_counter() - stage

        # Precompute physical sparse surface operators once; training learns only diffusion times.

        diffusion = DiffusionOperatorBuilder(
            resolution     = self.config.surface_resolution,
            spectral_modes = self.config.diffusion_spectral_modes_max,
            max_neighbors  = self.config.surface_neighbor_k_max,
        )
        diffusion_arrays = diffusion.build(arrays["surface_positions"], arrays["surface_normals"])
        arrays.update(diffusion_arrays)
        timings.update(diffusion.timings)

        archive  = ProteinArchive(self.config)
        metadata = archive.make_metadata(protein, provenance, arrays, warnings)

        atom_count    = sum(
            len(residue.atoms) for chain in protein.chains for residue in chain.residues
        )
        residue_count = sum(len(chain.residues) for chain in protein.chains)
        report        = {
            "identifier":              source.identifier,
            "status":                  "processed",
            "output":                  output_name,
            "atom_count":              atom_count,
            "residue_count":           residue_count,
            "atom_edge_count":         int(arrays["atom_edge_index"].shape[1]),
            "surface_point_count":     int(arrays["surface_positions"].shape[0]),
            "atom_spatial_candidate_count": int(
                np.count_nonzero(arrays["atom_edge_spatial_rank"])
            ),
            "surface_atom_neighbor_count": int(arrays["surface_atom_mask"].sum()),
            "diffusion_spectral_modes": len(arrays["diffusion_eigenvalues"]),
            "diffusion_gradient_entries": len(arrays["diffusion_gradient_x"]),
            "array_bytes":             sum(array.nbytes for array in arrays.values()),
            "seconds":                 time.perf_counter() - started,
            "stage_seconds":           timings,
            "warnings":                warnings,
        }
        return {
            "key":         key,
            "arrays":      arrays,
            "metadata":    metadata,
            "output_name": output_name,
            "report":      report,
        }

    def process(
        self,
        record        : Mapping[str, Any],
        manifest      : Path,
        structure_root: Path,
        output_root   : Path,
    ) -> dict[str, Any]:
        """Resume or build one NPZ and return only JSON-compatible report data.

        Args:
            record: Stable source mapping from ``ProteinSource``.
            manifest: TXT manifest used for local-path resolution.
            structure_root: Directory containing verified coordinate archives.
            output_root: Checkpoint-owned directory receiving NPZ files.

        Returns:
            Mapping with stable ``key`` and compact report under ``value``.

        Raises:
            TypeError: If the source or transformed record violates its contract.
            ValueError: If scientific construction or NPZ validation fails.
            OSError: If coordinate reading or atomic publication fails.
        """
        sink    = ProteinSink()
        resumed = sink.resume(record, output_root, self.config)
        if resumed is not None:
            return resumed

        transformed = self.transform(record, manifest, structure_root)
        sink.write(transformed, output_root)

        key = str(record["key"])
        return {"key": key, "value": sink.records[key]}
