"""Per-protein scientific transform for LambdaForge preprocessing."""

from __future__ import annotations

import os
import time

from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.StructureCache import StructureCache


class PreprocessPipeline:
    """Convert one manifest record into WISDOM arrays while LambdaForge owns orchestration."""

    def __init__(
        self,
        config          : PreprocessConfig,
        identifier_input: str  = "protein_identifiers",
        download_output : str  = "downloads",
        download        : bool = True,
    ) -> None:
        """Bind scientific settings and logical LambdaForge path names.

        Args:
            config: Settings that determine protein selection, graphs, surface and curvature.
            identifier_input: Named task input containing the protein TXT manifest.
            download_output: Named task output used as the race-safe RCSB structure cache.
            download: Whether a missing remote PDB entry may be downloaded.

        Raises:
            ValueError: If a logical input/output name is empty.
        """
        if not identifier_input.strip() or not download_output.strip():
            raise ValueError("logical input and output names cannot be empty")

        self.config           = config
        self.identifier_input = identifier_input
        self.download_output  = download_output
        self.download         = download

    def transform(
        self,
        record : ProcessingRecord,
        context: ProcessingWorkspace,
    ) -> ProcessingRecord:
        """Build one complete validated representation without writing dataset state.

        LambdaForge invokes this method sequentially, in threads, or in spawn-safe CPU processes.
        The method resolves/downloads one structure, reads its hierarchy with Gemmi, builds the
        sparse atomic graph and molecular surface, and prepares NPZ metadata. It intentionally
        leaves publication and aggregate reporting to ``ProteinSink`` while LambdaForge Work owns
        map concurrency, progress, retries, and checkpoints.

        Args:
            record: Stable-key record from ``ProteinSource`` whose value is one manifest line.
            context: LambdaForge context resolving named input/output locations in this attempt.

        Returns:
            A record with the same key and metadata whose value contains scientific arrays,
            provenance, output filename and human-readable scale diagnostics for ``ProteinSink``.

        Raises:
            TypeError: If the source value or output-name metadata violates the source contract.
            ValueError: If record grammar, parsing, scientific construction or validation fails.
            OSError: If source download/read or numerical construction cannot access its files.
        """
        if not isinstance(record.value, str):
            raise TypeError("protein preprocessing records must contain an identifier string")
        output_name = record.metadata.get("output_name")
        if not isinstance(output_name, str):
            raise TypeError("protein preprocessing record metadata requires output_name")

        # Numerical libraries must not create nested native pools inside LambdaForge CPU workers.
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"

        # Heavy scientific modules are imported after native thread limits are established.
        from wisdom.preprocessing.structure.AtomicStructureBuilder import AtomicStructureBuilder
        from wisdom.preprocessing.structure.ProteinReader import ProteinReader
        from wisdom.preprocessing.structure.StorageManager import StorageManager
        from wisdom.preprocessing.structure.SurfaceBuilder import SurfaceBuilder

        started      = time.perf_counter()
        manifest_dir = context.input(self.identifier_input).parent
        download_dir = context.output(self.download_output)
        download_dir.mkdir(parents=True, exist_ok=True)

        source = StructureCache(download_dir, self.download).resolve(record.value, manifest_dir)
        if source.is_local:
            # Dynamic paths inside the manifest still require declared-input coverage. LambdaForge
            # has no logical name for an arbitrary line, so this is the justified legacy helper.
            context.declared_input_path(source.path)

        # The central scientific path remains deliberately readable as executable pseudocode.
        protein, protein_metadata = ProteinReader(self.config).read(source)

        arrays = AtomicStructureBuilder(self.config.atom_radius).build(protein)

        surface_arrays, warnings = SurfaceBuilder(
            resolution=self.config.surface_resolution,
            probe_radius=self.config.probe_radius,
            atom_radius=self.config.atom_surface_radius,
            curvature_scales=self.config.curvature_scales,
        ).build(arrays["atom_positions"], arrays["vdw_radii"])

        duplicate_names = arrays.keys() & surface_arrays.keys()
        if duplicate_names:
            raise ValueError(f"duplicate array names: {sorted(duplicate_names)}")
        arrays.update(surface_arrays)

        storage  = StorageManager(self.config)
        metadata = storage.make_metadata(protein, protein_metadata, arrays, warnings)

        # Record structural scale without using wall time as a scientific performance claim.
        atom_count    = sum(
            len(residue.atoms) for chain in protein.chains for residue in chain.residues
        )
        residue_count = sum(len(chain.residues) for chain in protein.chains)
        report = {
            "identifier": source.identifier,
            "status": "processed",
            "output": output_name,
            "atom_count": atom_count,
            "residue_count": residue_count,
            "atom_edge_count": int(arrays["atom_edge_index"].shape[1]),
            "surface_point_count": int(arrays["surface_positions"].shape[0]),
            "surface_edge_count": int(arrays["surface_edge_index"].shape[1]),
            "surface_atom_edge_count": int(arrays["surface_atom_edge_index"].shape[1]),
            "array_bytes": sum(array.nbytes for array in arrays.values()),
            "seconds": time.perf_counter() - started,
            "warnings": warnings,
        }
        return record.with_value(
            {
                "arrays": arrays,
                "metadata": metadata,
                "output_name": output_name,
                "report": report,
            }
        )

    def process(
        self,
        record : ProcessingRecord,
        context: ProcessingWorkspace,
    ) -> ProcessingRecord:
        """Build, validate, and persist one protein as a JSON-checkpointable result.

        LambdaForge 0.12 requires ``Work.map`` results to be JSON-compatible. Scientific arrays
        are therefore written atomically by the worker to the checkpoint-owned geometry
        directory, while only the compact report returns through the framework map. A resumed map
        reuses that report and its durable NPZ instead of serializing large NumPy arrays.

        Args:
            record: Stable protein identifier and collision-safe output filename.
            context: Explicit manifest, download-cache, and processed-output paths.

        Returns:
            Record whose value is the JSON-compatible per-protein processing report.

        Raises:
            TypeError: If the source or transformed record violates its contract.
            ValueError: If scientific construction or NPZ validation fails.
            OSError: If coordinate acquisition or atomic NPZ persistence fails.
        """
        from wisdom.preprocessing.structure.ProteinSink import ProteinSink

        sink        = ProteinSink(
            identifier_input=self.identifier_input,
            dataset_output="processed",
            report_output="report",
        )
        resumed = sink.resume(record, context, self.config)
        if resumed is not None:
            return resumed

        transformed = self.transform(record, context)
        sink.write(transformed, context)
        return record.with_value(sink.records[record.key])
