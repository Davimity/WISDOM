"""LambdaForge 0.12 Work for WISDOM-DNA geometry, annotation, and publication."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import lambdaforge as lf
from lambdaforge.data import DatasetIndex

from wisdom.preprocessing.dna.DNAAnnotationSink import DNAAnnotationSink
from wisdom.preprocessing.dna.DNAAnnotationSource import DNAAnnotationSource
from wisdom.preprocessing.dna.DNAAnnotationTransform import DNAAnnotationTransform
from wisdom.preprocessing.dna.DNAPartitionTask import DNAPartitionTask
from wisdom.preprocessing.dna.DNAValidation import DNAValidation
from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessPipeline import PreprocessPipeline
from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource


class Preprocessing(lf.Work):
    """Build and publish the complete immutable WISDOM-DNA DatasetVersion."""

    def run(
        self,
        curation                    : Path,
        dataset_name                : str = "wisdom-dna",
        dataset_version             : str = "3",
        workers                     : int = 36,
        threads                     : int = 36,
        surface_resolution          : float = 1.0,
        probe_radius                : float = 1.4,
        atom_radius                 : float = 6.0,
        atom_surface_radius         : float = 6.0,
        curvature_scales            : Sequence[float] = (2.5, 5.0),
        positive_gap                : float = 1.4,
        negative_gap                : float = 3.0,
        sensitivity_gaps            : Sequence[float] = (1.0, 1.4, 2.0),
        sequence_identity           : float = 0.30,
        sequence_coverage           : float = 0.80,
        sequence_evalue             : float = 1e-3,
        structure_probability       : float = 0.50,
        structure_evalue            : float = 1e-3,
        train_fraction              : float = 0.70,
        validation_fraction         : float = 0.15,
        test_fraction               : float = 0.15,
        phenotype_min_cluster_size  : int = 15,
        phenotype_min_samples       : int = 5,
        phenotype_stability_minimum : float = 0.60,
        dilution_sizes              : Sequence[int] = (400, 200, 100, 75, 50, 25),
        partition_seed              : int = 2026,
        mmseqs_executable           : str = "mmseqs",
        foldseek_executable         : str = "foldseek",
    ) -> dict[str, Any]:
        """Generate geometry and DNA sidecars, partition, validate, and publish the dataset.

        The Work consumes a frozen split-free curation. LambdaForge ``self.map`` distributes
        independent geometry and annotation records across spawned CPU workers. Each worker writes
        validated scientific files into Work checkpoints and returns a JSON audit row, preventing
        large NumPy arrays from crossing process boundaries. Dataset-level specialist tools then
        form leakage groups and phenotype strata, validation audits every member, and
        ``self.outputs.dataset`` performs the only immutable publication.

        Args:
            curation: Curation artifact with ``curated-catalog.csv`` and
                ``curated-proteins.txt``.
            dataset_name: Stable LambdaForge Registry dataset name.
            dataset_version: Immutable content release label.
            workers: Spawned CPU workers for independent geometry and annotation records.
            threads: Threads passed to MMseqs2, Foldseek, and HDBSCAN.
            surface_resolution: Target solvent-surface spacing in ångströms.
            probe_radius: Solvent probe radius in ångströms.
            atom_radius: Sparse atom-graph cutoff in ångströms.
            atom_surface_radius: Sparse surface-to-atom cutoff in ångströms.
            curvature_scales: Positive geodesic curvature-fit radii in ångströms.
            positive_gap: Largest DNA-to-surface gap considered confidently positive in Å.
            negative_gap: Smallest gap considered confidently negative in Å.
            sensitivity_gaps: Alternative evaluation-only positive cutoffs in ångströms.
            sequence_identity: Minimum bilateral MMseqs2 pair identity fraction.
            sequence_coverage: Minimum aligned fraction of both sequences.
            sequence_evalue: Largest accepted MMseqs2 expectation value.
            structure_probability: Minimum Foldseek homology probability.
            structure_evalue: Largest accepted Foldseek expectation value.
            train_fraction: Target training membership fraction.
            validation_fraction: Target model-development membership fraction.
            test_fraction: Target final-evaluation membership fraction.
            phenotype_min_cluster_size: Smallest HDBSCAN physical cluster.
            phenotype_min_samples: HDBSCAN core-neighbour parameter.
            phenotype_stability_minimum: Median parameter-grid ARI required for phenotype labels.
            dilution_sizes: Absolute sizes for nested training-only subsets.
            partition_seed: Stable group-assignment and dilution-order seed.
            mmseqs_executable: Required MMseqs2 executable name or path.
            foldseek_executable: Required Foldseek executable name or path.

        Returns:
            Published dataset record plus geometry, partition, and validation summaries.

        Raises:
            ValueError: If curation or scientific parameters violate their contracts.
            RuntimeError: If processing, specialist tools, validation, or publication fails.
            OSError: If an immutable input or atomic output cannot be accessed.
        """
        curation = curation.resolve()
        required = ("curated-catalog.csv", "curated-proteins.txt")
        missing  = [name for name in required if not (curation / name).is_file()]
        if missing:
            raise ValueError(f"curation artifact is incomplete; missing {missing}")
        if not dataset_name.strip() or not dataset_version.strip() or min(workers, threads) < 1:
            raise ValueError("dataset identity must be non-empty and concurrency must be positive")

        # Geometry files live under compatible Work checkpoints. Map results stay small and safe,
        # while reruns can validate and reuse expensive per-protein archives.
        geometry_root = self.checkpoints.path("geometry")
        geometry       = ProcessingWorkspace(
            self.run_dir,
            inputs={"protein_identifiers": curation / "curated-proteins.txt"},
            outputs={
                "downloads": self.cache.path("structures"),
                "processed": geometry_root / "processed",
                "report": geometry_root / "preprocessing-report.json",
            },
        )
        protein_records = tuple(ProteinSource().records(geometry))
        protein_pipeline = PreprocessPipeline(
            config=PreprocessConfig(
                atom_radius=atom_radius,
                surface_resolution=surface_resolution,
                probe_radius=probe_radius,
                atom_surface_radius=atom_surface_radius,
                curvature_scales=tuple(curvature_scales),
            ),
        )
        geometry_results = self.map(
            protein_records,
            partial(protein_pipeline.process, context=geometry),
            key=lambda record: record.key,
            workers=workers,
            executor="process",
            resume=False,
            name="protein-geometry",
        )

        geometry_sink = ProteinSink(download_output="downloads")
        geometry_sink.records = {
            record.key: dict(record.value)
            for value in geometry_results
            for record in (ProcessingRecord.restore(value),)
        }
        geometry_sink.finalize(geometry)

        # Annotation workers consume exact curation and geometry provenance. Sidecars remain
        # separate from universal NPZ files so supervised targets cannot alter base geometry.
        annotation_root = self.checkpoints.path("annotation")
        annotation      = ProcessingWorkspace(
            self.run_dir,
            inputs={
                "curation": curation,
                "dataset_catalog": curation / "curated-catalog.csv",
                "base_preprocessing_report": geometry_root / "preprocessing-report.json",
                "structure_cache": self.cache.path("structures"),
            },
            outputs={
                "annotations": annotation_root / "annotations",
                "annotation-report": annotation_root / "annotations/annotation-report.json",
                "resolved-structures": annotation_root / "resolved-structures",
            },
        )
        annotation_records = tuple(DNAAnnotationSource().records(annotation))
        annotation_pipeline = DNAAnnotationTransform(
            positive_gap=positive_gap,
            negative_gap=negative_gap,
            sensitivity_gaps=tuple(sensitivity_gaps),
            structure_output="resolved-structures",
        )
        annotation_results = self.map(
            annotation_records,
            partial(annotation_pipeline.process, context=annotation),
            key=lambda record: record.key,
            workers=workers,
            executor="process",
            resume=False,
            name="dna-annotations",
        )

        annotation_sink = DNAAnnotationSink(
            annotation_output="annotations",
            report_output="annotation-report",
            curation_input="curation",
        )
        annotation_sink.records = {
            record.key: dict(record.value)
            for value in annotation_results
            for record in (ProcessingRecord.restore(value),)
        }
        annotation_sink.finalize(annotation)

        # Dataset-level tools operate on one attempt-owned copy because partitioning adds the final
        # index and evidence files that LambdaForge will fingerprint and publish atomically.
        final_root = self.run_dir / "dataset"
        if final_root.exists():
            shutil.rmtree(final_root)
        shutil.copytree(annotation_root / "annotations", final_root)
        partition = DNAPartitionTask(
            sequence_identity=sequence_identity,
            sequence_coverage=sequence_coverage,
            sequence_evalue=sequence_evalue,
            structure_probability=structure_probability,
            structure_evalue=structure_evalue,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            phenotype_min_cluster_size=phenotype_min_cluster_size,
            phenotype_min_samples=phenotype_min_samples,
            phenotype_stability_minimum=phenotype_stability_minimum,
            dilution_sizes=dilution_sizes,
            seed=partition_seed,
            threads=threads,
            mmseqs_executable=mmseqs_executable,
            foldseek_executable=foldseek_executable,
        ).run(final_root)

        validation = DNAValidation().audit(final_root, self.run_dir / "dna-validation")
        if validation["verdict"] != "PASS":
            raise RuntimeError("final WISDOM-DNA scientific validation failed before publication")

        members = tuple(self.published_members(final_root))
        record  = self.outputs.dataset(
            name=dataset_name,
            version=dataset_version,
            members=members,
            output="dataset",
            metadata={
                "description": "Leakage-safe DNA-binding proteins with universal WISDOM surfaces",
                "structural_schema": "2.1",
                "annotation_schema": "1.2",
                "partition_schema": "2.0",
                "supervision": "protein-level-only",
            },
            target_schema={
                "type": "object",
                "properties": {
                    "dna_binding": {"type": "integer", "enum": [0, 1]},
                    "local_ground_truth": {"type": "boolean"},
                },
                "required": ["dna_binding", "local_ground_truth"],
                "additionalProperties": False,
            },
        )
        self.outputs.artifact(
            "preprocessing-report",
            geometry_root / "preprocessing-report.json",
            role="report",
        )
        self.outputs.artifact(
            "annotation-report",
            final_root / "annotation-report.json",
            role="report",
        )
        self.outputs.artifact(
            "partition-report",
            final_root / "partition-report.json",
            role="report",
        )
        self.outputs.artifact("validation-report", self.run_dir / "dna-validation", role="report")
        self.metrics.log("published_members", len(members))
        return {
            **record,
            "published_members": len(members),
            "leakage_groups": partition["leakage_group_count"],
            "validation_verdict": validation["verdict"],
        }

    def published_members(self, final_root: Path) -> Iterable[Mapping[str, Any]]:
        """Translate a validated WISDOM index into LambdaForge dataset member mappings.

        Args:
            final_root: Attempt-owned final dataset root containing ``members.jsonl`` and assets.

        Yields:
            Member mappings accepted by :meth:`lambdaforge.Work.outputs.dataset`.

        Raises:
            OSError: If the validated member index cannot be read.
            ValueError: If an index member references malformed metadata or assets.
        """
        for index, member in enumerate(DatasetIndex(final_root / "members.jsonl")):
            assets = {name: final_root / asset.path for name, asset in member.assets.items()}
            if index == 0:
                assets["dataset_evidence"] = final_root / "evidence"
            yield {
                "id": member.member_id,
                "partitions": dict(member.partitions),
                "targets": dict(member.targets),
                "metadata": dict(member.metadata),
                "display": dict(member.display),
                "assets": assets,
            }
