"""Generate WISDOM geometry/annotations for one fixed DatasetDesign artifact."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import gemmi
import lambdaforge as lf
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.work import ManagedFile, RateLimit

from wisdom.preprocessing.dna.DNAAnnotationSink import DNAAnnotationSink
from wisdom.preprocessing.dna.DNAAnnotationSource import DNAAnnotationSource
from wisdom.preprocessing.dna.DNAAnnotationTransform import DNAAnnotationTransform
from wisdom.preprocessing.dna.DNAValidation import DNAValidation
from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessPipeline import PreprocessPipeline
from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource


class Preprocessing(lf.Work):
    """Generate geometry/sidecars and publish the immutable designed WISDOM-DNA dataset."""

    _download_retries    : int
    _structure_rate_limit: RateLimit

    def run(
        self,
        design              : Path,
        dataset_name        : str             = "wisdom-dna",
        dataset_version     : str             = "4",
        workers             : int             = 36,
        requests_per_second : float           = 4.0,
        retries             : int             = 5,
        surface_resolution  : float           = 1.0,
        probe_radius        : float           = 1.4,
        atom_radius         : float           = 6.0,
        atom_surface_radius : float           = 6.0,
        curvature_scales    : Sequence[float] = (2.5, 5.0),
        positive_gap        : float           = 1.4,
        negative_gap        : float           = 3.0,
        sensitivity_gaps    : Sequence[float] = (1.0, 1.4, 2.0),
    ) -> dict[str, Any]:
        """Process only canonical members and publish their fixed design metadata.

        DatasetDesign already fixed balancing, full-raw leakage groups, physical phenotypes,
        train/validation/test, and nested training dilutions. This Work cannot recompute those
        decisions. LambdaForge ``self.map`` distributes independent geometry and annotation
        records across spawned workers. Each WISDOM sink revalidates source/configuration hashes,
        schemas, and numerical invariants before resuming a large NPZ; the framework map therefore
        remains intentionally stateless and cannot bypass this stricter scientific boundary.

        Args:
            design: Exact ``dataset-design`` artifact containing ``catalog.csv``, fixed splits,
                dilution manifests, pair evidence, statistics, and provenance.
            dataset_name: Stable LambdaForge Dataset Registry name.
            dataset_version: New immutable version; conflicting bytes are never overwritten.
            workers: Spawned CPU workers for independent geometry and annotation records.
            requests_per_second: Aggregate RCSB request-start limit for missing structures.
            retries: Additional LambdaForge download attempts after the first failure.
            surface_resolution: Target solvent-surface point spacing in ångströms.
            probe_radius: Solvent probe radius added to atom radii in ångströms.
            atom_radius: Sparse atom-graph neighborhood cutoff in ångströms.
            atom_surface_radius: Sparse surface-to-atom communication cutoff in ångströms.
            curvature_scales: Positive fit radii as multiples of surface resolution.
            positive_gap: Largest DNA-to-surface gap confidently positive in ångströms.
            negative_gap: Smallest DNA-to-surface gap confidently negative in ångströms.
            sensitivity_gaps: Additional positive cutoffs retained for evaluation sensitivity.

        Returns:
            Published dataset record, member/group counts, and validation verdict.

        Raises:
            ValueError: If design, identity, concurrency, or scientific parameters fail.
            RuntimeError: If geometry, annotation, validation, or publication fails.
            OSError: If an immutable input or atomic output cannot be read or written.
        """
        design = design.resolve()
        required = (
            "catalog.csv",
            "REPORT.md",
            "selected.fasta",
            "proteins.txt",
            "proteins-labelled.txt",
            "train.txt",
            "train-labelled.txt",
            "validation.txt",
            "validation-labelled.txt",
            "test.txt",
            "test-labelled.txt",
            "provenance.json",
        )
        missing = [name for name in required if not (design / name).is_file()]
        if missing:
            raise ValueError(f"dataset design artifact is incomplete; missing {missing}")
        if not dataset_name.strip() or not dataset_version.strip():
            raise ValueError("dataset name and version cannot be empty")
        if isinstance(workers, bool) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if workers > int(self.resources.cpu):
            raise ValueError(
                f"workers={workers} exceeds LambdaForge cpu allocation={self.resources.cpu}"
            )

        catalog = self._catalog(design / "catalog.csv")
        identifiers = [str(row["identifier"]) for row in catalog]

        # LambdaForge downloads every unique selected PDB once before CPU-heavy geometry starts.
        # resume_map stores logical ManagedFile references, so retries restore byte-verified cache
        # entries without persisting machine-specific cache paths as scientific identity.
        self._download_retries     = retries
        self._structure_rate_limit = self.cache.rate_limit(
            "rcsb-preprocessing",
            requests_per_second=requests_per_second,
        )
        structure_hashes: dict[str, set[str]] = {}
        for row in catalog:
            pdb_id = str(row["pdb_id"]).lower()
            structure_hashes.setdefault(pdb_id, set()).add(str(row["structure_sha256"]))
        conflicts = sorted(
            pdb_id for pdb_id, hashes in structure_hashes.items() if len(hashes) != 1
        )
        if conflicts:
            raise ValueError(f"design assigns conflicting structure hashes to PDBs: {conflicts}")
        structure_jobs = [
            {"pdb_id": pdb_id, "expected_sha256": next(iter(structure_hashes[pdb_id]))}
            for pdb_id in sorted(structure_hashes)
        ]
        print(
            f"[Preprocessing] Fetching or restoring {len(structure_jobs)} unique structures "
            f"with {workers} workers",
            flush=True,
        )
        structures = self.resume_map(
            structure_jobs,
            self._fetch_structure,
            key="pdb_id",
            workers=workers,
            executor="thread",
            name="preprocessing-structures",
        )
        structure_roots = {Path(structure).parent for structure in structures}
        if len(structure_roots) != 1:
            raise RuntimeError("managed structures do not share one LambdaForge cache directory")
        structure_root = structure_roots.pop()

        manifest = self.checkpoints.path("geometry/protein-identifiers.txt")
        self._atomic_text(manifest, "".join(f"{identifier}\n" for identifier in identifiers))

        print(
            f"[Preprocessing] Generating geometry for {len(catalog)} designed proteins "
            f"with {workers} workers",
            flush=True,
        )
        geometry_root = self.checkpoints.path("geometry")
        geometry = ProcessingWorkspace(
            self.run_dir,
            inputs={
                "protein_identifiers": manifest,
                "structures": structure_root,
            },
            outputs={
                "processed": geometry_root / "processed",
                "report": geometry_root / "preprocessing-report.json",
            },
        )
        protein_records = tuple(ProteinSource().records(geometry))
        if {record.key for record in protein_records} != set(identifiers):
            raise ValueError("design catalog and geometry manifest disagree")
        pipeline = PreprocessPipeline(
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
            partial(pipeline.process, context=geometry),
            workers=workers,
            executor="process",
            name="protein-geometry",
        )
        geometry_sink = ProteinSink()
        geometry_sink.records = {
            record.key: dict(record.value)
            for value in geometry_results
            for record in (ProcessingRecord(value),)
        }
        geometry_sink.finalize(geometry)

        print(f"[Preprocessing] Projecting {len(catalog)} DNA target sidecars", flush=True)
        annotation_root = self.checkpoints.path("annotation")
        annotation = ProcessingWorkspace(
            self.run_dir,
            inputs={
                "dataset_design": design,
                "dataset_catalog": design / "catalog.csv",
                "base_preprocessing_report": geometry_root / "preprocessing-report.json",
                "structure_cache": structure_root,
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
            workers=workers,
            executor="process",
            name="dna-annotations",
        )
        annotation_sink = DNAAnnotationSink(
            annotation_output="annotations",
            report_output="annotation-report",
            design_input="dataset_design",
        )
        annotation_sink.records = {
            record.key: dict(record.value)
            for value in annotation_results
            for record in (ProcessingRecord(value),)
        }
        annotation_sink.finalize(annotation)

        # Arrays and the unchanged complete design meet at the only publication boundary.
        final_root = self.run_dir / "dataset"
        if final_root.is_symlink():
            raise ValueError("final dataset root cannot be a symlink")
        if final_root.exists():
            shutil.rmtree(final_root)
        shutil.copytree(annotation_root / "annotations", final_root)
        shutil.copytree(design, final_root / "design")
        self._write_index(final_root, catalog)

        validation = DNAValidation().audit(final_root, self.run_dir / "dna-validation")
        if validation["verdict"] != "PASS":
            raise RuntimeError("final WISDOM-DNA scientific validation failed before publication")

        members = tuple(self.published_members(final_root))
        record = self.outputs.dataset(
            name=dataset_name,
            version=dataset_version,
            members=members,
            output="dataset",
            metadata={
                "description": "Designed leakage-safe DNA proteins with WISDOM surfaces",
                "structural_schema": "2.1",
                "annotation_schema": "1.3",
                "design_schema": "1.2",
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
        self.outputs.artifact("validation-report", self.run_dir / "dna-validation", role="report")
        self.metrics.log("published_members", len(members))
        return {
            **record,
            "published_members": len(members),
            "leakage_groups": len({str(row["leakage_group"]) for row in catalog}),
            "validation_verdict": validation["verdict"],
        }

    def _fetch_structure(self, job: Mapping[str, Any]) -> ManagedFile:
        """Fetch one selected RCSB entry through LambdaForge's managed cache.

        Args:
            job: JSON-compatible mapping containing a lowercase ``pdb_id`` and the SHA-256 of the
                uncompressed mmCIF bytes fixed by DatasetDesign.

        Returns:
            Read-only managed ``.cif.gz`` file whose bytes passed a Gemmi parse check. The logical
            cache key, content digest, and byte count are recorded as a ``resume_map`` dependency.

        Raises:
            RuntimeError: If LambdaForge exhausts all HTTP attempts or validation keeps failing.
            ValueError: If the supplied PDB identifier or cache parameters are invalid.
        """
        pdb_id         = str(job["pdb_id"]).lower()
        expected_hash = str(job["expected_sha256"]).lower()
        if not pdb_id.isalnum():
            raise ValueError(f"invalid RCSB PDB identifier: {pdb_id!r}")
        if len(expected_hash) != 64 or not set(expected_hash) <= set("0123456789abcdef"):
            raise ValueError(f"invalid design structure SHA-256 for {pdb_id}")

        return self.cache.fetch(
            f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz",
            key        = f"structures/{pdb_id}.cif.gz",
            retries    = self._download_retries,
            timeout    = 180.0,
            validate   = partial(
                self._valid_structure_archive,
                expected_sha256=expected_hash,
            ),
            rate_limit = self._structure_rate_limit,
        )

    @staticmethod
    def _valid_structure_archive(file: ManagedFile, expected_sha256: str) -> bool:
        """Check a managed gzip archive against the exact structure used by design.

        Args:
            file: LambdaForge-managed RCSB ``.cif.gz`` candidate.
            expected_sha256: DatasetDesign digest of the uncompressed mmCIF bytes.

        Returns:
            ``True`` when Gemmi parses a non-empty structure and decompression reproduces the
            design digest; ``False`` asks LambdaForge to discard and rebuild the entry atomically.
        """
        try:
            digest = hashlib.sha256()
            with gzip.open(file, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return bool(gemmi.read_structure(str(file))) and digest.hexdigest() == expected_sha256
        except (OSError, RuntimeError, ValueError):
            return False

    @staticmethod
    def _catalog(path: Path) -> list[dict[str, Any]]:
        """Read and validate the fixed canonical design catalog.

        Args:
            path: DatasetDesign ``catalog.csv``.

        Returns:
            Identifier-sorted rows with nested JSON fields and scalar types restored.

        Raises:
            ValueError: If required fields, IDs, labels, splits, groups, or selected flags fail.
        """
        required = {
            "identifier",
            "label",
            "split",
            "selected",
            "leakage_group",
            "global_phenotype",
            "interface_phenotype",
            "origin",
            "label_evidence",
            "pdb_id",
            "protein_chain",
            "assembly_id",
            "protein_copy",
            "structure_sha256",
            "dna_chains",
            "binding_residue_indices",
            "local_gt_expected",
            "local_gt_method",
            "assembly_rotation",
            "assembly_translation",
        }
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            if not required.issubset(fields):
                raise ValueError(f"design catalog lacks fields: {sorted(required - fields)}")
            for line, raw in enumerate(reader, start=2):
                row: dict[str, Any] = dict(raw)
                try:
                    row["label"] = int(raw["label"])
                    row["protein_copy"] = int(raw["protein_copy"])
                    row["selected"] = str(raw["selected"]).lower() == "true"
                    row["local_gt_expected"] = str(raw["local_gt_expected"]).lower() == "true"
                    for field in (
                        "dna_chains",
                        "binding_residue_indices",
                        "assembly_rotation",
                        "assembly_translation",
                    ):
                        row[field] = json.loads(raw[field])
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid design catalog values at line {line}") from error
                if row["label"] not in {0, 1} or row["split"] not in {
                    "train",
                    "validation",
                    "test",
                }:
                    raise ValueError(f"invalid label/split at design catalog line {line}")
                if not row["selected"] or not str(row["leakage_group"]):
                    raise ValueError(f"canonical catalog line {line} is not selected/grouped")
                rows.append(row)
        rows.sort(key=lambda row: str(row["identifier"]))
        identifiers = [str(row["identifier"]) for row in rows]
        if not rows or len(identifiers) != len(set(identifiers)):
            raise ValueError("design catalog must contain unique non-empty identifiers")
        return rows

    @staticmethod
    def _write_index(root: Path, catalog: Sequence[Mapping[str, Any]]) -> None:
        """Join fixed design metadata to validated array assets in a DatasetIndex.

        Args:
            root: Self-contained pre-publication dataset root.
            catalog: Decoded canonical design rows.

        Returns:
            ``None`` after LambdaForge atomically writes ``members.jsonl``.

        Raises:
            ValueError: If annotation coverage and design membership disagree.
            OSError: If annotation reports or assets cannot be read.
        """
        report = json.loads((root / "annotation-report.json").read_text(encoding="utf-8"))
        annotation_rows = {str(row["identifier"]): row for row in report.get("records", ())}
        if set(annotation_rows) != {str(row["identifier"]) for row in catalog}:
            raise ValueError("annotation report and fixed design catalog have different members")
        dilution_members = {
            f"{replicate.name}/{path.stem}": {
                value.strip()
                for value in path.read_text(encoding="utf-8").splitlines()
                if value.strip()
            }
            for replicate in sorted((root / "design" / "dilutions").glob("replicate-*"))
            for path in sorted(replicate.glob("train-*.txt"))
        }
        members: list[DatasetMember] = []
        for row in catalog:
            identifier = str(row["identifier"])
            annotation = annotation_rows[identifier]
            base      = root / str(annotation["portable_base_path"])
            sidecar   = root / str(annotation["output"])
            structure = root / "structures" / f"{annotation['source_structure_sha256']}.cif"
            members.append(
                DatasetMember(
                    member_id=identifier,
                    partitions={
                        "split": str(row["split"]),
                        "leakage_group": str(row["leakage_group"]),
                        "global_phenotype": str(row["global_phenotype"]),
                        "interface_phenotype": str(row["interface_phenotype"]),
                    },
                    targets={
                        "dna_binding": int(row["label"]),
                        "local_ground_truth": bool(annotation["local_gt_available"]),
                    },
                    metadata={
                        "origin": str(row["origin"]),
                        "pdb_id": str(row["pdb_id"]),
                        "protein_chain": str(row["protein_chain"]),
                        "assembly_id": str(row["assembly_id"]),
                        "protein_copy": int(row["protein_copy"]),
                        "dilutions": sorted(
                            name
                            for name, values in dilution_members.items()
                            if identifier in values
                        ),
                    },
                    assets={
                        "universal_npz": DatasetAsset(
                            path=base.relative_to(root).as_posix(),
                            sha256=f"sha256:{annotation['base_npz_sha256']}",
                            size_bytes=base.stat().st_size,
                            media_type="application/x-npz",
                        ),
                        "dna_annotation": DatasetAsset(
                            path=sidecar.relative_to(root).as_posix(),
                            sha256=f"sha256:{annotation['sidecar_sha256']}",
                            size_bytes=sidecar.stat().st_size,
                            media_type="application/x-npz",
                        ),
                        "source_structure": DatasetAsset(
                            path=structure.relative_to(root).as_posix(),
                            sha256=f"sha256:{annotation['source_structure_sha256']}",
                            size_bytes=structure.stat().st_size,
                            media_type="chemical/x-mmcif",
                        ),
                    },
                )
            )
        DatasetIndex.write(root / "members.jsonl", members)

    def published_members(self, final_root: Path) -> Iterable[Mapping[str, Any]]:
        """Translate the validated index into LambdaForge streaming dataset members.

        Args:
            final_root: Self-contained pre-publication root with index and assets.

        Yields:
            Member mappings accepted by :meth:`lambdaforge.Work.outputs.dataset`.

        Raises:
            OSError: If the member index or a declared asset cannot be read.
        """
        for index, member in enumerate(DatasetIndex(final_root / "members.jsonl")):
            assets = {name: final_root / asset.path for name, asset in member.assets.items()}
            if index == 0:
                assets["dataset_design"] = final_root / "design"
            yield {
                "id": member.member_id,
                "partitions": dict(member.partitions),
                "targets": dict(member.targets),
                "metadata": dict(member.metadata),
                "display": dict(member.display),
                "assets": assets,
            }

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        """Publish complete manifest text with ``fsync`` and atomic replacement.

        Args:
            path: Final checkpoint-owned path.
            content: Complete UTF-8 payload.

        Returns:
            ``None`` after atomic replacement.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
