"""Validated atomic sink for DNA surface annotation sidecars."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.preprocessing import PreprocessingRecord, PreprocessingSink
from lambdaforge.tasks import ArtifactDeclaration, ArtifactType, TaskContext


class DNAAnnotationSink(PreprocessingSink):
    """Publish compact annotation archives aligned to immutable base surfaces."""

    def __init__(
        self,
        annotation_output: str        = "annotations",
        report_output    : str        = "annotation-report.json",
        selection_input  : str | None = None,
    ) -> None:
        """Bind named output locations for sidecars and their audit.

        Args:
            annotation_output: Named directory receiving ``*.dna.npz`` sidecars.
            report_output: Named JSON report output.
            selection_input: Optional named directory input containing the complete lightweight
                selection artifact. Its catalog, splits, IDs, and subsets are copied into the
                run-owned annotation root for audit and report generation; LambdaForge's later
                member-stream publication keeps that selection as a separate named artifact.

        Raises:
            ValueError: If a logical input or output name is invalid.
        """
        if (
            not annotation_output.strip()
            or not report_output.strip()
            or (selection_input is not None and not selection_input.strip())
        ):
            raise ValueError("annotation output names cannot be empty")
        self.annotation_output = annotation_output
        self.report_output     = report_output
        self.selection_input   = selection_input
        self.records: dict[str, dict[str, Any]] = {}

    def write(self, record: PreprocessingRecord, context: TaskContext) -> None:
        """Validate all sidecar arrays, then publish one pickle-free NPZ atomically.

        Args:
            record: Annotation transform output with arrays, metadata, and filename.
            context: LambdaForge context resolving the annotation directory.

        Raises:
            TypeError: If the record does not follow the transform/sink contract.
            ValueError: If schemas, lengths, masks, values, or fingerprints are inconsistent.
            OSError: If atomic NPZ publication or verification fails.
        """
        if not isinstance(record.value, Mapping):
            raise TypeError("annotation value must be a mapping")
        arrays      = record.value.get("arrays")
        metadata    = record.value.get("metadata")
        output_name = record.value.get("output_name")
        if not isinstance(arrays, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("annotation requires array and metadata mappings")
        if not isinstance(output_name, str) or Path(output_name).name != output_name:
            raise ValueError("annotation output_name must be a safe filename")
        self._validate(arrays, metadata)

        output_root = context.output(self.annotation_output)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / output_name
        temporary = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.{uuid4().hex}.tmp.npz"
        )
        try:
            np.savez_compressed(temporary, **arrays)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        with np.load(output_path, allow_pickle=False) as stored:
            self._validate({name: stored[name] for name in stored.files}, metadata)

        self.records[record.key] = {
            "identifier": record.key,
            "output": output_name,
            "surface_count": int(metadata["base_surface_count"]),
            "protein_label": int(metadata["protein_label"]),
            "base_npz_sha256": str(metadata["base_npz_sha256"]),
            "base_npz_path": str(metadata["base_npz_path"]),
            "source_structure_sha256": str(metadata["source_structure_sha256"]),
            "source_structure_path": str(metadata["source_structure_path"]),
            "split": str(metadata["split"]),
            "tier": str(metadata["tier"]),
            "local_gt_expected": bool(metadata["local_gt_expected"]),
            "local_gt_available": bool(metadata["local_gt_available"]),
            "local_gt_reason": str(metadata["local_gt_reason"]),
            "positive_surface_area": float(metadata["positive_surface_area"]),
            "total_surface_area": float(metadata["total_surface_area"]),
            "interface_fraction": float(metadata["interface_fraction"]),
            "number_of_positive_regions": int(metadata["number_of_positive_regions"]),
            "sidecar_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }

    def is_complete(self, key: str, context: TaskContext) -> bool:
        """Require exact existing sidecar validation before LambdaForge resumes a record.

        Args:
            key: Stable protein identifier.
            context: LambdaForge context locating annotation outputs.

        Returns:
            True only when a uniquely matching sidecar exists and validates completely.
        """
        output_root = context.output(self.annotation_output)
        for path in output_root.glob("*.dna.npz"):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    arrays   = {name: archive[name] for name in archive.files}
                    metadata = json.loads(str(arrays["annotation_metadata_json"].item()))
                self._validate(arrays, metadata)
                if metadata.get("base_identifier") != key:
                    continue
                self.records[key] = {
                    "identifier": key,
                    "output": path.name,
                    "surface_count": int(metadata["base_surface_count"]),
                    "protein_label": int(metadata["protein_label"]),
                    "base_npz_sha256": str(metadata["base_npz_sha256"]),
                    "base_npz_path": str(metadata["base_npz_path"]),
                    "source_structure_sha256": str(metadata["source_structure_sha256"]),
                    "source_structure_path": str(metadata["source_structure_path"]),
                    "split": str(metadata["split"]),
                    "tier": str(metadata["tier"]),
                    "local_gt_expected": bool(metadata["local_gt_expected"]),
                    "local_gt_available": bool(metadata["local_gt_available"]),
                    "local_gt_reason": str(metadata["local_gt_reason"]),
                    "positive_surface_area": float(metadata["positive_surface_area"]),
                    "total_surface_area": float(metadata["total_surface_area"]),
                    "interface_fraction": float(metadata["interface_fraction"]),
                    "number_of_positive_regions": int(metadata["number_of_positive_regions"]),
                    "sidecar_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                return True
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return False

    def finalize(self, context: TaskContext) -> tuple[ArtifactDeclaration, ...]:
        """Write an ordered audit and declare the annotation dataset.

        Args:
            context: LambdaForge context resolving annotation and report outputs.

        Returns:
            Dataset and JSON report artifact declarations.

        Raises:
            RuntimeError: If no sidecar was written during the task.
        """
        if not self.records:
            raise RuntimeError("DNA annotation produced no sidecar archives")
        output_root = context.output(self.annotation_output)
        self._materialize_bases(output_root)
        self._materialize_structures(output_root)

        # Preserve all portable curation contracts inside the immutable final publication. The
        # plain-text split lists are convenient for humans and identifiers.json preserves labels,
        # clusters, reserves, and the deterministic selection policy in one machine-readable file.
        if self.selection_input is not None:
            curation_root = context.input(self.selection_input)
            if not curation_root.is_dir():
                raise RuntimeError("selection input must resolve to a directory")
            for asset_name in (
                "catalog.csv",
                "catalog.parquet",
                "identifiers.json",
                "labels.csv",
                "proteins.txt",
                "README.md",
                "audit.json",
                "audit.md",
                "statistics.csv",
                "distributions.png",
                "train.txt",
                "val.txt",
                "test.txt",
                "validation_reserve.txt",
                "test_reserve.txt",
            ):
                source = curation_root / asset_name
                if source.is_file():
                    self._atomic_copy(source, output_root / asset_name)
            if not (output_root / "catalog.csv").is_file():
                raise RuntimeError("curated catalog input did not resolve to catalog.csv")

            subset_source = curation_root / "subsets"
            subset_target = output_root / "subsets"
            if subset_source.is_dir() and not subset_target.exists():
                temporary = subset_target.with_name(
                    f".{subset_target.name}.{os.getpid()}.{uuid4().hex}.tmp"
                )
                try:
                    shutil.copytree(subset_source, temporary)
                    os.replace(temporary, subset_target)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)

        main_records = sorted(
            (
                value
                for value in self.records.values()
                if value["split"] in {"train", "val", "test"}
            ),
            key=lambda value: str(value["identifier"]),
        )
        self._write_manifest(output_root / "manifest.csv", main_records)
        subset_counts = self._write_subset_manifests(output_root, main_records)

        report_path = context.output(self.report_output, create=True)
        local_records, replacements = self._local_evaluation_records()
        payload     = {
            "verdict": "PASS",
            "annotated_count": len(self.records),
            "local_gt_available_count": sum(
                bool(value["local_gt_available"]) for value in self.records.values()
            ),
            "local_gt_unavailable_positive_count": sum(
                value["protein_label"] == 1 and not value["local_gt_available"]
                for value in self.records.values()
            ),
            "local_evaluation_count": len(local_records),
            "local_evaluation_replacements": replacements,
            "subset_counts": subset_counts,
            "records": [self.records[key] for key in sorted(self.records)],
        }
        self._atomic_text(
            report_path,
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        self._write_manifest(output_root / "local-manifest.csv", local_records)
        self._write_dataset_index(output_root)
        return (
            ArtifactDeclaration(
                path=output_root.relative_to(context.run_dir),
                kind=ArtifactType.DATASET,
            ),
            ArtifactDeclaration(
                path=report_path.relative_to(context.run_dir),
                kind=ArtifactType.REPORT,
                media_type="application/json",
            ),
        )

    def _write_dataset_index(self, output_root: Path) -> None:
        """Publish LambdaForge's canonical logical-member index for the final dataset.

        Every main-split protein is one member. The index keeps the benchmark split as an arbitrary
        partition, the global DNA-binding label and local-ground-truth availability as targets,
        and the exact universal NPZ, aligned DNA sidecar, and curated source structure as
        checksummed assets. Positive reserve records remain auditable files for local-evaluation
        replacement but are deliberately absent from this logical-member index and every global
        training/evaluation manifest.

        Args:
            output_root: Final self-contained dataset root holding base archives and sidecars.

        Raises:
            OSError: If an indexed file cannot be read or the JSONL index cannot be replaced.
            ValueError: If a member ID or asset descriptor violates LambdaForge's dataset schema.
        """
        members: list[DatasetMember] = []
        for key in sorted(self.records):
            value           = self.records[key]
            if value["split"] not in {"train", "val", "test"}:
                continue
            base_path       = output_root / str(value["portable_base_path"])
            annotation_path = output_root / str(value["output"])
            structure_hash  = str(value["source_structure_sha256"])
            structure_paths = sorted((output_root / "structures").glob(f"{structure_hash}*"))

            base_asset = DatasetAsset(
                path=str(value["portable_base_path"]),
                sha256=f"sha256:{value['base_npz_sha256']}",
                size_bytes=base_path.stat().st_size,
                media_type="application/x-npz",
            )
            annotation_asset = DatasetAsset(
                path=str(value["output"]),
                sha256=f"sha256:{value['sidecar_sha256']}",
                size_bytes=annotation_path.stat().st_size,
                media_type="application/x-npz",
            )
            assets = {"universal_npz": base_asset, "dna_annotation": annotation_asset}
            if structure_paths:
                structure_path = structure_paths[0]
                assets["source_structure"] = DatasetAsset(
                    path=structure_path.relative_to(output_root).as_posix(),
                    sha256=f"sha256:{structure_hash}",
                    size_bytes=structure_path.stat().st_size,
                    media_type="chemical/x-mmcif",
                )
            members.append(
                DatasetMember(
                    member_id=key,
                    partitions={"split": str(value["split"]), "tier": str(value["tier"])},
                    targets={
                        "dna_binding": int(value["protein_label"]),
                        "local_ground_truth": bool(value["local_gt_available"]),
                    },
                    metadata={
                        "local_gt_reason": str(value["local_gt_reason"]),
                        "surface_point_count": int(value["surface_count"]),
                        "positive_surface_area_angstrom2": float(
                            value["positive_surface_area"]
                        ),
                        "total_surface_area_angstrom2": float(value["total_surface_area"]),
                        "interface_fraction": float(value["interface_fraction"]),
                        "positive_surface_regions": int(value["number_of_positive_regions"]),
                    },
                    assets=assets,
                )
            )
        DatasetIndex.write(output_root / "members.jsonl", members)

    def _write_subset_manifests(
        self,
        output_root: Path,
        rows       : list[dict[str, Any]],
    ) -> dict[str, dict[str, int]]:
        """Convert curation-selected subset IDs into portable NPZ/sidecar manifests.

        Selection, balancing, nesting, and cluster diversity belong entirely to the lightweight
        curation Task. This final sink only joins those immutable identifiers to the base and
        annotation paths created by the heavy recipe, preventing two implementations of the
        scientific sampling policy.

        Args:
            output_root: Final dataset root containing copied ``subsets/*/identifiers.json`` files.
            rows: Complete main-split annotation records indexed by stable identifier.

        Returns:
            Per-subset train/validation/test member counts for the annotation report.

        Raises:
            ValueError: If a subset manifest is malformed, unknown, duplicated, or changes a split.
            OSError: If a subset JSON or output manifest cannot be read or written.
        """
        indexed = {str(value["identifier"]): value for value in rows}
        counts : dict[str, dict[str, int]] = {}
        subset_root = output_root / "subsets"
        if not subset_root.is_dir():
            return counts
        for path in sorted(subset_root.glob("*/identifiers.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("records") if isinstance(payload, dict) else None
            if not isinstance(records, list):
                raise ValueError(f"subset {path.parent.name} has no records list")
            identifiers = [
                str(value.get("identifier", ""))
                for value in records
                if isinstance(value, dict)
            ]
            if len(identifiers) != len(records) or len(identifiers) != len(set(identifiers)):
                raise ValueError(f"subset {path.parent.name} has invalid or duplicate identifiers")
            unknown = sorted(set(identifiers) - indexed.keys())
            if unknown:
                raise ValueError(
                    f"subset {path.parent.name} has unknown identifiers: {unknown[:5]}"
                )
            selected = [indexed[identifier] for identifier in identifiers]
            declared_splits = {
                str(value["identifier"]): str(value["split"])
                for value in records
                if isinstance(value, dict)
            }
            if any(
                str(value["split"]) != declared_splits[str(value["identifier"])]
                for value in selected
            ):
                raise ValueError(f"subset {path.parent.name} changes an immutable split")
            selected.sort(key=lambda value: str(value["identifier"]))
            self._write_manifest(path.parent / "manifest.csv", selected)
            counts[path.parent.name] = {
                split: sum(value["split"] == split for value in selected)
                for split in ("train", "val", "test")
            }
        return counts

    def _local_evaluation_records(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Build validation/test local subsets with deterministic reserve replacement.

        Globally positive rows with zero projected positive points remain in the global manifest
        but are absent here. An available positive from the corresponding pre-reserved pool replaces
        it for localization only; the global split and training data are never rewritten.

        Returns:
            Local-evaluation manifest rows and explicit original/replacement audit mappings.
        """
        rows         = list(self.records.values())
        local_rows   = [
            dict(value)
            for value in rows
            if value["split"] in {"val", "test"}
            and (value["protein_label"] == 0 or value["local_gt_available"])
        ]
        replacements: list[dict[str, str]] = []
        used: set[str] = set()
        for split, reserve_split in (("val", "validation_reserve"), ("test", "test_reserve")):
            invalid = sorted(
                (
                    value
                    for value in rows
                    if value["split"] == split
                    and value["protein_label"] == 1
                    and not value["local_gt_available"]
                ),
                key=lambda value: str(value["identifier"]),
            )
            reserves = sorted(
                (
                    value
                    for value in rows
                    if value["split"] == reserve_split
                    and value["protein_label"] == 1
                    and value["local_gt_available"]
                ),
                key=lambda value: str(value["identifier"]),
            )
            for original, replacement in zip(invalid, reserves, strict=False):
                identifier = str(replacement["identifier"])
                if identifier in used:
                    continue
                replacement_row               = dict(replacement)
                replacement_row["split"]      = split
                replacement_row["reserve_of"] = str(original["identifier"])
                local_rows.append(replacement_row)
                used.add(identifier)
                replacements.append(
                    {
                        "split": split,
                        "original": str(original["identifier"]),
                        "reason": str(original["local_gt_reason"]),
                        "replacement": identifier,
                    }
                )
        return sorted(local_rows, key=lambda value: str(value["identifier"])), replacements

    @staticmethod
    def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
        """Publish a portable evaluation manifest from prepared record mappings.

        Args:
            path: Final CSV path.
            rows: Records with portable base and annotation paths.

        Raises:
            OSError: If writing or atomic replacement fails.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                fields = (
                    "file",
                    "annotation",
                    "label",
                    "split",
                    "identifier",
                    "tier",
                )
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for value in rows:
                    writer.writerow(
                        {
                            "file": value["portable_base_path"],
                            "annotation": value["output"],
                            "label": value["protein_label"],
                            "split": value["split"],
                            "identifier": value["identifier"],
                            "tier": value["tier"],
                        }
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _materialize_bases(self, output_root: Path) -> None:
        """Copy exact universal NPZ bytes into the portable annotation dataset.

        The source archives remain immutable. Packaging them beside sidecars avoids machine-local
        absolute paths in ``manifest.csv`` and lets one logical LambdaForge dataset mount move
        between workstations and clusters without path rewriting.

        Args:
            output_root: Annotation dataset root that will contain a ``base`` subdirectory.

        Raises:
            RuntimeError: If a source is missing or its copied bytes disagree with provenance.
            OSError: If an atomic copy cannot be completed.
        """
        base_root = output_root / "base"
        base_root.mkdir(parents=True, exist_ok=True)
        for value in self.records.values():
            source = Path(str(value["base_npz_path"]))
            digest = str(value["base_npz_sha256"])
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise RuntimeError("base NPZ changed before annotation dataset publication")
            target = base_root / f"{digest}.npz"
            if not target.is_file():
                temporary = target.with_name(
                    f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp"
                )
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError("portable base NPZ does not match annotation fingerprint")
            value["portable_base_path"] = target.relative_to(output_root).as_posix()

    def _materialize_structures(self, output_root: Path) -> None:
        """Copy selection-verified structures from the geometry cache into publication.

        Args:
            output_root: Final dataset root receiving a ``structures`` directory.

        Raises:
            RuntimeError: If a resolved structure is missing or differs from curation provenance.
            OSError: If atomic copying or digest verification fails.
        """
        structure_root = output_root / "structures"
        structure_root.mkdir(parents=True, exist_ok=True)
        for value in self.records.values():
            source = Path(str(value["source_structure_path"]))
            digest = str(value["source_structure_sha256"])
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise RuntimeError("resolved source structure differs from curation provenance")
            target = structure_root / f"{digest}.cif"
            if not target.is_file():
                self._atomic_copy(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError("portable source structure differs from curation provenance")

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        """Publish complete UTF-8 content by atomic replacement.

        Args:
            path: Final text or JSON path.
            content: Complete UTF-8 payload.

        Raises:
            OSError: If writing, syncing, or replacement fails.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        """Copy one immutable global asset with atomic replacement and byte verification.

        Args:
            source: Existing regular file supplied through a named task input.
            target: Final file path inside the annotation dataset root.

        Raises:
            OSError: If reading, copying, synchronizing, or replacing the file fails.
            RuntimeError: If the final bytes differ from the declared source bytes.
        """
        expected  = hashlib.sha256(source.read_bytes()).hexdigest()
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError("copied benchmark catalog does not match its source bytes")

    @staticmethod
    def _validate(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
        """Enforce sidecar schema, alignment, finite-value, and ambiguity invariants.

        Args:
            arrays: Sidecar array mapping.
            metadata: Parsed annotation provenance.

        Raises:
            ValueError: If any required field or numerical/alignment invariant fails.
        """
        required = {
            "surface_target_hard",
            "surface_valid_mask",
            "surface_target_soft",
            "surface_distance_to_dna",
            "surface_distance_valid",
            "surface_target_hard_sensitivity",
            "local_gt_available",
            "sensitivity_gaps",
            "base_npz_sha256",
            "annotation_metadata_json",
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"annotation sidecar is missing arrays: {sorted(missing)}")
        if any(array.dtype == object for array in arrays.values()):
            raise ValueError("annotation sidecars cannot contain object arrays")
        count = int(metadata.get("base_surface_count", -1))
        for name in (
            "surface_target_hard",
            "surface_valid_mask",
            "surface_target_soft",
            "surface_distance_to_dna",
            "surface_distance_valid",
        ):
            if arrays[name].shape != (count,):
                raise ValueError(f"{name} must align exactly with base surface length")
        hard  = arrays["surface_target_hard"]
        valid = arrays["surface_valid_mask"]
        soft  = arrays["surface_target_soft"]
        if not np.all(np.isin(hard, (0, 1))) or valid.dtype != np.bool_:
            raise ValueError("hard targets must be binary and valid mask must be Boolean")
        if not np.isfinite(soft).all() or np.any((soft < 0.0) | (soft > 1.0)):
            raise ValueError("soft targets must be finite probabilities")
        available = arrays["local_gt_available"]
        if available.shape != () or available.dtype != np.bool_:
            raise ValueError("local_gt_available must be one Boolean scalar")
        if bool(metadata.get("local_gt_available")) != bool(available.item()):
            raise ValueError("local GT availability disagrees between array and metadata")
        if metadata.get("protein_label") == 1 and not available.item() and np.any(valid):
            raise ValueError("locally unavailable positives cannot expose valid surface targets")
        distance_valid = arrays["surface_distance_valid"]
        distance       = arrays["surface_distance_to_dna"]
        if distance_valid.dtype != np.bool_ or not np.isfinite(distance[distance_valid]).all():
            raise ValueError("computable DNA distances must be finite")
        if np.any(np.isfinite(distance[~distance_valid])):
            raise ValueError("non-computable DNA distances must be NaN")
        sensitivity = arrays["surface_target_hard_sensitivity"]
        gaps        = arrays["sensitivity_gaps"]
        if sensitivity.shape != (count, len(gaps)) or not np.all(np.isin(sensitivity, (0, 1))):
            raise ValueError("sensitivity targets must have binary shape [M,T]")
        digest = str(arrays["base_npz_sha256"].item())
        if digest != metadata.get("base_npz_sha256") or len(digest) != 64:
            raise ValueError("sidecar/base fingerprint is malformed or inconsistent")
