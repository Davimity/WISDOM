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

from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace


class DNAAnnotationSink:
    """Publish compact annotation archives aligned to immutable base surfaces."""

    def __init__(
        self,
        annotation_output : str = "annotations",
        report_output     : str = "annotation-report.json",
        curation_input    : str | None = None,
    ) -> None:
        """Bind named output locations for sidecars and their audit.

        Args:
            annotation_output: Named directory receiving ``*.dna.npz`` sidecars.
            report_output: Named JSON report output.
            curation_input: Optional named directory containing the split-free curated catalog.

        Raises:
            ValueError: If a logical input or output name is invalid.
        """
        if (
            not annotation_output.strip()
            or not report_output.strip()
            or (curation_input is not None and not curation_input.strip())
        ):
            raise ValueError("annotation output names cannot be empty")
        self.annotation_output = annotation_output
        self.report_output     = report_output
        self.curation_input    = curation_input
        self.records: dict[str, dict[str, Any]] = {}

    def write(self, record: ProcessingRecord, context: ProcessingWorkspace) -> None:
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
        arrays = record.value.get("arrays")
        metadata = record.value.get("metadata")
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

        self.records[record.key] = self._report(record.key, output_path, metadata)

    def resume(
        self,
        record          : ProcessingRecord,
        context         : ProcessingWorkspace,
        positive_gap    : float,
        negative_gap    : float,
        sensitivity_gaps: tuple[float, ...],
    ) -> ProcessingRecord | None:
        """Reuse one sidecar only when geometry provenance and target settings still match.

        Args:
            record: Current joined catalog/base-NPZ record.
            context: Workspace locating checkpoint-owned annotation sidecars.
            positive_gap: Current confident-positive surface gap in ångströms.
            negative_gap: Current confident-negative surface gap in ångströms.
            sensitivity_gaps: Current evaluation-only cutoff sequence in ångströms.

        Returns:
            Compact ``ProcessingRecord`` report for a fully valid reusable sidecar, or ``None``
            when the worker must recompute it.

        Raises:
            TypeError: If the joined record value is not a mapping.
        """
        if not isinstance(record.value, Mapping):
            raise TypeError("annotation resume requires a mapping record")
        base_path = Path(str(record.value.get("base_npz", "")))
        sidecar   = context.output(self.annotation_output) / f"{base_path.stem}.dna.npz"
        if not base_path.is_file() or not sidecar.is_file():
            return None
        try:
            with np.load(sidecar, allow_pickle=False) as archive:
                arrays   = {name: archive[name] for name in archive.files}
                metadata = json.loads(str(arrays["annotation_metadata_json"].item()))
            self._validate(arrays, metadata)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

        expected_base = hashlib.sha256(base_path.read_bytes()).hexdigest()
        if metadata.get("base_identifier") != record.key:
            return None
        if metadata.get("base_npz_sha256") != expected_base:
            return None
        if metadata.get("source_structure_sha256") != record.value.get("structure_sha256"):
            return None
        if float(metadata.get("positive_gap_angstrom", -1.0)) != positive_gap:
            return None
        if float(metadata.get("negative_gap_angstrom", -1.0)) != negative_gap:
            return None
        if tuple(metadata.get("sensitivity_gaps_angstrom", ())) != sensitivity_gaps:
            return None

        return record.with_value(self._report(record.key, sidecar, metadata))

    @staticmethod
    def _report(
        key     : str,
        path    : Path,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create the compact dataset-level audit row for one validated sidecar.

        Args:
            key: Stable protein identifier.
            path: Existing validated sidecar path.
            metadata: Validated annotation metadata aligned to the universal NPZ.

        Returns:
            JSON-compatible row consumed by annotation finalization and partitioning.
        """
        return {
            "identifier": key,
            "output": path.name,
            "surface_count": int(metadata["base_surface_count"]),
            "protein_label": int(metadata["protein_label"]),
            "base_npz_sha256": str(metadata["base_npz_sha256"]),
            "base_npz_path": str(metadata["base_npz_path"]),
            "source_structure_sha256": str(metadata["source_structure_sha256"]),
            "source_structure_path": str(metadata["source_structure_path"]),
            "split": "",
            "tier": str(metadata.get("tier", "core")),
            "local_gt_expected": bool(metadata["local_gt_expected"]),
            "local_gt_available": bool(metadata["local_gt_available"]),
            "local_gt_reason": str(metadata["local_gt_reason"]),
            "positive_surface_weight": float(metadata["positive_surface_weight"]),
            "total_surface_weight": float(metadata["total_surface_weight"]),
            "interface_fraction": float(metadata["interface_fraction"]),
            "number_of_positive_regions": int(metadata["number_of_positive_regions"]),
            "sidecar_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def finalize(self, context: ProcessingWorkspace) -> tuple[Path, Path]:
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

        # Preserve the split-free curated catalog beside the geometry and annotations. Partition
        # assignment is intentionally a later dataset-level operation.
        if self.curation_input is not None:
            curation_root = context.input(self.curation_input)
            if not curation_root.is_dir():
                raise RuntimeError("curation input must resolve to a directory")
            source = curation_root / "curated-catalog.csv"
            if not source.is_file():
                raise RuntimeError("curation input lacks curated-catalog.csv")
            self._atomic_copy(source, output_root / "curated-catalog.csv")

        annotated_records = sorted(
            self.records.values(), key=lambda value: str(value["identifier"])
        )
        self._write_annotated_catalog(output_root / "annotated-catalog.csv", annotated_records)

        report_path = context.output(self.report_output, create=True)
        payload = {
            "verdict": "PASS",
            "annotated_count": len(self.records),
            "local_gt_available_count": sum(
                bool(value["local_gt_available"]) for value in self.records.values()
            ),
            "local_gt_unavailable_positive_count": sum(
                value["protein_label"] == 1 and not value["local_gt_available"]
                for value in self.records.values()
            ),
            "partition_assignment": "deferred",
            "records": [self.records[key] for key in sorted(self.records)],
        }
        self._atomic_text(
            report_path,
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        return output_root, report_path

    @staticmethod
    def _write_annotated_catalog(path: Path, rows: list[dict[str, Any]]) -> None:
        """Write the geometry/annotation facts required by the partition task.

        Args:
            path: Destination CSV below the self-contained annotation root.
            rows: One validated annotation record per logical protein.

        Raises:
            OSError: If the table cannot be atomically replaced.
        """
        fields = tuple(sorted({name for row in rows for name in row}))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
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
                temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError("portable base NPZ does not match annotation fingerprint")
            value["portable_base_path"] = target.relative_to(output_root).as_posix()

    def _materialize_structures(self, output_root: Path) -> None:
        """Copy curation-verified structures from the geometry cache into publication.

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
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
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
        hard = arrays["surface_target_hard"]
        valid = arrays["surface_valid_mask"]
        soft = arrays["surface_target_soft"]
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
        distance = arrays["surface_distance_to_dna"]
        if distance_valid.dtype != np.bool_ or not np.isfinite(distance[distance_valid]).all():
            raise ValueError("computable DNA distances must be finite")
        if np.any(np.isfinite(distance[~distance_valid])):
            raise ValueError("non-computable DNA distances must be NaN")
        sensitivity = arrays["surface_target_hard_sensitivity"]
        gaps = arrays["sensitivity_gaps"]
        if sensitivity.shape != (count, len(gaps)) or not np.all(np.isin(sensitivity, (0, 1))):
            raise ValueError("sensitivity targets must have binary shape [M,T]")
        digest = str(arrays["base_npz_sha256"].item())
        if digest != metadata.get("base_npz_sha256") or len(digest) != 64:
            raise ValueError("sidecar/base fingerprint is malformed or inconsistent")
