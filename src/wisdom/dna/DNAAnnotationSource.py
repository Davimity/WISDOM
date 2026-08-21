"""Join curated DNA labels to universal preprocessing outputs by exact identity."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from typing import Any

from lambdaforge.preprocessing import PreprocessingRecord, PreprocessingSource
from lambdaforge.tasks import TaskContext


class DNAAnnotationSource(PreprocessingSource):
    """Expose accepted catalog rows whose universal NPZ exists in the bound report."""

    def __init__(
        self,
        catalog_input        : str = "dataset_catalog",
        base_report_input    : str = "base_preprocessing_report",
        structure_cache_input: str = "structure_cache",
    ) -> None:
        """Bind the curated catalog and exact upstream preprocessing report.

        Args:
            catalog_input: Named input containing ``catalog.csv`` from the selection Task.
            base_report_input: Named workflow-bound ``preprocessing-report.json`` input.
            structure_cache_input: Named directory containing geometry's downloaded RCSB
                ``*.cif.gz`` files.

        Raises:
            ValueError: If either logical name is empty.
        """
        names = (catalog_input, base_report_input, structure_cache_input)
        if any(not name.strip() for name in names):
            raise ValueError("annotation input names cannot be empty")
        self.catalog_input         = catalog_input
        self.base_report_input     = base_report_input
        self.structure_cache_input = structure_cache_input

    def records(self, context: TaskContext) -> Iterable[PreprocessingRecord]:
        """Yield one joined row per accepted protein without guessing filenames.

        Args:
            context: LambdaForge task context resolving the two declared inputs.

        Yields:
            Records containing catalog provenance and the exact base NPZ path.

        Raises:
            ValueError: If headers, labels, identifiers, or upstream coverage disagree.
            OSError: If an input or cached coordinate file cannot be read or materialized.
        """
        report_path = context.input(self.base_report_input)
        report      = json.loads(report_path.read_text(encoding="utf-8"))
        records     = report.get("records") if isinstance(report, dict) else None
        if not isinstance(records, list):
            raise ValueError("base preprocessing report must contain a records list")
        outputs = {
            str(value["identifier"]): report_path.parent / "processed" / str(value["output"])
            for value in records
            if isinstance(value, dict)
            and value.get("status") in {"processed", "skipped"}
            and value.get("identifier")
            and value.get("output")
        }

        catalog_path = context.input(self.catalog_input)
        structure_cache = context.input(self.structure_cache_input)
        with catalog_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "base_identifier",
                "binding_residue_indices",
                "label",
                "local_gt_expected",
                "local_gt_method",
                "structure_path",
                "protein_chain",
                "dna_chains",
                "structure_sha256",
            }
            if not required.issubset(reader.fieldnames or ()):
                missing = sorted(required - set(reader.fieldnames or ()))
                raise ValueError(f"DNA catalog is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                identifier = str(row["base_identifier"]).strip()
                if identifier not in outputs:
                    raise ValueError(
                        f"DNA catalog line {line_number} has no successful base NPZ: {identifier}"
                    )
                value: dict[str, Any] = dict(row)
                value["label"]                   = int(row["label"])
                value["dna_chains"]              = json.loads(row["dna_chains"] or "[]")
                value["binding_residue_indices"] = json.loads(
                    row["binding_residue_indices"] or "[]"
                )
                value["local_gt_expected"] = row["local_gt_expected"].lower() == "true"
                value["base_npz"]           = str(outputs[identifier].resolve())

                # Geometry already downloaded every selected PDB entry in parallel. The source
                # only joins its archive path; decompression, hashing, and materialization remain
                # per-record work and therefore execute inside the CPU worker pool.
                pdb_id          = identifier.split("_", maxsplit=1)[0].lower()
                compressed_path = structure_cache / f"{pdb_id}.cif.gz"
                if not compressed_path.is_file():
                    raise FileNotFoundError(
                        f"geometry structure cache lacks {compressed_path.name}"
                    )
                value["structure_archive_path"] = str(compressed_path.resolve())
                yield PreprocessingRecord(key=identifier, value=value)
