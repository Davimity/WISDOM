"""Fixed train, validation, and test membership used by DNA preprocessing."""

import csv
import json
import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections.abc import Mapping


class DatasetManifests:
    """Load self-contained JSONL manifests or labelled TXT views joined to a catalog."""

    REQUIRED_FIELDS = frozenset({
        "identifier",
        "label",
        "split",
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
    })
    JSON_FIELDS = frozenset({
        "dna_chains",
        "binding_residue_indices",
        "assembly_rotation",
        "assembly_translation",
    })

    def __init__(
        self,
        train      : Path,
        validation : Path,
        test       : Path,
        catalog    : Path | None = None,
        dilutions  : Path | None = None,
    ) -> None:
        """Bind the three split views and optional compatibility evidence.

        A modern Selection output stores one complete JSON object per line and needs no
        catalog. Existing ``*-labelled.txt`` files contain only ``identifier<TAB>label``;
        when those files are used, ``catalog.csv`` supplies assembly, contact, provenance,
        leakage, and phenotype fields. An optional dilution directory restores training-view
        memberships without changing the proteins that are geometrically preprocessed.

        Args:
            train: Training JSONL or two-column labelled TXT file.
            validation: Validation JSONL or two-column labelled TXT file.
            test: Test JSONL or two-column labelled TXT file.
            catalog: Selection ``catalog.csv`` required by labelled TXT inputs.
            dilutions: Directory containing ``replicate-*/train-*-labelled.txt`` views.
        """
        self.train      = train
        self.validation = validation
        self.test       = test
        self.catalog    = catalog
        self.dilutions  = dilutions

    def load(self, work: lf.Work, verbose: bool = False) -> list[dict[str, Any]]:
        """Decode the fixed population and attach deterministic dilution memberships.

        Args:
            work: Active Work used for phase and optional per-protein logging.
            verbose: Log every decoded identifier, label, assembly, and copy when true.

        Returns:
            Identifier-sorted JSON-compatible records with explicit split and binary label.

        Raises:
            ValueError: If TXT inputs lack a catalog, identifiers repeat, or a manifest
                contradicts its catalog label/split or required scientific fields.
            OSError: If a manifest, catalog, or dilution file cannot be read.
        """
        manifests = (
            ("train",      self.train),
            ("validation", self.validation),
            ("test",       self.test),
        )
        uses_labelled_text = any(path.suffix.lower() != ".jsonl" for _, path in manifests)
        catalog_rows       = self._catalog_rows() if uses_labelled_text else {}

        rows: list[dict[str, Any]] = []
        for split, path in manifests:
            work.log(f"Reading {split} preprocessing records from {path.name}")
            if path.suffix.lower() == ".jsonl":
                split_rows = self._jsonl_rows(path)
            else:
                split_rows = self._labelled_rows(path, split, catalog_rows)

            for row in split_rows:
                missing = self.REQUIRED_FIELDS - row.keys()
                if missing:
                    raise ValueError(f"{path.name} lacks fields: {sorted(missing)}")
                if str(row["split"]) != split or int(row["label"]) not in {0, 1}:
                    raise ValueError(f"{path.name} has an invalid split or label")

                row["dilutions"] = list(row.get("dilutions", ()))
                rows.append(row)
                if verbose:
                    work.log(
                        f"Loaded {row['identifier']}: split={split}, label={row['label']}, "
                        f"assembly={row['assembly_id']}, copy={row['protein_copy']}",
                        level="debug",
                    )

        identifiers = [str(row["identifier"]) for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("train, validation, and test manifests repeat an identifier")

        train_labels = {
            str(row["identifier"]): int(row["label"])
            for row in rows
            if row["split"] == "train"
        }
        memberships = self._dilution_memberships(train_labels)
        for row in rows:
            identifier       = str(row["identifier"])
            row["dilutions"] = sorted(
                set(str(value) for value in row["dilutions"])
                | set(memberships.get(identifier, ()))
            )
        rows.sort(key=lambda row: str(row["identifier"]))

        work.log(
            f"Loaded {len(rows)} unique proteins: "
            f"{sum(row['split'] == 'train' for row in rows)} train, "
            f"{sum(row['split'] == 'validation' for row in rows)} validation, and "
            f"{sum(row['split'] == 'test' for row in rows)} test"
        )
        return rows

    def _catalog_rows(self) -> dict[str, dict[str, Any]]:
        """Decode the scientific fields needed to enrich two-column TXT inputs.

        Returns:
            Catalog rows keyed by exact protein identifier.

        Raises:
            ValueError: If labelled TXT inputs were selected without a catalog.
            OSError: If the catalog cannot be read.
        """
        if self.catalog is None:
            raise ValueError("catalog is required when preprocessing labelled TXT manifests")

        rows: dict[str, dict[str, Any]] = {}
        with self.catalog.open("r", encoding="utf-8", newline="") as stream:
            for raw in csv.DictReader(stream):
                row: dict[str, Any] = dict(raw)
                for field in self.JSON_FIELDS:
                    row[field] = json.loads(str(row[field]))

                row["label"]             = int(row["label"])
                row["protein_copy"]      = int(row["protein_copy"])
                row["local_gt_expected"] = str(row["local_gt_expected"]).lower() == "true"
                rows[str(row["identifier"])] = row
        return rows

    def _jsonl_rows(self, path: Path) -> list[dict[str, Any]]:
        """Read complete one-object-per-line preprocessing records.

        Args:
            path: UTF-8 JSONL manifest produced by current Selection code.

        Returns:
            Decoded object mappings in authored order.

        Raises:
            ValueError: If a non-empty line is not one JSON object.
            OSError: If the file cannot be read.
        """
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path.name} line {line_number} must be one JSON object")
            rows.append(dict(value))
        return rows

    def _labelled_rows(
        self,
        path        : Path,
        split       : str,
        catalog_rows: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Join a compact labelled split to the exact Selection catalog.

        Args:
            path: UTF-8 ``identifier<TAB>label`` manifest.
            split: Canonical split name expected in the catalog.
            catalog_rows: Decoded catalog keyed by identifier.

        Returns:
            Complete preprocessing records in manifest order.

        Raises:
            ValueError: If an identifier is absent or its label/split contradicts the catalog.
            OSError: If the labelled file cannot be read.
        """
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            identifier, label_text = line.split("\t")
            catalog = catalog_rows.get(identifier)
            if catalog is None:
                raise ValueError(f"{identifier} is absent from {self.catalog}")

            row   = dict(catalog)
            label = int(label_text)
            if int(row["label"]) != label or str(row["split"]) != split:
                raise ValueError(f"{identifier} labelled split contradicts catalog.csv")

            row["label"] = label
            rows.append(row)
        return rows

    def _dilution_memberships(
        self,
        train_labels: Mapping[str, int],
    ) -> dict[str, list[str]]:
        """Read optional nested training views from labelled dilution files.

        Args:
            train_labels: Canonical training labels keyed by identifier.

        Returns:
            Mapping from protein identifier to views such as ``replicate-00/train-25``.
            An absent dilution directory returns an empty mapping.

        Raises:
            ValueError: If a dilution references a non-training protein or changes its label.
            OSError: If a dilution file cannot be read.
        """
        memberships: dict[str, list[str]] = {}
        if self.dilutions is None:
            return memberships

        for path in sorted(self.dilutions.glob("replicate-*/train-*-labelled.txt")):
            view = f"{path.parent.name}/{path.stem.removesuffix('-labelled')}"
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    identifier, label_text = line.split("\t")
                    if train_labels.get(identifier) != int(label_text):
                        raise ValueError(f"{path} contradicts the fixed training population")
                    memberships.setdefault(identifier, []).append(view)
        return memberships
