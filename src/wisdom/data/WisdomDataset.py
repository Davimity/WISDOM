"""Minimal pickle-free ingestion of labeled WISDOM NPZ representations."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from lambdaforge.data import DatasetIndex
from torch import Tensor
from torch.utils.data import Dataset


class WisdomDataset(Dataset[Mapping[str, Tensor | str]]):
    """Load one explicit labeled split without changing preprocessed geometry."""

    LEGACY_COLUMNS = ("file", "label", "split")
    DNA_COLUMNS    = ("file", "annotation", "label", "split", "identifier", "tier")
    SPLITS         = frozenset({"train", "val", "test"})

    def __init__(
        self,
        manifest: str | Path,
        split   : str,
        subset  : str = "full",
    ) -> None:
        """Read and validate a compact ``file,label,split`` CSV manifest.

        Paths are resolved relative to the manifest directory. Rows keep CSV order and only rows
        whose explicit split equals ``split`` are retained; no random or implicit division occurs.
        NPZ files remain unopened until ``__getitem__`` so DataLoader workers do not share handles.

        Args:
            manifest: LambdaForge 0.11 managed dataset root containing ``index.jsonl``, or a legacy
                CSV with exactly ``file,label,split`` columns.
            split: Explicit subset to expose; one of ``train``, ``val``, or ``test``.
            subset: ``full`` or a configured selection name such as ``25pct``. Managed dataset
                members carry this view membership in metadata without duplicating heavy assets.

        Raises:
            ValueError: If the split, header, label, row split, path, or selected subset is invalid.
            OSError: If the manifest cannot be read.
        """
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {sorted(self.SPLITS)}")
        if not subset.strip():
            raise ValueError("subset cannot be empty")

        manifest_path = Path(manifest).resolve()
        if manifest_path.is_dir():
            managed_records = self._dataset_records(manifest_path, split, subset)
            self.manifest = manifest_path / "index.jsonl"
            self.split    = split
            self.records  = tuple(managed_records)
            return

        records: list[tuple[Path, Path | None, int, str, str]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            if columns not in {self.LEGACY_COLUMNS, self.DNA_COLUMNS}:
                raise ValueError(
                    f"manifest columns must be exactly {self.LEGACY_COLUMNS} or {self.DNA_COLUMNS}"
                )

            for line_number, row in enumerate(reader, start=2):
                row_split = str(row["split"]).strip()
                if row_split not in self.SPLITS:
                    raise ValueError(f"manifest line {line_number} has invalid split {row_split!r}")
                try:
                    label = int(str(row["label"]).strip())
                except ValueError as error:
                    raise ValueError(
                        f"manifest line {line_number} label must be integer 0 or 1"
                    ) from error
                if label not in {0, 1}:
                    raise ValueError(f"manifest line {line_number} label must be 0 or 1")

                file_value = str(row["file"]).strip()
                if not file_value:
                    raise ValueError(f"manifest line {line_number} file cannot be empty")
                path = Path(file_value)
                if not path.is_absolute():
                    path = manifest_path.parent / path
                path = path.resolve()
                if not path.is_file() or path.suffix.lower() != ".npz":
                    raise ValueError(f"manifest line {line_number} does not reference an NPZ file")
                annotation_path: Path | None = None
                identifier = path.stem
                tier       = "unspecified"
                if columns == self.DNA_COLUMNS:
                    annotation_value = str(row["annotation"]).strip()
                    annotation_path  = Path(annotation_value)
                    if not annotation_path.is_absolute():
                        annotation_path = manifest_path.parent / annotation_path
                    annotation_path = annotation_path.resolve()
                    if not annotation_path.is_file() or annotation_path.suffix.lower() != ".npz":
                        raise ValueError(
                            f"manifest line {line_number} does not reference an annotation NPZ"
                        )
                    identifier = str(row["identifier"]).strip()
                    tier       = str(row["tier"]).strip()
                    if not identifier:
                        raise ValueError(f"manifest line {line_number} identifier cannot be empty")
                if row_split == split:
                    records.append((path, annotation_path, label, identifier, tier))

        if not records:
            raise ValueError(f"manifest contains no records for split {split!r}")

        self.manifest = manifest_path
        self.split    = split
        self.records  = tuple(records)

    @staticmethod
    def _dataset_records(
        root  : Path,
        split : str,
        subset: str,
    ) -> list[tuple[Path, Path | None, int, str, str]]:
        """Read explicit labels, partitions, views, and assets from DatasetArtifact v2.

        Args:
            root: Resolved immutable LambdaForge dataset placement.
            split: Required main partition.
            subset: ``full`` or one deterministic dilution name.

        Returns:
            Ordered base/annotation paths and labels consumed by ``__getitem__``.

        Raises:
            ValueError: If the index, split, targets, view metadata, or assets are malformed.
            OSError: If the canonical index cannot be read.
        """
        index_path = root / "index.jsonl"
        if not index_path.is_file():
            raise ValueError("managed WISDOM dataset root must contain index.jsonl")

        records: list[tuple[Path, Path | None, int, str, str]] = []
        for member in DatasetIndex(index_path):
            if str(member.partitions.get("split", "")) != split:
                continue
            dilutions = member.metadata.get("dilutions", ())
            if subset != "full" and (
                not isinstance(dilutions, (list, tuple)) or subset not in dilutions
            ):
                continue
            try:
                label = int(member.targets["dna_binding"])
                base  = root / member.assets["universal_npz"].path
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"managed member {member.member_id!r} lacks WISDOM label/base asset"
                ) from error
            if label not in {0, 1} or not base.is_file() or base.suffix.lower() != ".npz":
                raise ValueError(f"managed member {member.member_id!r} has invalid base data")
            annotation_asset = member.assets.get("dna_annotation")
            annotation = root / annotation_asset.path if annotation_asset is not None else None
            if annotation is not None and (
                not annotation.is_file() or annotation.suffix.lower() != ".npz"
            ):
                raise ValueError(
                    f"managed member {member.member_id!r} has invalid DNA annotation"
                )
            records.append(
                (
                    base,
                    annotation,
                    label,
                    member.member_id,
                    str(member.partitions.get("tier", "unspecified")),
                )
            )
        if not records:
            raise ValueError(
                f"managed dataset contains no records for split={split!r}, subset={subset!r}"
            )
        return records

    def __len__(self) -> int:
        """Return the number of explicitly labeled proteins in the selected split.

        Returns:
            Count of retained manifest records.
        """
        return len(self.records)

    def __getitem__(self, index: int) -> Mapping[str, Tensor | str]:
        """Load only WISDOMv1 arrays and convert them to correctly typed tensors.

        Relation bit masks use preprocessing meanings ``1=spatial``, ``2=covalent`` and
        ``3=both``. WISDOMv1 maps them deterministically to R-GCN relation IDs ``0,1,2`` by
        subtracting one. Atom/surface graphs remain in their stored undirected-once ``src<dst``
        form here; ``WisdomCollator`` creates both directed message-passing orientations.

        Args:
            index: Zero-based selected-split record index.

        Returns:
            Mapping containing integer atom categories/edges/relations, float32 curvature and area
            tensors, the surface-to-atom incidence, and one scalar float32 binary target.

        Raises:
            IndexError: If ``index`` is outside the selected split.
            ValueError: If required arrays, shapes, finite values, relation semantics, or endpoint
                ranges are inconsistent with the current WISDOM NPZ contract.
            OSError: If the NPZ cannot be opened.
        """
        path, annotation_path, label, identifier, tier = self.records[index]
        required = {
            "atomic_numbers",
            "residue_type_ids",
            "atom_edge_index",
            "atom_edge_relation_mask",
            "surface_curvatures",
            "surface_edge_index",
            "surface_atom_edge_index",
            "surface_area_weights",
        }
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path.name} is missing WISDOMv1 arrays: {sorted(missing)}")
            values = {name: archive[name] for name in required}
            for name in ("surface_positions", "surface_normals"):
                if name in archive.files:
                    values[name] = archive[name]

        atom_count    = len(values["atomic_numbers"])
        surface_count = len(values["surface_area_weights"])

        # Categories and continuous point features establish model input dimensions.
        if values["atomic_numbers"].shape != (atom_count,) or atom_count == 0:
            raise ValueError("atomic_numbers must have non-empty shape [N]")
        if values["residue_type_ids"].shape != (atom_count,):
            raise ValueError("residue_type_ids must have shape [N]")
        if np.any(values["atomic_numbers"] <= 0):
            raise ValueError("atomic_numbers must contain positive element identifiers")

        curvatures = values["surface_curvatures"]
        weights    = values["surface_area_weights"]
        if curvatures.ndim != 3 or curvatures.shape[0] != surface_count or curvatures.shape[2] != 3:
            raise ValueError("surface_curvatures must have shape [M,S,3]")
        if surface_count == 0 or weights.shape != (surface_count,):
            raise ValueError("surface_area_weights must have non-empty shape [M]")
        if (
            not np.isfinite(curvatures).all()
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "surface curvature and area tensors must be finite with positive weights"
            )

        # Sparse topology remains compact; every endpoint is checked before tensor construction.
        atom_edges    = values["atom_edge_index"]
        surface_edges = values["surface_edge_index"]
        bipartite     = values["surface_atom_edge_index"]
        relation_mask = values["atom_edge_relation_mask"]
        for name, edge_index in (
            ("atom_edge_index", atom_edges),
            ("surface_edge_index", surface_edges),
            ("surface_atom_edge_index", bipartite),
        ):
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"{name} must have shape [2,E]")
            if edge_index.dtype.kind not in "iu":
                raise ValueError(f"{name} must use an integer dtype")
        if relation_mask.shape != (atom_edges.shape[1],) or not np.all(
            np.isin(relation_mask, (1, 2, 3))
        ):
            raise ValueError("atom relation masks must use preprocessing values 1, 2, or 3")
        if atom_edges.size and (
            atom_edges.min() < 0
            or atom_edges.max() >= atom_count
            or not np.all(atom_edges[0] < atom_edges[1])
        ):
            raise ValueError("atom_edge_index endpoints/order are invalid")
        if surface_edges.size and (
            surface_edges.min() < 0
            or surface_edges.max() >= surface_count
            or not np.all(surface_edges[0] < surface_edges[1])
        ):
            raise ValueError("surface_edge_index endpoints/order are invalid")
        if bipartite.size and (
            bipartite[0].min() < 0
            or bipartite[0].max() >= surface_count
            or bipartite[1].min() < 0
            or bipartite[1].max() >= atom_count
        ):
            raise ValueError("surface_atom_edge_index endpoints are invalid")

        output: dict[str, Tensor | str] = {
            "atomic_numbers": torch.from_numpy(values["atomic_numbers"].astype(np.int64)),
            "residue_type_ids": torch.from_numpy(values["residue_type_ids"].astype(np.int64)),
            "atom_edge_index": torch.from_numpy(atom_edges.astype(np.int64)),
            "atom_edge_types": torch.from_numpy(relation_mask.astype(np.int64) - 1),
            "surface_curvatures": torch.from_numpy(curvatures.astype(np.float32)),
            "surface_edge_index": torch.from_numpy(surface_edges.astype(np.int64)),
            "surface_atom_edge_index": torch.from_numpy(bipartite.astype(np.int64)),
            "surface_area_weights": torch.from_numpy(weights.astype(np.float32)),
            "target": torch.tensor(float(label), dtype=torch.float32),
        }
        for name in ("surface_positions", "surface_normals"):
            if name in values:
                output[name] = torch.from_numpy(values[name].astype(np.float32))
        if annotation_path is not None:
            base_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(annotation_path, allow_pickle=False) as annotation:
                annotation_required = {
                    "surface_target_hard",
                    "surface_valid_mask",
                    "surface_target_soft",
                    "surface_distance_to_dna",
                    "surface_distance_valid",
                    "surface_target_hard_sensitivity",
                    "sensitivity_gaps",
                    "base_npz_sha256",
                }
                missing_annotation = annotation_required - set(annotation.files)
                if missing_annotation:
                    raise ValueError(
                        f"{annotation_path.name} is missing arrays: {sorted(missing_annotation)}"
                    )
                if str(annotation["base_npz_sha256"].item()) != base_digest:
                    raise ValueError("annotation sidecar fingerprint does not match the base NPZ")
                for name in annotation_required - {
                    "base_npz_sha256",
                    "surface_target_hard_sensitivity",
                    "sensitivity_gaps",
                }:
                    if annotation[name].shape != (surface_count,):
                        raise ValueError(f"annotation array {name} must have shape [M]")
                sensitivity = annotation["surface_target_hard_sensitivity"]
                gaps        = annotation["sensitivity_gaps"]
                if sensitivity.ndim != 2 or sensitivity.shape[0] != surface_count:
                    raise ValueError("surface sensitivity targets must have shape [M,T]")
                if gaps.shape != (sensitivity.shape[1],):
                    raise ValueError("sensitivity_gaps must have shape [T]")
                output.update(
                    {
                        "surface_target_hard": torch.from_numpy(
                            annotation["surface_target_hard"].astype(np.int64)
                        ),
                        "surface_valid_mask": torch.from_numpy(
                            annotation["surface_valid_mask"].astype(np.bool_)
                        ),
                        "surface_target_soft": torch.from_numpy(
                            annotation["surface_target_soft"].astype(np.float32)
                        ),
                        "surface_distance_to_dna": torch.from_numpy(
                            annotation["surface_distance_to_dna"].astype(np.float32)
                        ),
                        "surface_distance_valid": torch.from_numpy(
                            annotation["surface_distance_valid"].astype(np.bool_)
                        ),
                        "surface_target_hard_sensitivity": torch.from_numpy(
                            sensitivity.astype(np.int64)
                        ),
                        "sensitivity_gaps": torch.from_numpy(gaps.astype(np.float32)),
                    }
                )
        output["identifier"] = identifier
        output["tier"]       = tier
        return output
