"""Minimal pickle-free ingestion of labeled WISDOM NPZ representations."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from lambdaforge.data import DatasetIndex
from torch import Tensor
from torch.utils.data import Dataset


class WisdomDataset(Dataset[Mapping[str, Tensor | str]]):
    """Load one explicit labeled split without changing preprocessed geometry."""

    # Trainable WISDOM generations consume only the bounded schema-3 structural representation.

    STRUCTURAL_SCHEMA_VERSION = "3.0"

    LEGACY_COLUMNS = ("file", "label", "split")
    DNA_COLUMNS    = ("file", "annotation", "label", "split", "identifier", "tier")
    SPLITS         = frozenset({"train", "val", "test"})

    def __init__(
        self,
        manifest               : str | Path,
        split                  : str,
        subset                 : str = "full",
        include_surface_targets: bool = False,
        include_diagnostics    : bool = False,
        include_surface_geometry: bool = False,
    ) -> None:
        """Read and validate a compact ``file,label,split`` CSV manifest.

        Paths are resolved relative to the manifest directory. Rows keep CSV order and only rows
        whose explicit split equals ``split`` are retained; no random or implicit division occurs.
        NPZ files remain unopened until ``__getitem__`` so DataLoader workers do not share handles.

        Args:
            manifest: LambdaForge 0.13 managed dataset root containing ``index.jsonl``, or a legacy
                CSV with exactly ``file,label,split`` columns.
            split: Explicit subset to expose; one of ``train``, ``val``, or ``test``.
            subset: ``full`` or a dilution name such as ``replicate-00/train-25``. Managed dataset
                members carry this view membership in metadata without duplicating heavy assets.
            include_surface_targets: Load only hard point targets and their validity mask for
                evaluation-only surface metrics.
            include_diagnostics: Load surface coordinates, normals, and DNA point targets needed
                by post-training localization analysis. Ordinary global-label training leaves this
                false to avoid hashing, decoding, collating, and transferring unused arrays.
            include_surface_geometry: Load coordinates, normals, and bounded local surface
                neighborhoods required by WISDOM v3 encoders. V1/V2 leave these arrays unopened.

        Raises:
            ValueError: If the split, header, label, row split, path, or selected subset is invalid.
            OSError: If the manifest cannot be read.
        """
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {sorted(self.SPLITS)}")
        if not subset.strip():
            raise ValueError("subset cannot be empty")

        self.include_surface_targets = include_surface_targets
        self.include_diagnostics     = include_diagnostics
        self.include_surface_geometry = include_surface_geometry

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
        managed_split = "validation" if split == "val" else split
        for member in DatasetIndex(index_path):
            if str(member.partitions.get("split", "")) != managed_split:
                continue
            dilutions = member.metadata.get("dilutions", ())
            if subset != "full" and (
                not isinstance(dilutions, (list, tuple)) or subset not in dilutions
            ):
                continue
            try:
                label      = int(member.targets["dna_binding"])
                base_asset = member.assets["universal_npz"]
                base       = root / base_asset.path
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"managed member {member.member_id!r} lacks WISDOM label/base asset"
                ) from error

            if label not in {0, 1}:
                raise ValueError(
                    f"managed member {member.member_id!r} has non-binary dna_binding={label!r}"
                )
            if not base.is_file():
                raise ValueError(
                    f"managed member {member.member_id!r} is missing universal_npz at "
                    f"{base_asset.path!r}"
                )
            if base_asset.kind != "file" or base_asset.media_type not in {
                None,
                "application/x-npz",
            }:
                raise ValueError(
                    f"managed member {member.member_id!r} universal_npz must be a file with "
                    "media_type='application/x-npz' when a media type is declared"
                )

            annotation_asset = member.assets.get("dna_annotation")
            annotation = root / annotation_asset.path if annotation_asset is not None else None
            if annotation is not None and not annotation.is_file():
                raise ValueError(
                    f"managed member {member.member_id!r} is missing dna_annotation at "
                    f"{annotation_asset.path!r}"
                )
            if annotation_asset is not None and (
                annotation_asset.kind != "file"
                or annotation_asset.media_type not in {None, "application/x-npz"}
            ):
                raise ValueError(
                    f"managed member {member.member_id!r} dna_annotation must be a file with "
                    "media_type='application/x-npz' when a media type is declared"
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

    def storage_bytes(self) -> dict[str, int]:
        """Measure unique persisted bytes selected by this dataset view.

        Filesystem metadata is sufficient here: the method does not open or hash any NPZ. Paths
        are deduplicated so repeated logical records cannot inflate preprocessing cost.

        Returns:
            Byte counts for universal NPZ files, DNA sidecars, and their total.

        Raises:
            OSError: If a selected asset disappears after dataset initialization.
        """
        base_paths = {record[0] for record in self.records}
        annotation_paths = {
            record[1]
            for record in self.records
            if record[1] is not None
        }

        base_bytes       = sum(path.stat().st_size for path in base_paths)
        annotation_bytes = sum(path.stat().st_size for path in annotation_paths)
        return {
            "universal_npz":  base_bytes,
            "dna_annotation": annotation_bytes,
            "total":          base_bytes + annotation_bytes,
        }

    def __getitem__(self, index: int) -> Mapping[str, Tensor | str]:
        """Load the bounded schema-3 model arrays and optional evaluation diagnostics.

        Atomic pairs remain stored once with ``src<dst`` together with covalent flags and spatial
        activation ranks. Surface-to-atom neighborhoods remain padded at ``Jmax``. Diffusion
        operators retain per-protein point order and are packed by :class:`WisdomCollator` rather
        than joined into an artificial block matrix.

        Args:
            index: Zero-based selected-split record index.

        Returns:
            Mapping containing bounded topology, compact atom-neighbor geometry, intrinsic surface
            operators, one scalar float32 target, and requested host-side diagnostics.

        Raises:
            IndexError: If ``index`` is outside the selected split.
            ValueError: If required arrays, shapes, finite values, relation semantics, or endpoint
                ranges are inconsistent with the current WISDOM NPZ contract.
            OSError: If the NPZ cannot be opened.
        """
        path, annotation_path, label, identifier, tier = self.records[index]
        required = {
            "metadata_json",
            "atomic_numbers",
            "residue_type_ids",
            "atom_edge_index",
            "atom_edge_is_covalent",
            "atom_edge_spatial_rank",
            "surface_curvatures",
            "surface_area_weights",
            "surface_atom_neighbors",
            "surface_atom_distances",
            "surface_atom_normal_offsets",
            "surface_atom_tangential_distances",
            "surface_atom_mask",
            "diffusion_mass",
            "diffusion_eigenvalues",
            "diffusion_eigenvectors",
            "diffusion_gradient_index",
            "diffusion_gradient_x",
            "diffusion_gradient_y",
        }
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise ValueError(
                    f"{path.name} uses an unsupported pre-schema-3 representation; "
                    "run WISDOM preprocessing again and publish a new DatasetVersion"
                )
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("preprocessing_schema_version") != self.STRUCTURAL_SCHEMA_VERSION:
                raise ValueError(
                    f"{path.name} uses structural schema "
                    f"{metadata.get('preprocessing_schema_version')!r}; WISDOM requires "
                    f"{self.STRUCTURAL_SCHEMA_VERSION}, "
                    "so the immutable dataset must be preprocessed again"
                )
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path.name} is missing schema-3 arrays: {sorted(missing)}")
            values = {name: archive[name] for name in required if name != "metadata_json"}
            if self.include_diagnostics or self.include_surface_geometry:
                geometry_names = {
                    "surface_positions",
                    "surface_normals",
                    "surface_neighbors",
                    "surface_neighbor_distances",
                    "surface_neighbor_mask",
                }
                missing_geometry = geometry_names - set(archive.files)
                if missing_geometry:
                    raise ValueError(
                        f"{path.name} is missing surface geometry: {sorted(missing_geometry)}"
                    )
                values.update({name: archive[name] for name in geometry_names})

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

        # Bounded atomic candidates remain inactive until the collator applies the requested K.
        atom_edges    = values["atom_edge_index"]
        is_covalent   = values["atom_edge_is_covalent"]
        spatial_rank  = values["atom_edge_spatial_rank"]
        if atom_edges.ndim != 2 or atom_edges.shape[0] != 2 or atom_edges.dtype.kind not in "iu":
            raise ValueError("atom_edge_index must be an integer array with shape [2,E]")
        if is_covalent.shape != (atom_edges.shape[1],) or is_covalent.dtype != np.bool_:
            raise ValueError("atom_edge_is_covalent must be Boolean with shape [E]")
        if spatial_rank.shape != (atom_edges.shape[1],) or spatial_rank.dtype.kind not in "iu":
            raise ValueError("atom_edge_spatial_rank must be integer with shape [E]")
        if np.any(~is_covalent & (spatial_rank == 0)):
            raise ValueError("an atomic edge is neither covalent nor spatial")
        if atom_edges.size and (
            atom_edges.min() < 0
            or atom_edges.max() >= atom_count
            or not np.all(atom_edges[0] < atom_edges[1])
        ):
            raise ValueError("atom_edge_index endpoints/order are invalid")

        neighbor_shape = values["surface_atom_neighbors"].shape
        if len(neighbor_shape) != 2 or neighbor_shape[0] != surface_count:
            raise ValueError("surface_atom_neighbors must have shape [M,Jmax]")
        for name in (
            "surface_atom_distances",
            "surface_atom_normal_offsets",
            "surface_atom_tangential_distances",
            "surface_atom_mask",
        ):
            if values[name].shape != neighbor_shape:
                raise ValueError(f"{name} must have shape [M,Jmax]")
        atom_neighbors = values["surface_atom_neighbors"]
        atom_mask      = values["surface_atom_mask"]
        if np.any(atom_mask.sum(axis=1) == 0) or np.any(atom_neighbors[~atom_mask] != -1):
            raise ValueError("surface atom masks and sentinels are inconsistent")
        if np.any(atom_neighbors[atom_mask] < 0) or np.any(
            atom_neighbors[atom_mask] >= atom_count
        ):
            raise ValueError("surface atom neighbor is out of range")

        mass         = values["diffusion_mass"]
        eigenvalues  = values["diffusion_eigenvalues"]
        eigenvectors = values["diffusion_eigenvectors"]
        gradient_index = values["diffusion_gradient_index"]
        if mass.shape != (surface_count,) or np.any(mass <= 0.0):
            raise ValueError("diffusion_mass must be positive with shape [M]")
        if eigenvectors.shape != (surface_count, len(eigenvalues)):
            raise ValueError("diffusion eigenvectors must have shape [M,Q]")
        if gradient_index.ndim != 2 or gradient_index.shape[0] != 2:
            raise ValueError("diffusion_gradient_index must have shape [2,G]")
        for name in ("diffusion_gradient_x", "diffusion_gradient_y"):
            if values[name].shape != (gradient_index.shape[1],):
                raise ValueError(f"{name} must have shape [G]")

        output: dict[str, Tensor | str] = {
            "atomic_numbers": torch.from_numpy(values["atomic_numbers"].astype(np.int64)),
            "residue_type_ids": torch.from_numpy(values["residue_type_ids"].astype(np.int64)),
            "atom_edge_index": torch.from_numpy(atom_edges.astype(np.int64)),
            "atom_edge_is_covalent": torch.from_numpy(is_covalent.astype(np.bool_)),
            "atom_edge_spatial_rank": torch.from_numpy(spatial_rank.astype(np.int64)),
            "surface_curvatures": torch.from_numpy(curvatures.astype(np.float32)),
            "surface_atom_neighbors": torch.from_numpy(atom_neighbors.astype(np.int64)),
            "surface_atom_distances": torch.from_numpy(
                values["surface_atom_distances"].astype(np.float32)
            ),
            "surface_atom_normal_offsets": torch.from_numpy(
                values["surface_atom_normal_offsets"].astype(np.float32)
            ),
            "surface_atom_tangential_distances": torch.from_numpy(
                values["surface_atom_tangential_distances"].astype(np.float32)
            ),
            "surface_atom_mask": torch.from_numpy(atom_mask.astype(np.bool_)),
            "surface_area_weights": torch.from_numpy(weights.astype(np.float32)),
            "diffusion_mass": torch.from_numpy(mass.astype(np.float32)),
            "diffusion_eigenvalues": torch.from_numpy(eigenvalues.astype(np.float32)),
            "diffusion_eigenvectors": torch.from_numpy(eigenvectors.astype(np.float32)),
            "diffusion_gradient_index": torch.from_numpy(gradient_index.astype(np.int64)),
            "diffusion_gradient_x": torch.from_numpy(
                values["diffusion_gradient_x"].astype(np.float32)
            ),
            "diffusion_gradient_y": torch.from_numpy(
                values["diffusion_gradient_y"].astype(np.float32)
            ),
            "target": torch.tensor(float(label), dtype=torch.float32),
        }
        for name in (
            "surface_positions",
            "surface_normals",
            "surface_neighbors",
            "surface_neighbor_distances",
            "surface_neighbor_mask",
        ):
            if name in values:
                dtype = np.int64 if name == "surface_neighbors" else (
                    np.bool_ if name == "surface_neighbor_mask" else np.float32
                )
                output[name] = torch.from_numpy(values[name].astype(dtype))
        if annotation_path is not None and (
            self.include_surface_targets or self.include_diagnostics
        ):
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
                # Dataset publication already verifies immutable asset checksums and preprocessing
                # validates this sidecar fingerprint. Rehashing a multi-megabyte NPZ on every
                # loader process/epoch would add I/O without strengthening the managed placement.
                if annotation["base_npz_sha256"].shape != ():
                    raise ValueError("annotation base fingerprint must be a scalar")
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
                    }
                )
                if self.include_diagnostics:
                    output.update(
                        {
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
