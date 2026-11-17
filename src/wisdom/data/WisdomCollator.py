"""Bounded disjoint batching for variable-size WISDOM proteins."""

from __future__ import annotations

import torch

from typing import Any
from torch import Tensor
from collections.abc import Mapping, Sequence
from wisdom.data.StructuralInputs import StructuralInputs
from wisdom.data.AtomicStructuralInputs import AtomicStructuralInputs
from wisdom.preprocessing.common.structure.assets.graphs.ExtendedSurfaceAtom import (
    ExtendedSurfaceAtom,
)


class WisdomCollator:
    """Activate nested K/J budgets and pack independent surface operators per protein."""

    def __init__(
        self,
        atom_spatial_k: int = 16,
        surface_atom_k: int = 16,
        diffusion_spectral_modes: int = 128,
        transfer_source: str = "base",
        spectral_source: str = "base",
        surface_graph_source: str = "base",
    ) -> None:
        """Set runtime topology budgets without changing persisted scientific geometry.

        Args:
            atom_spatial_k: Active per-atom spatial rank ``K``; covalent edges always remain.
            surface_atom_k: Prefix width ``J`` selected from every compact atom-neighbor row.
            diffusion_spectral_modes: Maximum low-frequency modes selected per protein.
            transfer_source: base or explicitly required extended_surface_atom CSR candidates.
            spectral_source: base or explicitly required spectral_extension operators.
            surface_graph_source: base or filtered, attaching validated surface-local edges.
        Raises:
            ValueError: If any runtime budget is not positive.
        """
        if atom_spatial_k < 1 or surface_atom_k < 1 or diffusion_spectral_modes < 1:
            raise ValueError("collator K, J, and spectral-mode budgets must be positive")
        self.atom_spatial_k = atom_spatial_k
        self.surface_atom_k = surface_atom_k
        self.diffusion_spectral_modes = diffusion_spectral_modes
        if (
            transfer_source not in {"base", "extended"}
            or spectral_source not in {"base", "extension"}
            or surface_graph_source not in {"base", "filtered"}
        ):
            raise ValueError("unsupported explicit transfer/spectrum/surface-graph source")
        self.transfer_source, self.spectral_source = transfer_source, spectral_source
        self.surface_graph_source = surface_graph_source

    @staticmethod
    def initialize_worker(worker_id: int) -> None:
        """Use named shared memory when a DataLoader sends tensor-rich protein batches.

        Args:
            worker_id: DataLoader worker index, required by its initialization callback API.
                Transport is identical for every worker; this does not alter its native RNG seed.

        Returns:
            None. Changes only this worker's CPU-tensor sharing strategy to ``file_system``.
            PyTorch owns shared-memory allocation and cleanup through ``torch_shm_manager``.

        Per-protein operators and optional sidecars retain many independent tensor storages.
        The default ``file_descriptor`` strategy caches a descriptor per storage and can exhaust
        a 1024-descriptor limit while valid batches are prefetched. Named shared memory avoids
        those cached descriptors without changing tensor values, batching, or operating-system
        limits. Set it inside each worker so both forked and freshly spawned workers receive it.
        """
        torch.multiprocessing.set_sharing_strategy("file_system")  # type: ignore[no-untyped-call]

    def __call__(
        self,
        samples: Sequence[Mapping[str, Tensor | str]],
    ) -> Mapping[str, Any]:
        """Build one disconnected atomic batch and one ordered operator pack.

        Stored undirected atomic pairs are filtered by ``covalent OR spatial_rank<=K`` and remain
        compact. The atomic encoder evaluates both message directions directly, so duplicating
        indices and attributes here would consume twice the host/device transfer memory without
        changing a single message. Compact surface-atom tables are sliced to their first ``J``
        columns and valid atom IDs receive the protein atom offset. Spectral and sparse gradient
        operators stay in a list aligned with ``surface_ptr``; concatenating them would create a
        large artificial block matrix with no scientific meaning.

        Args:
            samples: Non-empty ordered samples returned by :class:`WisdomDataset`.

        Returns:
            Concatenated atom/surface tensors, runtime relations, compact transfer tables,
            ``surface_ptr[B+1]``, per-protein operator dictionaries, optional geometry/targets, and
            one target per protein. Prepared global atomic packs remain protein-local under
            ``atomic_operators`` and are never padded into a dense batch matrix.

        Raises:
            ValueError: If a budget exceeds persisted maxima or a required tensor is unavailable.
        """
        if not samples:
            raise ValueError("cannot collate an empty protein batch")

        values: dict[str, list[Tensor]] = {
            name: []
            for name in (
                "atomic_numbers",
                "residue_type_ids",
                "atom_role_ids",
                "atom_hybridization_ids",
                "formal_charges",
                "atom_aromaticity",
                "atom_hbond_donor",
                "atom_hbond_acceptor",
                "residue_hydropathy",
                "residue_polarity",
                "atom_edge_index",
                "atom_edge_is_spatial",
                "atom_edge_is_covalent",
                "atom_edge_distance",
                "atom_edge_bond_order",
                "atom_edge_same_residue",
                "atom_edge_same_chain",
                "atom_edge_residue_separation",
                "atom_batch",
                "surface_curvatures",
                "surface_atom_neighbors",
                "surface_atom_distances",
                "surface_atom_normal_offsets",
                "surface_atom_tangential_distances",
                "surface_atom_mask",
                "surface_area_weights",
                "surface_batch",
                "surface_positions",
                "surface_normals",
                "surface_neighbors",
                "surface_neighbor_distances",
                "surface_neighbor_mask",
                "atom_positions",
                "target",
            )
        }
        annotation_names = (
            "surface_target_hard",
            "surface_valid_mask",
            "surface_target_soft",
            "surface_distance_to_dna",
            "surface_distance_to_target",
            "surface_distance_valid",
            "surface_target_hard_sensitivity",
        )
        annotations: dict[str, list[Tensor]] = {name: [] for name in annotation_names}
        operators: list[dict[str, Tensor]] = []
        identifiers: list[str] = []
        tiers: list[str] = []
        leakage_groups: list[str] = []
        global_phenotypes: list[str] = []
        local_phenotypes: list[str] = []
        surface_ptr = [0]

        has_surface_targets = all(
            "surface_target_hard" in sample and "surface_valid_mask" in sample for sample in samples
        )
        has_full_annotations = has_surface_targets and all(
            all(name in sample for name in annotation_names[2:]) for sample in samples
        )
        has_geometry = all("surface_positions" in sample for sample in samples)
        has_atom_geometry = all("atom_positions" in sample for sample in samples)
        has_identity = all("identifier" in sample for sample in samples)

        atom_offset = 0
        surface_offset = 0
        for batch_index, sample in enumerate(samples):
            if self.transfer_source == "extended":
                # Registration checks limits; short dense rows retain -1/false padding.

                extended = {
                    key.split("__", 1)[1]: value.numpy()
                    for key, value in sample.items()
                    if key.startswith("extended_surface_atom__") and isinstance(value, Tensor)
                }
                dense = ExtendedSurfaceAtom.dense(extended, self.surface_atom_k)
                sample = {
                    **sample,
                    **{
                        key: torch.from_numpy(value).long()
                        if key == "surface_atom_neighbors"
                        else torch.from_numpy(value)
                        for key, value in dense.items()
                    },
                }
            atomic_numbers = self._tensor(sample, "atomic_numbers")
            surface_weights = self._tensor(sample, "surface_area_weights")
            atom_count = len(atomic_numbers)
            surface_count = len(surface_weights)

            values["atomic_numbers"].append(atomic_numbers)
            values["residue_type_ids"].append(self._tensor(sample, "residue_type_ids"))
            integer_features = ("atom_role_ids", "atom_hybridization_ids")
            scalar_features = (
                "formal_charges",
                "atom_aromaticity",
                "atom_hbond_donor",
                "atom_hbond_acceptor",
                "residue_hydropathy",
                "residue_polarity",
            )
            for name in integer_features:
                values[name].append(
                    self._optional_atom_tensor(sample, name, atomic_numbers, torch.long)
                )
            for name in scalar_features:
                values[name].append(
                    self._optional_atom_tensor(sample, name, atomic_numbers, torch.float32)
                )
            values["atom_batch"].append(torch.full((atom_count,), batch_index, dtype=torch.long))
            if has_atom_geometry:
                values["atom_positions"].append(self._tensor(sample, "atom_positions"))

            # Activate the nested spatial budget while retaining every covalent pair. The two
            # membership masks remain separate so one physical pair may feed both model branches.

            stored_edges = self._tensor(sample, "atom_edge_index")
            distances = self._tensor(sample, "atom_edge_distance")
            covalent = self._tensor(sample, "atom_edge_is_covalent").bool()
            ranks = self._tensor(sample, "atom_edge_spatial_rank")
            spatial = (ranks > 0) & (ranks <= self.atom_spatial_k)
            active = covalent | spatial

            active_edges = stored_edges[:, active] + atom_offset
            values["atom_edge_index"].append(active_edges)
            for name, feature in (
                ("atom_edge_is_spatial", spatial),
                ("atom_edge_is_covalent", covalent),
                ("atom_edge_bond_order", self._tensor(sample, "atom_edge_bond_order")),
                ("atom_edge_same_residue", self._tensor(sample, "atom_edge_same_residue")),
                ("atom_edge_same_chain", self._tensor(sample, "atom_edge_same_chain")),
                (
                    "atom_edge_residue_separation",
                    self._tensor(sample, "atom_edge_residue_separation"),
                ),
            ):
                values[name].append(feature[active])
            active_distances = distances[active].to(torch.float32)
            values["atom_edge_distance"].append(active_distances)

            # Slice the compact transfer table; invalid sentinels remain -1 after offsetting.

            stored_neighbors = self._tensor(sample, "surface_atom_neighbors")
            if self.surface_atom_k > stored_neighbors.shape[1]:
                raise ValueError(
                    f"surface_atom_k={self.surface_atom_k} exceeds persisted Jmax="
                    f"{stored_neighbors.shape[1]}"
                )
            atom_mask = self._tensor(sample, "surface_atom_mask")[:, : self.surface_atom_k].bool()
            neighbors = stored_neighbors[:, : self.surface_atom_k].clone()
            neighbors[atom_mask] += atom_offset

            curvatures = self._tensor(sample, "surface_curvatures")
            values["surface_curvatures"].append(curvatures)
            values["surface_atom_neighbors"].append(neighbors)
            values["surface_atom_distances"].append(
                self._tensor(sample, "surface_atom_distances")[:, : self.surface_atom_k]
            )
            values["surface_atom_normal_offsets"].append(
                self._tensor(sample, "surface_atom_normal_offsets")[:, : self.surface_atom_k]
            )
            values["surface_atom_tangential_distances"].append(
                self._tensor(sample, "surface_atom_tangential_distances")[:, : self.surface_atom_k]
            )
            values["surface_atom_mask"].append(atom_mask)
            values["surface_area_weights"].append(surface_weights)
            values["surface_batch"].append(
                torch.full((surface_count,), batch_index, dtype=torch.long)
            )

            # Operators keep local surface indices and are moved recursively by Training.

            eigenvalues = self._tensor(
                sample,
                "spectral_extension__eigenvalues"
                if self.spectral_source == "extension"
                else "diffusion_eigenvalues",
            )
            mode_count = min(self.diffusion_spectral_modes, len(eigenvalues))
            operators.append(
                {
                    "mass": self._tensor(sample, "diffusion_mass"),
                    "eigenvalues": eigenvalues[:mode_count],
                    "eigenvectors": self._tensor(
                        sample,
                        "spectral_extension__eigenvectors"
                        if self.spectral_source == "extension"
                        else "diffusion_eigenvectors",
                    )[:, :mode_count],
                    "gradient_index": self._tensor(sample, "diffusion_gradient_index"),
                    "gradient_x": self._tensor(sample, "diffusion_gradient_x"),
                    "gradient_y": self._tensor(sample, "diffusion_gradient_y"),
                }
            )
            if self.surface_graph_source == "filtered":
                operators[-1].update(
                    {
                        "local_edge_index": self._tensor(
                            sample, "surface_local_graph__surface_edge_index"
                        ),
                        "positions": self._tensor(sample, "surface_positions"),
                        "normals": self._tensor(sample, "surface_normals"),
                        "area": surface_weights,
                    }
                )
            surface_ptr.append(surface_offset + surface_count)

            if has_geometry:
                values["surface_positions"].append(self._tensor(sample, "surface_positions"))
                values["surface_normals"].append(self._tensor(sample, "surface_normals"))
                local_neighbors = self._tensor(sample, "surface_neighbors").clone()
                local_mask = self._tensor(sample, "surface_neighbor_mask").bool()
                local_neighbors[local_mask] += surface_offset
                values["surface_neighbors"].append(local_neighbors)
                values["surface_neighbor_distances"].append(
                    self._tensor(sample, "surface_neighbor_distances")
                )
                values["surface_neighbor_mask"].append(local_mask)

            values["target"].append(self._tensor(sample, "target"))
            if has_identity:
                identifiers.append(str(sample["identifier"]))
                tiers.append(str(sample["tier"]))
                leakage_groups.append(str(sample["leakage_group"]))
                global_phenotypes.append(str(sample["global_phenotype"]))
                local_phenotypes.append(str(sample["local_phenotype"]))
            if has_surface_targets:
                for name in annotation_names[:2]:
                    annotations[name].append(self._tensor(sample, name))
            if has_full_annotations:
                for name in annotation_names[2:]:
                    annotations[name].append(self._tensor(sample, name))

            atom_offset += atom_count
            surface_offset += surface_count

        tensor_names = (
            "atomic_numbers",
            "residue_type_ids",
            "atom_role_ids",
            "atom_hybridization_ids",
            "formal_charges",
            "atom_aromaticity",
            "atom_hbond_donor",
            "atom_hbond_acceptor",
            "residue_hydropathy",
            "residue_polarity",
            "atom_batch",
            "surface_curvatures",
            "surface_atom_neighbors",
            "surface_atom_distances",
            "surface_atom_normal_offsets",
            "surface_atom_tangential_distances",
            "surface_atom_mask",
            "surface_area_weights",
            "surface_batch",
        )
        batch: dict[str, Any] = {name: torch.cat(values[name]) for name in tensor_names}
        batch.update(
            {
                "atom_edge_index": torch.cat(values["atom_edge_index"], dim=1),
                "atom_edge_is_spatial": torch.cat(values["atom_edge_is_spatial"]).bool(),
                "atom_edge_is_covalent": torch.cat(values["atom_edge_is_covalent"]).bool(),
                "atom_edge_distance": torch.cat(values["atom_edge_distance"]),
                "atom_edge_bond_order": torch.cat(values["atom_edge_bond_order"]).float(),
                "atom_edge_same_residue": torch.cat(values["atom_edge_same_residue"]).bool(),
                "atom_edge_same_chain": torch.cat(values["atom_edge_same_chain"]).bool(),
                "atom_edge_residue_separation": torch.cat(
                    values["atom_edge_residue_separation"]
                ).float(),
                "surface_ptr": torch.tensor(surface_ptr, dtype=torch.long),
                "surface_operators": operators,
                "target": torch.stack(values["target"]),
                "active_atom_spatial_k": torch.tensor(self.atom_spatial_k),
                "active_surface_atom_k": torch.tensor(self.surface_atom_k),
            }
        )
        if has_geometry:
            for name in (
                "surface_positions",
                "surface_normals",
                "surface_neighbors",
                "surface_neighbor_distances",
                "surface_neighbor_mask",
            ):
                batch[name] = torch.cat(values[name])
        if has_atom_geometry:
            batch["atom_positions"] = torch.cat(values["atom_positions"])
        if "atom_residue_indices" in samples[0]:
            # Residue addresses stay protein-local, paired with explicit atom_batch ownership.

            batch["atom_residue_indices"] = torch.cat(
                [self._tensor(sample, "atom_residue_indices") for sample in samples]
            )
        if any("surface_explicit_features" in sample for sample in samples):
            # All members must expose exactly the requested channel order; a missing member
            # cannot silently become an all-zero explicit representation.

            batch["surface_explicit_features"] = torch.cat(
                [self._tensor(sample, "surface_explicit_features") for sample in samples]
            )
        if has_identity:
            batch["identifier"] = identifiers
            batch["tier"] = tiers
            batch["leakage_group"] = leakage_groups
            batch["global_phenotype"] = global_phenotypes
            batch["local_phenotype"] = local_phenotypes
        if has_surface_targets:
            for name in annotation_names[:2]:
                batch[name] = torch.cat(annotations[name])
        if has_full_annotations:
            for name in annotation_names[2:]:
                batch[name] = torch.cat(annotations[name])
            sensitivity_gaps = self._tensor(samples[0], "sensitivity_gaps")
            if any(
                not torch.equal(self._tensor(sample, "sensitivity_gaps"), sensitivity_gaps)
                for sample in samples[1:]
            ):
                raise ValueError("all DNA sidecars in a batch must use the same sensitivity gaps")
            batch["sensitivity_gaps"] = sensitivity_gaps

        batch.update(StructuralInputs.collate(samples))
        if any("atomic_structure__spatial_k" in sample for sample in samples):
            batch["atomic_operators"] = [
                AtomicStructuralInputs.pack(sample, self.atom_spatial_k) for sample in samples
            ]
        return batch

    @staticmethod
    def _optional_atom_tensor(
        sample: Mapping[str, Tensor | str],
        name: str,
        reference: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        """Read one generic atom descriptor with a legacy zero fallback.

        The fallback keeps direct callers that construct old in-memory samples operational. Real
        schema-3 datasets always provide the descriptors through :class:`WisdomDataset`, so an
        enabled modern feature never silently loses information in managed training.

        Args:
            sample: One protein tensor mapping.
            name: Descriptor field to read.
            reference: Atomic-number vector defining atom count and device.
            dtype: Required output dtype.

        Returns:
            Existing descriptor ``[N]`` or a zero vector with the same atom count.

        Raises:
            ValueError: If an existing descriptor is not a tensor with shape ``[N]``.
        """
        value = sample.get(name)
        if value is None:
            return torch.zeros(reference.shape, dtype=dtype, device=reference.device)
        if not isinstance(value, Tensor) or value.shape != reference.shape:
            raise ValueError(f"{name} must be a tensor with shape [N]")
        return value.to(dtype=dtype)

    @staticmethod
    def _tensor(mapping: Mapping[str, Any], name: str) -> Tensor:
        """Return a required tensor field with a precise collator error.

        Args:
            mapping: Sample or partially assembled batch mapping.
            name: Required tensor field name.

        Returns:
            Tensor stored under ``name``.

        Raises:
            ValueError: If the field is absent or is not a tensor.
        """
        value = mapping.get(name)
        if not isinstance(value, Tensor):
            raise ValueError(f"collator field {name!r} must be a tensor")
        return value
