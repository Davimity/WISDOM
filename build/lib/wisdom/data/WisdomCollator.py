"""Bounded disjoint batching for variable-size WISDOM proteins."""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


class WisdomCollator:
    """Activate nested K/J budgets and pack independent surface operators per protein."""

    def __init__(
        self,
        atom_spatial_k          : int = 16,
        surface_atom_k          : int = 16,
        diffusion_spectral_modes: int = 128,
        relation_mode           : str = "full_relational",
        curvature_scale_count   : int = 0,
    ) -> None:
        """Set runtime topology budgets without changing persisted scientific geometry.

        Args:
            atom_spatial_k: Active per-atom spatial rank ``K``; covalent edges always remain.
            surface_atom_k: Prefix width ``J`` selected from every compact atom-neighbor row.
            diffusion_spectral_modes: Maximum low-frequency modes selected per protein.
            relation_mode: ``full_relational``, ``unified_relation``, ``spatial_only``, or
                ``covalent_only``; this changes edge information, not message computation.
            curvature_scale_count: Number of smallest persisted scales to retain. Zero keeps all.

        Raises:
            ValueError: If any runtime budget is not positive.
        """
        if atom_spatial_k < 1 or surface_atom_k < 1 or diffusion_spectral_modes < 1:
            raise ValueError("collator K, J, and spectral-mode budgets must be positive")
        if relation_mode not in {
            "full_relational",
            "unified_relation",
            "spatial_only",
            "covalent_only",
        }:
            raise ValueError("unsupported atomic relation mode")
        if curvature_scale_count < 0:
            raise ValueError("curvature_scale_count cannot be negative")

        self.atom_spatial_k           = atom_spatial_k
        self.surface_atom_k           = surface_atom_k
        self.diffusion_spectral_modes = diffusion_spectral_modes
        self.relation_mode            = relation_mode
        self.curvature_scale_count    = curvature_scale_count

    def __call__(
        self,
        samples: Sequence[Mapping[str, Tensor | str]],
    ) -> Mapping[str, Any]:
        """Build one disconnected atomic batch and one ordered operator pack.

        Stored undirected atomic pairs are filtered by ``covalent OR spatial_rank<=K`` and expanded
        to both message directions. Compact surface-atom tables are sliced to their first ``J``
        columns and valid atom IDs receive the protein atom offset. Spectral and sparse gradient
        operators stay in a list aligned with ``surface_ptr``; concatenating them would create a
        large artificial block matrix with no scientific meaning.

        Args:
            samples: Non-empty ordered samples returned by :class:`WisdomDataset`.

        Returns:
            Concatenated atom/surface tensors, runtime relations, compact transfer tables,
            ``surface_ptr[B+1]``, per-protein operator dictionaries, optional geometry/targets, and
            one target per protein.

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
                "atom_edge_types",
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
                "target",
            )
        }
        annotation_names = (
            "surface_target_hard",
            "surface_valid_mask",
            "surface_target_soft",
            "surface_distance_to_dna",
            "surface_distance_valid",
            "surface_target_hard_sensitivity",
        )
        annotations: dict[str, list[Tensor]] = {name: [] for name in annotation_names}
        operators: list[dict[str, Tensor]] = []
        identifiers: list[str] = []
        tiers: list[str]       = []
        surface_ptr            = [0]

        has_surface_targets = all(
            "surface_target_hard" in sample and "surface_valid_mask" in sample
            for sample in samples
        )
        has_full_annotations = has_surface_targets and all(
            all(name in sample for name in annotation_names[2:]) for sample in samples
        )
        has_geometry = all("surface_positions" in sample for sample in samples)
        has_identity = all("identifier" in sample for sample in samples)

        atom_offset    = 0
        surface_offset = 0
        for batch_index, sample in enumerate(samples):
            atomic_numbers = self._tensor(sample, "atomic_numbers")
            surface_weights = self._tensor(sample, "surface_area_weights")
            atom_count      = len(atomic_numbers)
            surface_count   = len(surface_weights)

            values["atomic_numbers"].append(atomic_numbers)
            values["residue_type_ids"].append(self._tensor(sample, "residue_type_ids"))
            integer_features = ("atom_role_ids", "atom_hybridization_ids")
            scalar_features  = (
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
            values["atom_batch"].append(
                torch.full((atom_count,), batch_index, dtype=torch.long)
            )

            # Select one nested atomic topology and derive the closed relation IDs at runtime.

            stored_edges = self._tensor(sample, "atom_edge_index")
            covalent     = self._tensor(sample, "atom_edge_is_covalent").bool()
            ranks        = self._tensor(sample, "atom_edge_spatial_rank")
            spatial     = (ranks > 0) & (ranks <= self.atom_spatial_k)

            if self.relation_mode == "spatial_only":
                active = spatial
            elif self.relation_mode == "covalent_only":
                active = covalent
            else:
                active = covalent | spatial

            active_edges = stored_edges[:, active] + atom_offset
            if self.relation_mode == "full_relational":
                active_types = torch.where(
                    covalent[active] & spatial[active],
                    torch.full_like(ranks[active], 2),
                    torch.where(
                        covalent[active],
                        torch.ones_like(ranks[active]),
                        ranks[active] * 0,
                    ),
                ).long()
            else:
                active_types = torch.zeros_like(ranks[active], dtype=torch.long)
            values["atom_edge_index"].extend((active_edges, active_edges.flip(0)))
            values["atom_edge_types"].extend((active_types, active_types))

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
            if self.curvature_scale_count:
                if self.curvature_scale_count > curvatures.shape[1]:
                    raise ValueError(
                        "curvature_scale_count exceeds the persisted curvature-scale count"
                    )
                curvatures = curvatures[:, : self.curvature_scale_count]
            values["surface_curvatures"].append(curvatures)
            values["surface_atom_neighbors"].append(neighbors)
            values["surface_atom_distances"].append(
                self._tensor(sample, "surface_atom_distances")[:, : self.surface_atom_k]
            )
            values["surface_atom_normal_offsets"].append(
                self._tensor(sample, "surface_atom_normal_offsets")[:, : self.surface_atom_k]
            )
            values["surface_atom_tangential_distances"].append(
                self._tensor(sample, "surface_atom_tangential_distances")[
                    :, : self.surface_atom_k
                ]
            )
            values["surface_atom_mask"].append(atom_mask)
            values["surface_area_weights"].append(surface_weights)
            values["surface_batch"].append(
                torch.full((surface_count,), batch_index, dtype=torch.long)
            )

            # Operators keep local surface indices and are moved recursively by Training.

            eigenvalues = self._tensor(sample, "diffusion_eigenvalues")
            mode_count  = min(self.diffusion_spectral_modes, len(eigenvalues))
            operators.append(
                {
                    "mass": self._tensor(sample, "diffusion_mass"),
                    "eigenvalues": eigenvalues[:mode_count],
                    "eigenvectors": self._tensor(sample, "diffusion_eigenvectors")[
                        :, :mode_count
                    ],
                    "gradient_index": self._tensor(sample, "diffusion_gradient_index"),
                    "gradient_x": self._tensor(sample, "diffusion_gradient_x"),
                    "gradient_y": self._tensor(sample, "diffusion_gradient_y"),
                }
            )
            surface_ptr.append(surface_offset + surface_count)

            if has_geometry:
                values["surface_positions"].append(self._tensor(sample, "surface_positions"))
                values["surface_normals"].append(self._tensor(sample, "surface_normals"))
                local_neighbors = self._tensor(sample, "surface_neighbors").clone()
                local_mask      = self._tensor(sample, "surface_neighbor_mask").bool()
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
            if has_surface_targets:
                for name in annotation_names[:2]:
                    annotations[name].append(self._tensor(sample, name))
            if has_full_annotations:
                for name in annotation_names[2:]:
                    annotations[name].append(self._tensor(sample, name))

            atom_offset    += atom_count
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
                "atom_edge_types": torch.cat(values["atom_edge_types"]),
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
        if has_identity:
            batch["identifier"] = identifiers
            batch["tier"]       = tiers
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

        return batch

    @staticmethod
    def _optional_atom_tensor(
        sample   : Mapping[str, Tensor | str],
        name     : str,
        reference: Tensor,
        dtype    : torch.dtype,
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
