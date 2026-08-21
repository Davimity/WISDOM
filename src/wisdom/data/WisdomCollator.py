"""Disjoint-graph batching for variable-size WISDOM proteins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


class WisdomCollator:
    """Concatenate proteins and offset all three sparse graph domains deterministically."""

    def __call__(
        self,
        samples: Sequence[Mapping[str, Tensor | str]],
    ) -> Mapping[str, Tensor | list[str]]:
        """Build one disconnected tensor graph batch from ordered protein samples.

        Preprocessing stores an undirected atomic or surface pair once with ``src<dst``. LambdaForge
        graph encoders consume directed ``source -> destination`` edges, so each stored pair is
        expanded to both orientations here. Bipartite edges retain their semantic orientation
        ``surface -> atom`` because WISDOMv1 gathers atom embeddings explicitly by its two rows.

        Args:
            samples: Non-empty ordered sequence returned by ``WisdomDataset``. Protein ``b`` becomes
                batch index ``b`` without random reordering inside this operation.

        Returns:
            Mapping of concatenated features, bidirectional atom/surface graph edges, offset
            surface-to-atom incidence, ``atom_batch[N]``, ``surface_batch[M]`` and ``target[B]``.

        Raises:
            ValueError: If ``samples`` is empty or any offset endpoint leaves its concatenated node
                domain, indicating an inconsistent dataset/collator contract.
        """
        if not samples:
            raise ValueError("cannot collate an empty protein batch")

        atomic_numbers: list[Tensor]          = []
        residue_type_ids: list[Tensor]        = []
        atom_edges: list[Tensor]              = []
        atom_edge_types: list[Tensor]         = []
        atom_batches: list[Tensor]            = []
        surface_curvatures: list[Tensor]      = []
        surface_edges: list[Tensor]           = []
        bipartite_edges: list[Tensor]         = []
        surface_weights: list[Tensor]         = []
        surface_positions: list[Tensor]       = []
        surface_normals: list[Tensor]         = []
        surface_batches: list[Tensor]         = []
        targets: list[Tensor]                 = []
        identifiers: list[str]                = []
        tiers: list[str]                      = []
        annotations: dict[str, list[Tensor]]  = {
            name: []
            for name in (
                "surface_target_hard",
                "surface_valid_mask",
                "surface_target_soft",
                "surface_distance_to_dna",
                "surface_distance_valid",
                "surface_target_hard_sensitivity",
            )
        }
        has_annotations = all("surface_target_hard" in sample for sample in samples)
        has_geometry    = all("surface_positions" in sample for sample in samples)
        has_identity    = all("identifier" in sample for sample in samples)

        atom_offset    = 0
        surface_offset = 0
        for batch_index, sample in enumerate(samples):
            atomic_number_value = sample["atomic_numbers"]
            surface_weight_value = sample["surface_area_weights"]
            if not isinstance(atomic_number_value, Tensor) or not isinstance(
                surface_weight_value, Tensor
            ):
                raise ValueError("collator tensor fields must contain tensors")
            atom_count    = len(atomic_number_value)
            surface_count = len(surface_weight_value)

            atomic_numbers.append(atomic_number_value)
            residue_type_ids.append(self._tensor(sample, "residue_type_ids"))
            atom_batches.append(
                torch.full((atom_count,), batch_index, dtype=torch.long)
            )

            stored_atom_edges = self._tensor(sample, "atom_edge_index") + atom_offset
            atom_edges.extend((stored_atom_edges, stored_atom_edges.flip(0)))
            atom_types = self._tensor(sample, "atom_edge_types")
            atom_edge_types.extend((atom_types, atom_types))

            surface_curvatures.append(self._tensor(sample, "surface_curvatures"))
            surface_weights.append(surface_weight_value)
            if has_geometry:
                surface_positions.append(self._tensor(sample, "surface_positions"))
                surface_normals.append(self._tensor(sample, "surface_normals"))
            surface_batches.append(
                torch.full((surface_count,), batch_index, dtype=torch.long)
            )

            stored_surface_edges = self._tensor(sample, "surface_edge_index") + surface_offset
            surface_edges.extend((stored_surface_edges, stored_surface_edges.flip(0)))

            bipartite_offset = torch.tensor(
                [[surface_offset], [atom_offset]],
                dtype=torch.long,
            )
            bipartite_edges.append(
                self._tensor(sample, "surface_atom_edge_index") + bipartite_offset
            )
            targets.append(self._tensor(sample, "target"))
            if has_identity:
                identifiers.append(str(sample["identifier"]))
                tiers.append(str(sample["tier"]))
            if has_annotations:
                for name in annotations:
                    annotations[name].append(self._tensor(sample, name))

            atom_offset    += atom_count
            surface_offset += surface_count

        batch: dict[str, Tensor | list[str]] = {
            "atomic_numbers": torch.cat(atomic_numbers),
            "residue_type_ids": torch.cat(residue_type_ids),
            "atom_edge_index": torch.cat(atom_edges, dim=1),
            "atom_edge_types": torch.cat(atom_edge_types),
            "atom_batch": torch.cat(atom_batches),
            "surface_curvatures": torch.cat(surface_curvatures),
            "surface_edge_index": torch.cat(surface_edges, dim=1),
            "surface_atom_edge_index": torch.cat(bipartite_edges, dim=1),
            "surface_area_weights": torch.cat(surface_weights),
            "surface_batch": torch.cat(surface_batches),
            "target": torch.stack(targets),
        }
        if has_geometry:
            batch["surface_positions"] = torch.cat(surface_positions)
            batch["surface_normals"]   = torch.cat(surface_normals)
        if has_identity:
            batch["identifier"] = identifiers
            batch["tier"]       = tiers
        if has_annotations:
            batch.update({name: torch.cat(values) for name, values in annotations.items()})
            sensitivity_gaps = self._tensor(samples[0], "sensitivity_gaps")
            if any(
                not torch.equal(self._tensor(sample, "sensitivity_gaps"), sensitivity_gaps)
                for sample in samples[1:]
            ):
                raise ValueError("all DNA sidecars in a batch must use the same sensitivity gaps")
            batch["sensitivity_gaps"] = sensitivity_gaps

        # Assertions remain active because an invalid offset would silently mix proteins.
        atom_edge_batch = self._tensor(batch, "atom_edge_index")
        if atom_edge_batch.numel() and (
            atom_edge_batch.min() < 0
            or atom_edge_batch.max() >= atom_offset
        ):
            raise ValueError("batched atom edge index is out of range")
        surface_edge_batch = self._tensor(batch, "surface_edge_index")
        if surface_edge_batch.numel() and (
            surface_edge_batch.min() < 0
            or surface_edge_batch.max() >= surface_offset
        ):
            raise ValueError("batched surface edge index is out of range")

        bipartite = self._tensor(batch, "surface_atom_edge_index")
        if bipartite.numel() and (
            bipartite[0].min() < 0
            or bipartite[0].max() >= surface_offset
            or bipartite[1].min() < 0
            or bipartite[1].max() >= atom_offset
        ):
            raise ValueError("batched surface-to-atom edge index is out of range")
        return batch

    @staticmethod
    def _tensor(mapping: Mapping[str, Tensor | str | list[str]], name: str) -> Tensor:
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
