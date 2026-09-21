"""Ground-truth-free deletion and insertion audits for WISDOM surface evidence."""

from typing import Any, cast

import torch

from torch import Tensor

from wisdom.models.PoolingType import PoolingType


class SurfaceFaithfulnessAudit:
    """Measure whether a protein prediction actually depends on its highest-scored regions."""

    FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)

    def compute(
        self,
        model       : torch.nn.Module,
        output      : dict[str, Tensor],
        batch       : dict[str, Any],
    ) -> tuple[dict[str, float], int]:
        """Audit surface-derived positive predictions by equal-area deletion and insertion.

        For every globally positive protein, points are ordered by the declared local logit. A
        top, bottom, and deterministic-random region covering approximately the same represented
        area is removed before the model's existing pooling boundary. Deletion records the fall in
        positive probability. Insertion retains only the selected region and records the resulting
        probability. No local target is read, so these diagnostics remain available on tasks
        without surface annotations.

        Args:
            model: WISDOM model that produced ``output``. V2+ exposes ``pool_surface_logits``;
                the faithful V1 MAX boundary is handled directly.
            output: Model output containing local logits, surface embeddings, and protein logits.
            batch: Device batch containing area weights, owners, operators, pointers, and labels.

        Returns:
            Sum of each per-positive-protein audit metric and the number of positive proteins.
            Callers aggregate sums across batches before dividing by the count.

        Raises:
            ValueError: If the selected pooling requires a diffusion operator that cannot remain
                valid after deleting points, or if required point arrays are inconsistent.
        """
        pooling_type = PoolingType(
            getattr(getattr(model, "pooling_head", None), "pooling_type", PoolingType.MAX)
        )
        if pooling_type is PoolingType.LOCAL_MEAN_MAX:
            raise ValueError(
                "pooling-boundary deletion is undefined for local_mean_max because removing "
                "points changes its diffusion operator"
            )

        logits      = output["surface_logits"].reshape(-1)
        embeddings  = output["surface_embeddings"]
        areas       = cast(Tensor, batch["surface_area_weights"]).reshape(-1)
        owners      = cast(Tensor, batch["surface_batch"]).reshape(-1).long()
        surface_ptr = cast(Tensor, batch["surface_ptr"]).reshape(-1).long()
        targets     = cast(Tensor, batch["target"]).reshape(-1)
        protein_count = len(targets)

        if len(logits) != len(areas) or len(logits) != len(owners):
            raise ValueError("faithfulness audit requires aligned surface arrays")

        baseline_logits = output.get("surface_protein_logits", output["logits"])
        baseline        = torch.sigmoid(baseline_logits.reshape(-1))
        positives       = targets > 0.5
        positive_count  = int(positives.sum())
        if positive_count == 0:
            return {}, 0

        sums: dict[str, float] = {}
        for fraction in self.FRACTIONS:
            selected = {
                mode: self._area_region(logits, areas, owners, protein_count, fraction, mode)
                for mode in ("top", "random", "bottom")
            }
            deletion: dict[str, Tensor] = {}
            insertion: dict[str, Tensor] = {}
            for mode, region in selected.items():
                deletion[mode] = baseline - torch.sigmoid(
                    self._pool(model, logits, embeddings, areas, owners, surface_ptr, ~region)
                )
                insertion[mode] = torch.sigmoid(
                    self._pool(model, logits, embeddings, areas, owners, surface_ptr, region)
                )

            label = round(100 * fraction)
            for mode in selected:
                sums[f"faithfulness_deletion_{mode}_{label}"] = float(
                    deletion[mode][positives].sum()
                )
                sums[f"faithfulness_insertion_{mode}_{label}"] = float(
                    insertion[mode][positives].sum()
                )
            sums[f"faithfulness_deletion_top_minus_random_{label}"] = float(
                (deletion["top"] - deletion["random"])[positives].sum()
            )
            sums[f"faithfulness_insertion_top_minus_random_{label}"] = float(
                (insertion["top"] - insertion["random"])[positives].sum()
            )
        return sums, positive_count

    @staticmethod
    def _area_region(
        logits       : Tensor,
        areas        : Tensor,
        owners       : Tensor,
        protein_count: int,
        fraction     : float,
        mode         : str,
    ) -> Tensor:
        """Select one approximately equal-area region inside every protein.

        Args:
            logits: Local evidence ``[M]``.
            areas: Positive represented areas ``[M]``.
            owners: Point-to-protein owners ``[M]``.
            protein_count: Batch protein count.
            fraction: Target area fraction in ``(0,1)``.
            mode: ``top``, ``bottom``, or deterministic ``random`` ordering.

        Returns:
            Boolean point mask selecting at least one but never every point in each protein with
            more than one point.
        """
        selected = torch.zeros(len(logits), dtype=torch.bool, device=logits.device)
        indices  = torch.arange(len(logits), device=logits.device)
        for protein in range(protein_count):
            local = indices[owners == protein]
            if len(local) == 0:
                continue
            if mode == "top":
                order = local[torch.argsort(logits[local], descending=True)]
            elif mode == "bottom":
                order = local[torch.argsort(logits[local], descending=False)]
            else:
                keys  = torch.remainder(local * 1_103_515_245 + protein * 12_345, 2_147_483_647)
                order = local[torch.argsort(keys)]

            normalized = areas[order] / areas[order].sum().clamp_min(
                torch.finfo(areas.dtype).eps
            )
            count = int((normalized.cumsum(0) < fraction).sum()) + 1
            count = min(count, max(1, len(order) - 1))
            selected[order[:count]] = True
        return selected

    @staticmethod
    def _pool(
        model       : torch.nn.Module,
        logits      : Tensor,
        embeddings  : Tensor,
        areas       : Tensor,
        owners      : Tensor,
        surface_ptr : Tensor,
        retained    : Tensor,
    ) -> Tensor:
        """Apply the model's unchanged pooling rule to a retained point subset.

        Args:
            model: WISDOM model with either V1 MAX or a public V2 pooling boundary.
            logits: Local evidence ``[M]``.
            embeddings: Surface representation ``[M,H]``.
            areas: Represented area ``[M]``.
            owners: Protein owners ``[M]``.
            surface_ptr: Original point prefix boundaries ``[B+1]``.
            retained: Boolean points kept after deletion or for insertion.

        Returns:
            Surface-derived protein logits ``[B]`` under the same pooling family.
        """
        filtered_owners = owners[retained]
        protein_count   = len(surface_ptr) - 1
        counts          = torch.bincount(filtered_owners, minlength=protein_count)
        filtered_ptr    = torch.cat(
            (
                counts.new_zeros(1),
                counts.cumsum(0),
            )
        )
        pooling = getattr(model, "pool_surface_logits", None)
        if callable(pooling):
            result = pooling(
                logits[retained],
                embeddings[retained],
                areas[retained],
                filtered_owners,
                (),
                filtered_ptr,
            )
            return cast(dict[str, Tensor], result)["logits"]

        output = logits.new_full((protein_count,), float("-inf"))
        return output.scatter_reduce(
            0,
            filtered_owners,
            logits[retained],
            reduce="amax",
            include_self=True,
        )
