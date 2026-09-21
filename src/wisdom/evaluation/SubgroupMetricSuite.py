"""Stratified diagnostics for shortcut and dataset-dominance audits."""

from __future__ import annotations

import torch

from torch import Tensor
from collections.abc import Sequence

from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite


class SubgroupMetricSuite:
    """Measure global and local quality inside predefined validation strata."""

    MINIMUM_PHENOTYPE_MEMBERS = 8
    MINIMUM_LOCAL_MEMBERS     = 4

    def compute(
        self,
        protein_probabilities : Tensor,
        protein_targets       : Tensor,
        atom_counts           : Tensor,
        surface_counts        : Tensor,
        surface_probabilities : Tensor,
        surface_targets       : Tensor,
        surface_validity      : Tensor,
        surface_owners        : Tensor,
        global_phenotypes     : Sequence[str],
        interface_phenotypes  : Sequence[str],
        tiers                 : Sequence[str],
    ) -> dict[str, float | None]:
        """Compute size, surface-size, prevalence, phenotype, and tier strata.

        Protein-size and surface-size quartiles include both global classes, so their threshold-free
        global score ``G = 0.70*AUPRC + 0.30*AUROC`` is defined whenever a quartile contains both
        labels. Surface quality is the mean per-positive-protein point AUPRC inside the same
        stratum.
        Positive-site prevalence quartiles contain only positive proteins; their global score is
        intentionally unavailable because AUROC is mathematically undefined in a one-class group.

        Phenotype labels are emitted only when at least eight proteins or four locally evaluable
        positives support the estimate. This prevents a plot from presenting a single-protein value
        as a stable family result. Leakage groups are deliberately excluded: they enforce split
        independence and do not represent a biological phenotype.

        Args:
            protein_probabilities: Finite global probabilities with shape ``[B]``.
            protein_targets: Binary global labels with shape ``[B]``.
            atom_counts: Positive atom counts with shape ``[B]``.
            surface_counts: Positive surface-point counts with shape ``[B]``.
            surface_probabilities: Point probabilities with shape ``[M]``.
            surface_targets: Binary hard local targets with shape ``[M]``.
            surface_validity: Boolean ambiguity/availability mask with shape ``[M]``.
            surface_owners: Protein index for each point with shape ``[M]``.
            global_phenotypes: Dataset global-shape phenotype for every protein.
            interface_phenotypes: Positive-interface phenotype for every protein.
            tiers: Dataset difficulty tier for every protein.

        Returns:
            Flat scalar metric mapping. Every subgroup reports membership counts; scientifically
            undefined global or surface metrics remain ``None``.

        Raises:
            ValueError: If protein or point arrays are empty, misaligned, or non-finite.
        """
        probabilities = protein_probabilities.detach().view(-1).float().cpu()
        labels        = protein_targets.detach().view(-1).long().cpu()
        atoms         = atom_counts.detach().view(-1).long().cpu()
        points        = surface_counts.detach().view(-1).long().cpu()
        local_scores  = surface_probabilities.detach().view(-1).float().cpu()
        local_targets = surface_targets.detach().view(-1).long().cpu()
        valid         = surface_validity.detach().view(-1).bool().cpu()
        owners        = surface_owners.detach().view(-1).long().cpu()

        protein_count = len(labels)
        if protein_count == 0 or any(
            len(values) != protein_count
            for values in (probabilities, atoms, points)
        ):
            raise ValueError("subgroup metrics require aligned non-empty protein arrays")
        if any(
            len(values) != len(local_scores)
            for values in (local_targets, valid, owners)
        ):
            raise ValueError("subgroup metrics require aligned point arrays")
        if any(
            len(values) != protein_count
            for values in (global_phenotypes, interface_phenotypes, tiers)
        ):
            raise ValueError("subgroup metadata must align with proteins")
        if (
            not torch.isfinite(probabilities).all()
            or not torch.isfinite(local_scores).all()
            or torch.any((probabilities < 0.0) | (probabilities > 1.0))
            or torch.any((local_scores < 0.0) | (local_scores > 1.0))
            or torch.any((labels != 0) & (labels != 1))
            or torch.any((local_targets != 0) & (local_targets != 1))
            or torch.any(atoms <= 0)
            or torch.any(points <= 0)
        ):
            raise ValueError("subgroup metric inputs contain invalid values")
        if len(owners) and (owners.min() < 0 or owners.max() >= protein_count):
            raise ValueError("subgroup point owner is outside the protein vector")
        if not torch.equal(torch.bincount(owners, minlength=protein_count), points):
            raise ValueError("subgroup surface counts disagree with point ownership")

        # One point-level AUPRC and prevalence value is retained per evaluable positive protein.

        surface_auprc = torch.full((protein_count,), torch.nan, dtype=torch.float32)
        prevalence    = torch.full((protein_count,), torch.nan, dtype=torch.float32)
        binary_suite  = BinaryMetricSuite()

        for protein_index in range(protein_count):
            if labels[protein_index] != 1:
                continue
            mask = (owners == protein_index) & valid
            truth = local_targets[mask]
            if not len(truth) or truth.min() == truth.max():
                continue

            score = binary_suite.compute(local_scores[mask], truth, ("auprc",))["auprc"]
            if score is not None:
                surface_auprc[protein_index] = float(score)
                prevalence[protein_index]    = float(truth.float().mean())

        groups: dict[str, Tensor] = {}
        groups.update(self._quartiles("atom_count", atoms.float()))
        groups.update(self._quartiles("surface_count", points.float()))

        positive_indices = torch.isfinite(prevalence).nonzero(as_tuple=False).view(-1)
        if len(positive_indices):
            for name, local_mask in self._quartiles(
                "surface_prevalence",
                prevalence[positive_indices],
            ).items():
                mask = torch.zeros(protein_count, dtype=torch.bool)
                mask[positive_indices[local_mask]] = True
                groups[name] = mask

        groups.update(self._categorical_groups("tier", tiers, minimum=4))
        groups.update(
            self._categorical_groups(
                "global_phenotype",
                global_phenotypes,
                minimum=self.MINIMUM_PHENOTYPE_MEMBERS,
                local_scores=surface_auprc,
            )
        )
        groups.update(
            self._categorical_groups(
                "interface_phenotype",
                interface_phenotypes,
                minimum=self.MINIMUM_PHENOTYPE_MEMBERS,
                local_scores=surface_auprc,
            )
        )

        output: dict[str, float | None] = {}
        for name, mask in groups.items():
            self._measure_group(
                output,
                name,
                mask,
                probabilities,
                labels,
                surface_auprc,
                binary_suite,
            )
        return output

    @staticmethod
    def _quartiles(name: str, values: Tensor) -> dict[str, Tensor]:
        """Assign a finite numeric protein property to empirical quartile intervals.

        Args:
            name: Stable metric prefix describing the measured property.
            values: Finite one-dimensional property values.

        Returns:
            Non-empty Boolean masks named ``subgroup_<name>_q1`` through ``q4``. Equal boundary
            values remain together, so some intervals may be empty rather than splitting ties.

        Raises:
            ValueError: If values are empty, non-finite, or not one-dimensional.
        """
        if values.ndim != 1 or not len(values) or not torch.isfinite(values).all():
            raise ValueError("quartile values must be a finite non-empty vector")

        boundaries = torch.quantile(values, torch.tensor((0.25, 0.50, 0.75)))
        assignments = torch.bucketize(values, boundaries, right=False)
        return {
            f"subgroup_{name}_q{quartile + 1}": assignments == quartile
            for quartile in range(4)
            if torch.any(assignments == quartile)
        }

    @staticmethod
    def _categorical_groups(
        name        : str,
        values      : Sequence[str],
        minimum     : int,
        local_scores: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Create adequately supported groups from controlled dataset categories.

        Args:
            name: Stable metric-family prefix.
            values: One category string per protein.
            minimum: Minimum total group membership.
            local_scores: Optional per-protein local scores. A category below ``minimum`` is still
                retained when at least four locally evaluable positives support surface analysis.

        Returns:
            Boolean masks keyed by lower-case, metric-safe category names. ``unspecified``,
            ``unavailable`` and ``not_applicable`` categories are omitted.
        """
        output: dict[str, Tensor] = {}
        excluded = {"", "unspecified", "unavailable", "not_applicable"}
        for value in sorted(set(values)):
            if value.lower() in excluded:
                continue
            mask = torch.tensor([item == value for item in values], dtype=torch.bool)
            local_count = (
                int(torch.isfinite(local_scores[mask]).sum())
                if local_scores is not None
                else 0
            )
            enough_local = local_count >= SubgroupMetricSuite.MINIMUM_LOCAL_MEMBERS
            if int(mask.sum()) < minimum and not enough_local:
                continue

            safe = "_".join(
                part
                for part in "".join(
                    character.lower() if character.isalnum() else " "
                    for character in value
                ).split()
                if part
            )
            output[f"subgroup_{name}_{safe}"] = mask
        return output

    @staticmethod
    def _measure_group(
        output       : dict[str, float | None],
        name         : str,
        mask         : Tensor,
        probabilities: Tensor,
        labels       : Tensor,
        surface_auprc: Tensor,
        binary_suite : BinaryMetricSuite,
    ) -> None:
        """Append definition-aware global and surface metrics for one protein mask.

        Args:
            output: Result mapping updated in place.
            name: Stable subgroup prefix.
            mask: Boolean protein-membership vector.
            probabilities: Aligned global probabilities.
            labels: Aligned binary global labels.
            surface_auprc: Per-protein local AUPRC with ``NaN`` for unavailable values.
            binary_suite: Shared threshold-free binary metric implementation.
        """
        count          = int(mask.sum())
        group_labels   = labels[mask]
        local_available = mask & torch.isfinite(surface_auprc)

        output[f"{name}_count"]          = float(count)
        output[f"{name}_positive_count"] = float(group_labels.sum())

        global_metrics = binary_suite.compute(
            probabilities[mask],
            group_labels,
            ("auprc", "auroc"),
        )
        auprc = global_metrics["auprc"]
        auroc = global_metrics["auroc"]
        output[f"{name}_protein_auprc"] = auprc
        output[f"{name}_protein_auroc"] = auroc
        output[f"{name}_global_score"] = (
            0.70 * float(auprc) + 0.30 * float(auroc)
            if auprc is not None and auroc is not None
            else None
        )
        output[f"{name}_surface_count"] = float(local_available.sum())
        output[f"{name}_surface_auprc"] = (
            float(surface_auprc[local_available].mean())
            if torch.any(local_available)
            else None
        )
