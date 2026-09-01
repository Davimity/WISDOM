"""Definition-aware diagnostics for weakly supervised surface predictions."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite


class SurfaceMetricSuite:
    """Compare local model scores with evaluation-only DNA surface targets."""

    METRIC_NAMES = ("auprc", "auroc", "balanced_accuracy", "f1")

    def compute(
        self,
        probabilities  : Tensor,
        targets        : Tensor,
        valid_mask     : Tensor,
        surface_batch  : Tensor,
        protein_targets: Tensor,
    ) -> dict[str, float | None]:
        """Compute pooled-point and per-positive-protein localization diagnostics.

        ``surface_micro_*`` pools every unambiguous point before measuring performance. It tests
        whether local scores remain low on curated negative proteins as well as whether they find
        positive interfaces, but proteins with more points contribute more observations.
        ``surface_positive_macro_*`` first measures each globally positive protein independently
        and then takes an arithmetic mean, so every evaluable positive protein has equal weight.

        The point predictions are never transformed into a loss here. AUPRC and AUROC assess score
        ranking without choosing a threshold. Balanced accuracy and F1 apply probability threshold
        0.5; they are useful calibration diagnostics but are not model-selection objectives.

        Args:
            probabilities: Local sigmoid scores with finite shape ``[M]`` and values in ``[0,1]``.
            targets: Hard DNA-interface targets with binary shape ``[M]``.
            valid_mask: Boolean shape ``[M]`` excluding the physical ambiguity band and unavailable
                local ground truth.
            surface_batch: Integer owner of every point with shape ``[M]`` and values in ``[0,B)``.
            protein_targets: Global binary labels with shape ``[B]``.

        Returns:
            Micro and positive-protein macro AUPRC, AUROC, balanced accuracy, and F1, plus counts of
            valid points and evaluable positive proteins. Undefined metrics remain ``None``.

        Raises:
            ValueError: If arrays are misaligned, non-finite, non-binary, or reference an invalid
                protein owner.
        """
        scores   = probabilities.detach().view(-1).float().cpu()
        truth    = targets.detach().view(-1).long().cpu()
        valid    = valid_mask.detach().view(-1).bool().cpu()
        owners   = surface_batch.detach().view(-1).long().cpu()
        proteins = protein_targets.detach().view(-1).long().cpu()

        point_count = len(scores)
        if point_count == 0 or any(
            len(values) != point_count
            for values in (truth, valid, owners)
        ):
            raise ValueError("surface metrics require non-empty aligned point arrays")
        if not torch.isfinite(scores).all() or torch.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("surface probabilities must be finite values in [0,1]")
        if not torch.all((truth == 0) | (truth == 1)):
            raise ValueError("surface targets must contain only zero and one")
        if not len(proteins) or not torch.all((proteins == 0) | (proteins == 1)):
            raise ValueError("protein targets must be a non-empty binary vector")
        if owners.min() < 0 or owners.max() >= len(proteins):
            raise ValueError("surface point owner is outside the protein target vector")
        if not torch.all(owners[1:] >= owners[:-1]):
            raise ValueError("surface point owners must remain grouped in protein order")

        binary_suite = BinaryMetricSuite()
        output       = self._empty_output()
        valid_count  = int(valid.sum())

        output["surface_valid_points"] = float(valid_count)
        if valid_count:
            micro = binary_suite.compute(scores[valid], truth[valid], self.METRIC_NAMES)
            for name in self.METRIC_NAMES:
                output[f"surface_micro_{name}"] = micro[name]

        per_protein: dict[str, list[float]] = {
            name: []
            for name in self.METRIC_NAMES
        }
        owner_counts = torch.bincount(owners, minlength=len(proteins))
        owner_starts = torch.cat((torch.zeros(1, dtype=torch.long), owner_counts.cumsum(dim=0)))

        for protein_id in torch.nonzero(proteins == 1, as_tuple=False).view(-1).tolist():
            start          = int(owner_starts[protein_id])
            stop           = int(owner_starts[protein_id + 1])
            protein_valid  = valid[start:stop]
            if not protein_valid.any():
                continue

            metrics = binary_suite.compute(
                scores[start:stop][protein_valid],
                truth[start:stop][protein_valid],
                self.METRIC_NAMES,
            )
            if metrics["auprc"] is None:
                continue

            for name in self.METRIC_NAMES:
                value = metrics[name]
                if value is not None:
                    per_protein[name].append(value)

        evaluated = len(per_protein["auprc"])
        output["surface_positive_proteins"] = float(evaluated)
        for name, values in per_protein.items():
            if values:
                output[f"surface_positive_macro_{name}"] = math.fsum(values) / len(values)
        return output

    def _empty_output(self) -> dict[str, float | None]:
        """Create the stable local-metric schema before definition-aware calculation.

        Returns:
            Mapping containing every public surface metric as ``None`` and both counts as zero.
        """
        output: dict[str, float | None] = {
            "surface_valid_points":      0.0,
            "surface_positive_proteins": 0.0,
        }
        for name in self.METRIC_NAMES:
            output[f"surface_micro_{name}"]          = None
            output[f"surface_positive_macro_{name}"] = None
        return output
