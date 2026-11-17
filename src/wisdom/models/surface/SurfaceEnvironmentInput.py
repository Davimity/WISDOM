"""Training-only normalization and matched-capacity controls for masked free-space channels."""

import torch

from typing import Any
from torch import Tensor, nn
from collections.abc import Mapping, Iterable, Sequence


class SurfaceEnvironmentInput(nn.Module):
    """Add fixed physical information without copying local targets into model inputs."""

    FEATURES = ("travel_depth", "euclidean_depth", "occlusion_near", "occlusion_far",
                "empty_radius")
    mean: Tensor
    scale: Tensor
    fitted: Tensor

    def __init__(self, hidden_dim: int, features: Sequence[str] = ("travel_depth",),
                 mode: str = "real", standard_deviation_floor: float = 1e-6) -> None:
        """Declare a zero-initialized residual input branch and train-fitted scalar statistics.

        Args:
            hidden_dim: Unchanged surface latent width H.
            features: Ordered distinct physical channel names; masks accompany each value.
            mode: real, mask_only or curvature_control, with identical projection dimensions.
            standard_deviation_floor: Positive floor for valid training population deviations.

        Raises:
            ValueError: Unknown/repeated features, unsupported control or empty source set.
        """
        super().__init__()
        if not features or set(features) - set(self.FEATURES) or (
            len(set(features)) != len(features)
        ):
            raise ValueError("environment features must be distinct supported physical sources")
        if mode not in {"real", "mask_only", "curvature_control"}:
            raise ValueError("environment mode must be real, mask_only or curvature_control")
        self.features, self.mode, self.floor = tuple(features), mode, standard_deviation_floor
        self.register_buffer("mean", torch.zeros(len(features)))
        self.register_buffer("scale", torch.ones(len(features)))
        self.register_buffer("fitted", torch.tensor(False))
        self.projection = nn.Linear(2 * len(features), hidden_dim, bias=False)
        nn.init.zeros_(self.projection.weight)

    def raw(self, members: Sequence[Mapping[str, Any]]) -> tuple[Tensor, Tensor]:
        """Join point values and masks while retaining exact per-protein point order.

        Args:
            members: Parent-validated environmental sidecars in collator protein order.

        Returns:
            Values [M,F] in physical units and boolean masks [M,F]. Missing support is not zero
            depth, and a failed source cannot be silently replaced by a different descriptor.
        """
        values = torch.cat([torch.stack([m[name] for name in self.features], dim=1)
                            for m in members])
        valid = torch.cat([torch.stack([m[name + "_valid"] for name in self.features], dim=1)
                           for m in members]).bool()
        return values, valid

    def fit(self, batches: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Fit stable FP64 moments on valid values of the selected training population only.

        Args:
            batches: Collated training members; validation/test and task targets are not read.

        Returns:
            Per-channel support, physical-unit mean and population standard deviation, saved
            together with buffers in the checkpoint for frozen inference and auditing.

        Raises:
            ValueError: A requested physical channel has no valid training support.
        """
        counts = torch.zeros(len(self.features), dtype=torch.float64)
        means, m2 = torch.zeros_like(counts), torch.zeros_like(counts)
        for batch in batches:
            values, valid = self.raw(batch["structural_assets"]["surface_environment"]["members"])
            for column in range(len(self.features)):
                x = values[valid[:, column], column].detach().double().cpu()
                if not len(x):
                    continue
                count = counts[column] + len(x)
                delta = x.mean() - means[column]
                m2[column] += (x-x.mean()).square().sum() + delta.square() * (
                    counts[column] * len(x) / count)
                means[column] += delta * len(x) / count
                counts[column] = count
        if (counts == 0).any():
            missing = [f for f, n in zip(self.features, counts, strict=True) if not n]
            raise ValueError(f"no valid selected-training environment support: {missing}")
        scales = (m2/counts).sqrt().clamp_min(self.floor)
        self.mean.copy_(means.float())
        self.scale.copy_(scales.float())
        self.fitted.fill_(True)
        return {f: {"count": int(n), "mean": float(a), "scale": float(b)}
                for f, n, a, b in zip(self.features, counts, means, scales, strict=True)}

    def forward(self, members: Sequence[Mapping[str, Any]], curvature: Tensor) -> Tensor:
        """Return an [M,H] correction; no independent predictor or pooling path is introduced.

        Args:
            members: Exact point-bound environmental members in collator order.
            curvature: Existing gated/transformed geometry [M,C], used only by the capacity control.

        Returns:
            Residual latent correction. Initially zero with a nonzero input, so the projection
            learns immediately. Invalid values remain masked after centering and scaling.

        Raises:
            RuntimeError: Normalization was not fitted on the authorized training population.
            ValueError: Sidecar rows differ from the active surface domain.
        """
        if not self.fitted:
            raise RuntimeError("surface environment requires selected-training normalization")
        values, valid = self.raw(members)
        if len(values) != len(curvature):
            raise ValueError("environmental values belong to a different surface point domain")
        normalized = (values-self.mean)/self.scale
        if self.mode == "mask_only":
            normalized = torch.zeros_like(normalized)
        elif self.mode == "curvature_control":
            columns = torch.arange(len(self.features), device=curvature.device) % curvature.shape[1]
            normalized = curvature[:, columns]
        return self.projection(torch.cat((normalized*valid, valid.float()), dim=1))
