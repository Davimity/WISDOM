"""Training-only standardization for frozen WISDOM surface embeddings."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


class EmbeddingScaler:
    """Store per-dimension train statistics without observing validation or test."""

    def __init__(self, mean: Tensor, scale: Tensor) -> None:
        """Create a reproducible affine embedding transform.

        Args:
            mean: Training mean with shape ``[H]``.
            scale: Positive training standard deviation with shape ``[H]``.

        Raises:
            ValueError: If vectors are misaligned, empty, non-finite, or non-positive.
        """
        mean  = mean.detach().float().cpu()
        scale = scale.detach().float().cpu()
        if mean.ndim != 1 or not len(mean) or scale.shape != mean.shape:
            raise ValueError("embedding scaler statistics must be aligned vectors [H]")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("embedding scaler statistics must be finite")
        if torch.any(scale <= 0.0):
            raise ValueError("embedding scaler scales must be positive")

        self.mean  = mean
        self.scale = scale

    @classmethod
    def fit(cls, training_embeddings: Tensor, epsilon: float = 1.0e-6) -> EmbeddingScaler:
        """Estimate independent feature statistics from training points only.

        Args:
            training_embeddings: Frozen WISDOM representations ``float [N,H]`` from train.
            epsilon: Minimum standard deviation used for a constant dimension.

        Returns:
            Fitted scaler. Validation and test do not participate in its statistics.

        Raises:
            ValueError: If embeddings or ``epsilon`` are invalid.
        """
        if training_embeddings.ndim != 2 or not len(training_embeddings):
            raise ValueError("training embeddings must have non-empty shape [N,H]")
        if epsilon <= 0.0 or not torch.isfinite(training_embeddings).all():
            raise ValueError("embedding scaler input or epsilon is invalid")

        values = training_embeddings.detach().float().cpu()
        mean   = values.mean(dim=0)
        scale  = values.std(dim=0, correction=0).clamp_min(epsilon)
        return cls(mean, scale)

    def transform(self, embeddings: Tensor) -> Tensor:
        """Standardize embeddings on their current device.

        Args:
            embeddings: Representations with final width ``H``.

        Returns:
            Values ``(h-mean)/scale`` with the same shape and device.

        Raises:
            ValueError: If the final width differs from the fitted width.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != len(self.mean):
            raise ValueError("embedding width disagrees with the fitted scaler")
        mean  = self.mean.to(device=embeddings.device, dtype=embeddings.dtype)
        scale = self.scale.to(device=embeddings.device, dtype=embeddings.dtype)
        return (embeddings - mean) / scale

    def inverse(self, standardized: Tensor) -> Tensor:
        """Recover the original WISDOM embedding coordinate system.

        Args:
            standardized: Standardized representations ``[N,H]``.

        Returns:
            Reconstructed physical model embeddings ``[N,H]``.
        """
        if standardized.ndim != 2 or standardized.shape[1] != len(self.mean):
            raise ValueError("standardized embedding width disagrees with the scaler")
        mean  = self.mean.to(device=standardized.device, dtype=standardized.dtype)
        scale = self.scale.to(device=standardized.device, dtype=standardized.dtype)
        return standardized * scale + mean

    def state(self) -> dict[str, Tensor]:
        """Return the pickle-free tensor state stored with concept checkpoints.

        Returns:
            CPU ``mean`` and ``scale`` vectors.
        """
        return {"mean": self.mean.clone(), "scale": self.scale.clone()}

    @classmethod
    def from_state(cls, state: Mapping[str, Tensor]) -> EmbeddingScaler:
        """Restore a scaler from its serialized tensors.

        Args:
            state: Mapping containing ``mean`` and ``scale`` tensors.

        Returns:
            Restored scaler.

        Raises:
            ValueError: If either tensor is absent.
        """
        if "mean" not in state or "scale" not in state:
            raise ValueError("embedding scaler state requires mean and scale")
        return cls(state["mean"], state["scale"])
