"""Sparse invariant surface messages, independently usable alongside diffusion or as a control."""

import torch

from torch import Tensor, nn
from lambdaforge.nn import Scatter
from collections.abc import Mapping


class SurfaceLocalMessages(nn.Module):
    """Use only the published filtered graph; never reconstruct a dense surface attention map."""

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "distance",
        sigma: float = 2.0,
        area_weighting: bool = False,
        initial_scale: float = 0.0,
        chunk_size: int = 65536,
        channel_attention: bool = False,
    ) -> None:
        """Build a nonzero message branch and one independently trainable residual multiplier.

        Args:
            hidden_dim: Shared scalar width H.
            mode: distance Gaussian weights or attention based on states and invariant geometry.
            sigma: Positive physical distance normalization/bandwidth in Å.
            area_weighting: Include sender represented area before normalization, default false.
            initial_scale: Initial gamma; zero recovers DiffusionNet and still learns at step one.
            chunk_size: Maximum pairs per message MLP evaluation, not a change in topology.
            channel_attention: Normalize an independent attention distribution per output channel.

        Raises:
            ValueError: Unsupported mode or nonpositive scale/chunk size.
        """
        super().__init__()
        if mode not in {"distance", "attention"} or sigma <= 0 or chunk_size < 1:
            raise ValueError("local surface mode/sigma/chunk_size is invalid")
        self.mode, self.sigma, self.chunk_size = mode, sigma, chunk_size
        self.area_weighting = area_weighting
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.scorer = (
            nn.Sequential(
                nn.Linear(2 * hidden_dim + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim if channel_attention else 1),
            )
            if mode == "attention"
            else None
        )
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, features: Tensor, operator: Mapping[str, Tensor]) -> Tensor:
        """Aggregate one molecule's directed messages and return only its residual increment.

        Args:
            features: Scalar [M,H] states; there is no cross-protein index domain.
            operator: local_edge_index [2,E] with each undirected pair once, positions/normals
                [M,3] and represented area [M]. All belong to this same published surface.

        Returns:
            gamma times the normalized local message [M,H], exactly zero for isolated points.
            Positivity of attention does not imply diffusion mass/energy conservation.
        """
        edges = operator["local_edge_index"].long()
        edges = torch.cat((edges, edges.flip(0)), dim=1)
        source, destination = edges
        positions, normals = operator["positions"], operator["normals"]
        relative = positions[source] - positions[destination]
        distance = relative.norm(dim=1)
        geometry = torch.stack(
            (
                distance / self.sigma,
                (normals[source] * normals[destination]).sum(1),
                (relative * normals[destination]).sum(1) / self.sigma,
                (relative * normals[source]).sum(1) / self.sigma,
            ),
            dim=1,
        )
        if not len(source):
            return torch.zeros_like(features)
        messages, scores = [], []
        for start in range(0, len(source), self.chunk_size):
            stop = start + self.chunk_size
            inputs = torch.cat(
                (
                    features[destination[start:stop]],
                    features[source[start:stop]],
                    geometry[start:stop],
                ),
                dim=1,
            )
            messages.append(self.message(inputs))
            scores.append(
                self.scorer(inputs)
                if self.scorer is not None
                else (-0.5 * (distance[start:stop] / self.sigma).square())[:, None]
            )
        logits = torch.cat(scores).float()
        if self.area_weighting:
            logits = logits + operator["area"][source].float().clamp_min(1e-30).log()[:, None]
        weights = Scatter.segment_softmax(logits, destination, len(features))
        result = Scatter.sum(weights * torch.cat(messages).float(), destination, len(features))
        return (self.scale * result).to(features.dtype)
