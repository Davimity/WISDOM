"""Read-only references to the existing disjoint surface representation; never local targets."""

import torch

from torch import Tensor
from dataclasses import dataclass
from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class SurfaceEvidenceContext:
    """Borrow batch tensors without copying geometry or exposing ground truth to refiners.

    Evidence Z is [M,D] (H, X or [H,X] according to representation mode).
    Areas/owners are [M], pointers [B+1], positions/normals [M,3],
    curvatures [M,S,3], and stored neighbor IDs/masks [M,K]. Coordinates use Å; operators
    contain mass-orthonormal eigenvectors and eigenvalues in Å^-2 for each protein.
    Optional geometry is unnecessary for identity and spectral heat.
    """

    evidence_features: Tensor
    area_weights : Tensor
    owners       : Tensor
    surface_ptr  : Tensor
    operators    : Sequence[Mapping[str, Tensor]]
    positions    : Tensor | None = None
    normals      : Tensor | None = None
    curvatures   : Tensor | None = None
    neighbors    : Tensor | None = None
    neighbor_mask: Tensor | None = None

    def edges(self) -> tuple[Tensor, Tensor]:
        """Take the undirected union of stored neighbors, restricted to each protein.

        Returns:
            Unique endpoint IDs (i,j), each [E] long, with i<j. Self edges, padding and
            cross-protein neighbors are excluded. No all-pairs geometry is constructed.

        Raises:
            ValueError: If a graph refiner lacks the stored neighbor IDs or validity mask.
        """
        if self.neighbors is None or self.neighbor_mask is None:
            raise ValueError("surface evidence refinement requires stored surface neighbors")
        count = len(self.owners)
        source = torch.arange(count, device=self.owners.device)[:, None]
        source = source.expand_as(self.neighbors)
        target = self.neighbors.clamp(0, count - 1)
        valid  = self.neighbor_mask & (self.neighbors >= 0) & (self.neighbors < count)
        valid  = valid & (source != target) & (self.owners[source] == self.owners[target])

        # Integer pair encoding deduplicates reciprocal neighbors in O(MK) storage, not M².

        left   = torch.minimum(source[valid], target[valid])
        right  = torch.maximum(source[valid], target[valid])
        pairs  = torch.unique(left * count + right)
        return pairs // count, pairs % count

    def geometry(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Build symmetric generic geometry on the selected sparse edges.

        Args:
            left: First endpoint IDs [E].
            right: Second endpoint IDs [E], same protein as left.

        Returns:
            Squared Euclidean distance in Å², normal misalignment 1-dot(n_i,n_j),
            and mean squared difference of signed-log curvature channels, all [E].
            Each stored channel is nondimensionalized by one in its own Å-based units
            before signed log1p; scales/channels contribute equally, without local labels.

        Raises:
            ValueError: If positions, normals or curvatures were not loaded.
        """
        if self.positions is None or self.normals is None or self.curvatures is None:
            raise ValueError("graph evidence refinement requires positions, normals and curvatures")
        positions  = self.positions.float()
        normals    = torch.nn.functional.normalize(self.normals.float(), dim=-1)
        curvatures = self.curvatures.float().flatten(1)
        curvature  = curvatures.sign() * torch.log1p(curvatures.abs())

        distance   = (positions[left] - positions[right]).square().sum(-1)
        alignment  = 1.0 - (normals[left] * normals[right]).sum(-1).clamp(-1.0, 1.0)
        difference = (curvature[left] - curvature[right]).square().mean(-1)
        return distance, alignment, difference

    def weights(
        self,
        left           : Tensor,
        right          : Tensor,
        geometry_sigma : float,
        normal_sigma   : float,
        curvature_sigma: float,
    ) -> Tensor:
        """Calculate symmetric geometric conductance, bounded in [0,1].

        Args:
            left: First edge endpoints [E].
            right: Second edge endpoints [E].
            geometry_sigma: Spatial bandwidth in Å, strictly positive.
            normal_sigma: Dimensionless normal-misalignment bandwidth, positive.
            curvature_sigma: Signed-log curvature-difference bandwidth, positive.

        Returns:
            exp(-d²/(2*sigma_x²)-(1-dot)/sigma_n-delta_k²/(2*sigma_k²)), [E].
        """
        distance, alignment, difference = self.geometry(left, right)
        exponent = (
            -distance / (2.0 * geometry_sigma ** 2)
            -alignment / normal_sigma
            -difference / (2.0 * curvature_sigma ** 2)
        )
        return exponent.exp()
