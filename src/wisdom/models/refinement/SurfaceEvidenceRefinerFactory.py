"""Central construction of the eight orthogonal surface evidence operators."""

from typing import Any
from collections.abc import Callable
from wisdom.models.refinement.CRFEvidenceRefiner import CRFEvidenceRefiner
from wisdom.models.refinement.HeatEvidenceRefiner import HeatEvidenceRefiner
from wisdom.models.refinement.NoOpEvidenceRefiner import NoOpEvidenceRefiner
from wisdom.models.refinement.GraphTVEvidenceRefiner import GraphTVEvidenceRefiner
from wisdom.models.refinement.SurfaceEvidenceRefiner import SurfaceEvidenceRefiner
from wisdom.models.refinement.LearnedHeatEvidenceRefiner import LearnedHeatEvidenceRefiner
from wisdom.models.refinement.SurfaceEvidenceRefinerType import SurfaceEvidenceRefinerType
from wisdom.models.refinement.LearnedAnisotropicEvidenceRefiner import (
    LearnedAnisotropicEvidenceRefiner,
)
from wisdom.models.refinement.GeometricAnisotropicEvidenceRefiner import (
    GeometricAnisotropicEvidenceRefiner,
)
from wisdom.models.refinement.EmbeddingAnisotropicEvidenceRefiner import (
    EmbeddingAnisotropicEvidenceRefiner,
)


class SurfaceEvidenceRefinerFactory:
    """Construct exactly one independent evidence operator without changing pooling."""

    @staticmethod
    def build(
        refiner_type: SurfaceEvidenceRefinerType | str,
        hidden_dim  : int,
        **parameters: Any,
    ) -> SurfaceEvidenceRefiner:
        """Construct one operator from public surface_refiner_* model parameters.

        Args:
            refiner_type: One of the eight explicit enum values.
            hidden_dim: Existing surface embedding width, used only by the tiny learned scorer.
            **parameters: Public model parameters with defaults fixed in WisdomV2/Training.

        Returns:
            A parameterless fixed operator or a learned operator owned by the model/optimizer.

        Raises:
            ValueError: If the enum or the selected operator's mathematical domain is invalid.
        """
        kind = SurfaceEvidenceRefinerType(refiner_type)
        common = {
            "strength":       parameters["surface_refiner_strength"],
            "steps":          parameters["surface_refiner_steps"],
            "geometry_sigma": parameters["surface_refiner_geometry_sigma"],
        }
        constructors: dict[SurfaceEvidenceRefinerType, Callable[[], SurfaceEvidenceRefiner]] = {
            SurfaceEvidenceRefinerType.NONE: lambda: NoOpEvidenceRefiner(),
            SurfaceEvidenceRefinerType.HEAT: lambda: HeatEvidenceRefiner(
                parameters["surface_refiner_heat_length"],
            ),
            SurfaceEvidenceRefinerType.LEARNED_HEAT: lambda: LearnedHeatEvidenceRefiner(
                parameters["surface_refiner_heat_length_init"],
            ),
            SurfaceEvidenceRefinerType.GEOMETRIC_ANISOTROPIC: lambda:
                GeometricAnisotropicEvidenceRefiner(
                    **common, normal_sigma=parameters["surface_refiner_normal_sigma"],
                    curvature_sigma=parameters["surface_refiner_curvature_sigma"],
                ),
            SurfaceEvidenceRefinerType.EMBEDDING_ANISOTROPIC: lambda:
                EmbeddingAnisotropicEvidenceRefiner(
                    **common, temperature=parameters["surface_refiner_embedding_temperature"],
                    embedding_detach=parameters["surface_refiner_embedding_detach"],
                ),
            SurfaceEvidenceRefinerType.LEARNED_ANISOTROPIC: lambda:
                LearnedAnisotropicEvidenceRefiner(
                    **common, embedding_dim=hidden_dim,
                    hidden_dim=parameters["surface_refiner_learned_hidden_dim"],
                ),
            SurfaceEvidenceRefinerType.GRAPH_TV: lambda: GraphTVEvidenceRefiner(
                parameters["surface_refiner_tv_lambda"], parameters["surface_refiner_tv_steps"],
                parameters["surface_refiner_tv_step_size"],
                parameters["surface_refiner_tv_epsilon"],
            ),
            SurfaceEvidenceRefinerType.CRF: lambda: CRFEvidenceRefiner(
                parameters["surface_refiner_crf_strength"], parameters["surface_refiner_crf_steps"],
                parameters["surface_refiner_crf_damping"],
            ),
        }
        return constructors[kind]()
