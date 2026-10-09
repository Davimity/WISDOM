"""Closed vocabulary for differentiable, pre-pooling surface evidence refinement."""

from enum import Enum


class SurfaceEvidenceRefinerType(str, Enum):
    """Select one evidence operator independently of the protein pooling family."""

    NONE                  = "none"
    HEAT                  = "heat"
    LEARNED_HEAT          = "learned_heat"
    GEOMETRIC_ANISOTROPIC = "geometric_anisotropic"
    EMBEDDING_ANISOTROPIC = "embedding_anisotropic"
    LEARNED_ANISOTROPIC   = "learned_anisotropic"
    GRAPH_TV              = "graph_tv"
    CRF                   = "crf"
