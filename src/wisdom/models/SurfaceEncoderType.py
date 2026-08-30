"""Closed surface-propagation hypotheses for WISDOM v3."""

from enum import Enum


class SurfaceEncoderType(str, Enum):
    """Name each controlled v3 replacement of the surface encoder."""

    DIFFUSION = "diffusion"
    DMASIF = "dmasif"
    DELTACONV = "deltaconv"
    PTV3 = "ptv3"
    POINT_MAMBA = "point_mamba"
