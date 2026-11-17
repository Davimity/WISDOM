"""Named weak-supervision profiles for controlled WISDOM loss screens."""

from enum import Enum


class WeakLossProfile(str, Enum):
    """Name weak-supervision candidates that avoid accidental factorial combinations."""

    CUSTOM              = "custom"
    BASELINE            = "baseline"
    NEGATIVE_010        = "negative_010"
    NEGATIVE_030        = "negative_030"
    NEGATIVE_100        = "negative_100"
    POSITIVE_EXIST_010  = "positive_exist_010"
    POSITIVE_EXIST_030  = "positive_exist_030"
    POSITIVE_EXIST_100  = "positive_exist_100"
    NEGATIVE_REGION_010 = "negative_region_010"
    NEGATIVE_REGION_030 = "negative_region_030"
    NEGATIVE_RANK_010   = "negative_rank_010"
    NEGATIVE_RANK_030   = "negative_rank_030"
    CARDINALITY_005     = "cardinality_005"
    CARDINALITY_010     = "cardinality_010"
    CARDINALITY_020     = "cardinality_020"
    TV_001              = "tv_001"
    TV_005              = "tv_005"
    DIRICHLET_0001      = "dirichlet_0001"
    DIRICHLET_001       = "dirichlet_001"

    def overrides(self) -> dict[str, float | None]:
        """Return the exact loss coefficients represented by this candidate.

        Returns:
            Training loss overrides. ``custom`` preserves every explicitly supplied coefficient;
            every other profile starts from global BCE and activates only its named hypothesis.
        """
        if self is self.CUSTOM:
            return {}

        baseline: dict[str, float | None] = {
            "negative_surface_lambda":   0.0,
            "positive_existence_lambda": 0.0,
            "regional_positive_lambda":  0.0,
            "regional_ranking_lambda":   0.0,
            "cardinality_lambda":        0.0,
            "minimum_positive_area":     0.0,
            "maximum_positive_area":     None,
            "total_variation_lambda":    0.0,
            "dirichlet_lambda":          0.0,
        }
        additions: dict[WeakLossProfile, dict[str, float | None]] = {
            self.BASELINE:            {},
            self.NEGATIVE_010:        {"negative_surface_lambda": 0.10},
            self.NEGATIVE_030:        {"negative_surface_lambda": 0.30},
            self.NEGATIVE_100:        {"negative_surface_lambda": 1.00},
            self.POSITIVE_EXIST_010:  {
                "negative_surface_lambda":   0.30,
                "positive_existence_lambda": 0.10,
            },
            self.POSITIVE_EXIST_030:  {
                "negative_surface_lambda":   0.30,
                "positive_existence_lambda": 0.30,
            },
            self.POSITIVE_EXIST_100:  {
                "negative_surface_lambda":   0.30,
                "positive_existence_lambda": 1.00,
            },
            self.NEGATIVE_REGION_010: {
                "negative_surface_lambda": 0.30,
                "regional_positive_lambda": 0.10,
            },
            self.NEGATIVE_REGION_030: {
                "negative_surface_lambda": 0.30,
                "regional_positive_lambda": 0.30,
            },
            self.NEGATIVE_RANK_010:   {
                "negative_surface_lambda": 0.30,
                "regional_ranking_lambda": 0.10,
            },
            self.NEGATIVE_RANK_030:   {
                "negative_surface_lambda": 0.30,
                "regional_ranking_lambda": 0.30,
            },
            self.CARDINALITY_005:     {
                "negative_surface_lambda": 0.30,
                "cardinality_lambda":      0.10,
                "minimum_positive_area":   0.005,
            },
            self.CARDINALITY_010:     {
                "negative_surface_lambda": 0.30,
                "cardinality_lambda":      0.10,
                "minimum_positive_area":   0.010,
            },
            self.CARDINALITY_020:     {
                "negative_surface_lambda": 0.30,
                "cardinality_lambda":      0.10,
                "minimum_positive_area":   0.020,
            },
            self.TV_001:              {
                "negative_surface_lambda": 0.30,
                "total_variation_lambda":  0.01,
            },
            self.TV_005:              {
                "negative_surface_lambda": 0.30,
                "total_variation_lambda":  0.05,
            },
            self.DIRICHLET_0001:      {
                "negative_surface_lambda": 0.30,
                "dirichlet_lambda":        0.001,
            },
            self.DIRICHLET_001:       {
                "negative_surface_lambda": 0.30,
                "dirichlet_lambda":        0.01,
            },
        }
        return baseline | additions[self]
