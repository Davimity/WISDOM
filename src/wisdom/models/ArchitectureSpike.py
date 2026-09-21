"""Named isolated architectural spikes from the pre-freeze WISDOM plan."""

from enum import Enum


class ArchitectureSpike(str, Enum):
    """Name single-factor pre-V1 architectural candidates from the master plan."""

    CUSTOM         = "custom"
    BASELINE       = "baseline"
    POINT_GEOMETRY = "point_geometry"
    DEPTH_CONTROL  = "depth_control"
    BIDIRECTIONAL  = "bidirectional"
    VECTOR_8       = "vector_8"
    VECTOR_16      = "vector_16"

    def overrides(self) -> dict[str, str | int | bool]:
        """Return the isolated model change represented by this spike.

        Returns:
            Model overrides. Non-custom profiles restore the scalar single-pass baseline first,
            then activate exactly one tested hypothesis.
        """
        if self is self.CUSTOM:
            return {}

        baseline: dict[str, str | int | bool] = {
            "surface_geometry_transfer": False,
            "interaction_round":          "single",
            "vector_atomic_channels":     0,
        }
        additions: dict[ArchitectureSpike, dict[str, str | int | bool]] = {
            self.BASELINE:       {},
            self.POINT_GEOMETRY: {"surface_geometry_transfer": True},
            self.DEPTH_CONTROL:  {"interaction_round": "depth_control"},
            self.BIDIRECTIONAL:  {"interaction_round": "bidirectional"},
            self.VECTOR_8:       {"vector_atomic_channels": 8},
            self.VECTOR_16:      {"vector_atomic_channels": 16},
        }
        return baseline | additions[self]
