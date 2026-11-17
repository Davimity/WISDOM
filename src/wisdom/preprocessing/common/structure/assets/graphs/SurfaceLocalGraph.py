"""Reuse the geometric coherence graph instead of introducing another surface-neighbour rule."""

import numpy as np

from pathlib import Path
from collections.abc import Mapping
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.geometry.SurfaceBuilder import SurfaceBuilder


class SurfaceLocalGraph:
    """Publish the existing filtered graph; it approximates locality, not exact geodesics."""

    FAMILY = "surface_local_graph"

    def __init__(self, resolution: float = 1.0) -> None:
        """Set the existing radius/normal/chord rule's physical scale.

        Args:
            resolution: Positive h in Å; use the surface's original spacing unless comparing scales.
        """
        self.parameters = {"resolution": resolution}
        self.builder = SurfaceBuilder(resolution)

    def build(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return sorted undirected edges and actual connected components without repairs.

        Args:
            arrays: Oriented surface positions/normals [M,3], in one coordinate system.

        Returns:
            Existing distance [E], edge_index [2,E], component_ids [M], plus degree [M].
            Relative vectors/normal offsets can be derived cheaply; they are not duplicated here.
        """
        result, _ = self.builder.build_graph(arrays["surface_positions"], arrays["surface_normals"])
        result["degree"] = np.bincount(result["surface_edge_index"].ravel(),
            minlength=len(arrays["surface_positions"])).astype(np.int32)
        return result

    def write(self, base: Path, output: Path) -> None:
        """Store reusable filtered connectivity and explicit isolation diagnostics.

        Args:
            base: Immutable universal geometry.
            output: LF-owned temporary output.
        """
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in ("surface_positions", "surface_normals")}
        values = self.build(arrays)
        StructuralAsset.write(output, base, values, self.FAMILY, self.parameters,
            {"algorithm": "existing-radius-normal-chord-filter-v1", "units": "angstrom",
             "isolated": int(np.count_nonzero(values["degree"] == 0)),
             "components": int(values["surface_component_ids"].max()) + 1})

    def validate(self, path: Path, base: Path) -> bool:
        """Reconstruct the exact filtered topology before immutable publication.

        Args:
            path: Optional local graph asset.
            base: Structural source.

        Returns:
            True after all array/configuration checks pass.

        Raises:
            ValueError: Wrong parameters or graph/diagnostic arrays.
        """
        actual, header = StructuralAsset.read(path, base, self.FAMILY)
        if header["parameters"] != self.parameters or header.get("units") != "angstrom" or \
                header.get("algorithm") != "existing-radius-normal-chord-filter-v1":
            raise ValueError("surface local graph configuration mismatch")
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in ("surface_positions", "surface_normals")}
        StructuralAsset.compare(actual, self.build(arrays))
        return True
