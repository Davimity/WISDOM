"""Optional larger fixed spectral budgets without rewriting points, graphs or gradients."""

import json
import numpy as np

from pathlib import Path
from wisdom.preprocessing.common.structure.geometry.DiffusionOperatorBuilder import (
    DiffusionOperatorBuilder,
)
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset


class SpectralExtension:
    """Reuse a stored prefix or solve additional modes only when explicitly requested."""

    FAMILY = "spectral_extension"

    def __init__(self, modes: int = 256) -> None:
        """Choose a positive fixed operator budget, independent of the downstream network.

        Args:
            modes: Number of lowest modes; physical limit is point_count-1.
        """
        if modes < 1:
            raise ValueError("spectral extension modes must be positive")
        self.parameters = {"modes": modes}

    def write(self, base: Path, output: Path) -> None:
        """Store an exact prefix or newly solved modes on the same immutable cloud.

        Args:
            base: Existing surface/operator archive.
            output: Managed sidecar output, never the base path.
        """
        with np.load(base, allow_pickle=False) as stored:
            values = stored["diffusion_eigenvalues"]
            vectors = stored["diffusion_eigenvectors"]
            positions = stored["surface_positions"]
            metadata = json.loads(str(stored["metadata_json"].item()))
        count = min(self.parameters["modes"], len(positions) - 1)
        reused = len(values) >= count
        if reused:
            arrays = {"eigenvalues": values[:count], "eigenvectors": vectors[:, :count]}
        else:
            arrays = DiffusionOperatorBuilder(
                metadata["config"]["surface_resolution"], count).spectrum(positions)
        StructuralAsset.write(output, base, arrays, self.FAMILY, self.parameters,
            {"algorithm": "existing-robust-laplacian-eigsh-v1", "reused_prefix": reused,
             "units": "eigenvalues angstrom^-2; physical mass-normalized eigenvectors"})

    def validate(self, path: Path, base: Path) -> bool:
        """Check mass orthonormality and sparse eigen-equation residuals without another solve.

        Args:
            path: Optional spectral asset.
            base: Its exact physical point cloud and mass/operator convention.

        Returns:
            True after shape, ordering, orthogonality and residual checks.

        Raises:
            ValueError: Wrong configuration or incompatible eigenpairs.
        """
        arrays, header = StructuralAsset.read(path, base, self.FAMILY)
        with np.load(base, allow_pickle=False) as stored:
            points = stored["surface_positions"].astype(np.float64)
            metadata = json.loads(str(stored["metadata_json"].item()))
        if header["parameters"] != self.parameters:
            raise ValueError("spectral extension configuration mismatch")
        count = min(self.parameters["modes"], len(points) - 1)
        values, vectors = arrays["eigenvalues"], arrays["eigenvectors"]
        if values.shape != (count,) or vectors.shape != (len(points), count) or \
                np.any(np.diff(values) < 0) or np.any(values < -1e-5):
            raise ValueError("spectral extension support/order mismatch")
        builder = DiffusionOperatorBuilder(metadata["config"]["surface_resolution"], count)
        points -= points.mean(axis=0, keepdims=True)
        stiffness, mass = builder._laplacian(points)
        gram = vectors.T @ (mass[:, None] * vectors)
        residual = stiffness @ vectors - (mass[:, None] * vectors) * values
        scale = max(1., float(np.linalg.norm(stiffness @ vectors)))
        if not np.allclose(gram, np.eye(count), atol=2e-3, rtol=2e-3) or \
                np.linalg.norm(residual) / scale > 2e-3:
            raise ValueError("spectral extension eigen-equation/orthonormality mismatch")
        return True
