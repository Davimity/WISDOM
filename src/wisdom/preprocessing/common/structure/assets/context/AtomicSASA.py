"""Optional deterministic Shrake--Rupley solvent exposure, independent of surface quadrature."""

import numpy as np

from pathlib import Path
from scipy.spatial import cKDTree
from collections.abc import Mapping
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.geometry.SurfaceBuilder import SurfaceBuilder


class AtomicSASA:
    """Estimate exposed expanded-sphere area per retained atom, in square ångströms."""

    FAMILY = "atomic_sasa"

    def __init__(self, probe_radius: float = 1.4, samples: int = 960) -> None:
        """Set deterministic spherical quadrature, not WISDOM surface sampling.

        Args:
            probe_radius: Positive solvent probe radius in Å.
            samples: Positive number of Fibonacci directions per atom; controls approximation.
        """
        if probe_radius <= 0 or samples < 24:
            raise ValueError("SASA requires a positive probe and at least 24 samples")
        self.parameters = {"probe_radius": probe_radius, "samples": samples}

    def build(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Count exposed directions using exact neighbouring expanded-sphere containment.

        Args:
            arrays: Atom centers [N,3] and positive van der Waals radii [N], in Å.

        Returns:
            sasa [N] float32 Å² and valid [N] boolean. Area is 4*pi*(r+probe)^2 times
            the exposed fraction, not the normalized surface_area_weights quadrature.
            Finite sampling has directional error; no analytic exactness is claimed.
        """
        atoms = arrays["atom_positions"].astype(np.float64)
        radii = arrays["vdw_radii"].astype(np.float64) + self.parameters["probe_radius"]
        count = int(self.parameters["samples"])
        directions = SurfaceBuilder(1).fibonacci_sphere(count)
        neighborhoods = cKDTree(atoms).query_ball_point(atoms, radii + radii.max())
        area = np.empty(len(atoms), np.float32)

        # A buried direction can never become exposed again under further sphere intersections.
        # Test only surviving sample IDs, with the exact former distance arithmetic/comparison;
        # this changes neither quadrature directions, boundary convention nor stored float32 area.

        for i, (atom, radius, candidates) in enumerate(zip(
            atoms, radii, neighborhoods, strict=True
        )):
            exposed = np.arange(count)
            points = atom + radius * directions
            for other in candidates:
                if other != i:
                    outside = np.sum((points[exposed] - atoms[other])**2, axis=1) >= radii[other]**2
                    exposed = exposed[outside]
                    if not len(exposed):
                        break
            area[i] = 4 * np.pi * radius**2 * (len(exposed) / count)
        return {"sasa": area, "valid": np.ones(len(atoms), np.bool_)}

    def write(self, base: Path, output: Path) -> None:
        """Write an LF-owned base-bound exposure asset.

        Args:
            base: Validated immutable universal NPZ.
            output: Native managed temporary output.
        """
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in ("atom_positions", "vdw_radii")}
        StructuralAsset.write(output, base, self.build(arrays), self.FAMILY, self.parameters,
                              {"algorithm": "shrake-rupley-fibonacci-v1", "units": "angstrom^2"})

    def validate(self, path: Path, base: Path) -> bool:
        """Verify exact quadrature reconstruction and immutable binding.

        Args:
            path: Optional SASA asset.
            base: Immutable source NPZ.

        Returns:
            True after numerical, parameter and binding validation.

        Raises:
            ValueError: Wrong method/configuration or inconsistent exposed areas.
        """
        actual, header = StructuralAsset.read(path, base, self.FAMILY)
        if header["parameters"] != self.parameters or header.get("units") != "angstrom^2" or \
                header.get("algorithm") != "shrake-rupley-fibonacci-v1":
            raise ValueError("SASA configuration mismatch")
        with np.load(base, allow_pickle=False) as stored:
            expected = self.build({key: stored[key] for key in ("atom_positions", "vdw_radii")})
        StructuralAsset.compare(actual, expected)
        return True
