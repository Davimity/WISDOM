"""Independent deterministic surface discretizations bound to unchanged physical atoms."""

import json
import math
import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from scipy.spatial.transform import Rotation
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.geometry.DiffusionOperatorBuilder import (
    DiffusionOperatorBuilder,
)
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.geometry.SurfaceBuilder import SurfaceBuilder
from wisdom.preprocessing.common.structure.geometry.SurfaceAtomNeighborhoodBuilder import (
    SurfaceAtomNeighborhoodBuilder,
)


class SurfaceView:
    """Regenerate all point-dependent fields together, never duplicate the physical atom graph."""

    FAMILY = "surface_view"

    def __init__(self, view_id: int = 1, resolution: float | None = None,
                 kind: str | None = None, seed: int = 2026) -> None:
        """Bind an explicit view identifier and optional spacing override.

        Args:
            view_id: Nonnegative deterministic ID; zero uses exactly the historical phase/grid.
            resolution: Optional positive surface spacing in Å; None preserves the base setting.
            kind: None retains the historical combined view. phase/grid/resolution/rigid isolate
                the declared V12 numerical factor; identity reproduces the unperturbed sampler.
            seed: Stable physical-view seed, default 2026; independent of training randomness.
        """
        if view_id < 0 or (resolution is not None and resolution <= 0):
            raise ValueError("view ID must be nonnegative and spacing must be positive")
        self.parameters: dict[str, Any] = {"view_id": view_id, "resolution": resolution}
        if kind is not None:
            if kind not in {"identity", "phase", "grid", "resolution", "rigid"}:
                raise ValueError("unknown isolated surface-view kind")
            self.parameters.update(kind=kind, seed=seed)

    def build(self, base: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Construct one coherent view using fixed atoms/radii and base physical probe.

        Args:
            base: Validated universal NPZ whose atom arrays are read, never rewritten.

        Returns:
            New surface geometry, atom candidates and all intrinsic operators, plus timings.
            No task labels are consumed or propagated by index onto a different cloud.
        """
        with np.load(base, allow_pickle=False) as stored:
            meta = json.loads(str(stored["metadata_json"].item()))
            atoms = stored["atom_positions"]
            radii = stored["vdw_radii"]
            identities = {key: stored[key] for key in (
                "chain_indices", "residue_indices", "atom_names")}
        config = PreprocessConfig(**meta["config"])
        resolution = self.parameters["resolution"] or config.surface_resolution
        builder = self.sampler(config, identities)
        kind = self.parameters.get("kind")
        rotation = Rotation.random(random_state=int(self.parameters.get("seed", 2026))).as_matrix()
        translation = np.asarray([2., -3., 1.])
        if kind == "rigid":
            atoms = atoms.astype(float) @ rotation.T + translation
        arrays, warnings = builder.build(atoms, radii)
        arrays.update(SurfaceAtomNeighborhoodBuilder(config.surface_atom_radius,
            config.surface_atom_k_max).build(atoms, arrays["surface_positions"],
                                            arrays["surface_normals"]))
        operator = DiffusionOperatorBuilder(resolution, config.diffusion_spectral_modes_max,
                                            config.surface_neighbor_k_max)
        arrays.update(operator.build(arrays["surface_positions"], arrays["surface_normals"]))
        if kind == "rigid":
            arrays = self.rigid_transform(arrays, rotation.T, -rotation.T @ translation)
        return arrays, {"sampling_seconds": builder.timings, "warnings": warnings,
                        "point_count": len(arrays["surface_positions"])}

    def write(self, base: Path, output: Path) -> None:
        """Persist only the new surface domain and its matching operators.

        Args:
            base: Immutable physical geometry archive.
            output: Native LF-managed temporary path.
        """
        arrays, details = self.build(base)
        StructuralAsset.write(output, base, arrays, self.FAMILY, self.parameters,
            {"algorithm": ("isolated-physical-view-v2" if "kind" in self.parameters else
                           "fibonacci-phase-and-voxel-offset-v1"), "units": "angstrom",
             "physical_atoms_unchanged": True, "warnings": details["warnings"],
             "point_count": details["point_count"]})

    def validate(self, path: Path, base: Path) -> bool:
        """Validate joined physical geometry and all view operators, not ZIP presence alone.

        Args:
            path: Optional surface-view asset.
            base: Immutable physical atoms and original task-independent settings.

        Returns:
            True if the declared sampler reproduces the stored points and the complete joined
            schema passes scientific archive validation. Unrounded reproduced points resolve
            active-sphere cutoff decisions for normals without relaxing stored geometry checks.

        Raises:
            ValueError: Wrong binding/configuration or incompatible geometry/operators.
        """
        arrays, header = StructuralAsset.read(path, base, self.FAMILY)
        algorithm = ("isolated-physical-view-v2" if "kind" in self.parameters else
                     "fibonacci-phase-and-voxel-offset-v1")
        if header["parameters"] != self.parameters or header.get("algorithm") != algorithm or (
            header.get("physical_atoms_unchanged") is not True
        ):
            raise ValueError("surface view configuration mismatch")
        with np.load(base, allow_pickle=False) as stored:
            meta = json.loads(str(stored["metadata_json"].item()))
            joined = {key: stored[key] for key in stored.files if key != "metadata_json"}
        config = dict(meta["config"])
        physical = joined["atom_positions"]
        if self.parameters.get("kind") == "rigid":
            rotation = Rotation.random(random_state=int(self.parameters["seed"])).as_matrix()
            translation = np.asarray([2., -3., 1.])
            physical = physical.astype(float) @ rotation.T + translation
        sampled, _ = self.sampler(PreprocessConfig(**config), joined).sample(
            physical, joined["vdw_radii"])
        reference = sampled
        if self.parameters.get("kind") == "rigid":
            reference = (sampled - translation) @ rotation
            sampled = (sampled.astype(np.float32) - translation) @ rotation
        matches = (np.allclose(sampled.astype(np.float32), arrays["surface_positions"],
                              rtol=0, atol=2e-5) if self.parameters.get("kind") == "rigid" else
                   np.array_equal(sampled.astype(np.float32), arrays["surface_positions"]))
        if not matches:
            raise ValueError("surface view points do not reproduce the declared sampling policy")
        config["surface_resolution"] = self.parameters["resolution"] or config["surface_resolution"]
        surface_keys = {key for key in joined if key.startswith(("surface_", "diffusion_"))}
        if set(arrays) != surface_keys:
            raise ValueError("surface view must replace all point-dependent arrays together")
        joined.update(arrays)

        # Sampling has already reproduced the exact view. Its unrounded points disambiguate
        # sphere membership at the normal blend's hard cutoff: float32 rounding can toggle an
        # active sphere even though coordinates agree to less than a millionth of an Å.

        ProteinArchive(PreprocessConfig(**config)).validate(joined, surface_reference=reference)
        return True

    def sampler(self, config: PreprocessConfig,
                identities: dict[str, np.ndarray] | None = None) -> SurfaceBuilder:
        """Resolve the same explicit view policy for construction and scientific validation.

        Args:
            config: Original atom/radius/probe and surface-geometry settings.
            identities: Chain-copy, physical residue and atom-name arrays in exact parent order;
                required by the per-atom stable-hash phase protocol, never inferred from labels.

        Returns:
            Sampler using historical phase/grid for view zero or fixed ID-derived offsets.
            No global random state, task target or model parameter participates.
        """
        view = int(self.parameters["view_id"] or 0)
        resolution = self.parameters["resolution"] or config.surface_resolution
        phase = (view * 0.6180339887498949 % 1) * 2 * math.pi if view else 0.0
        offset = ((view * 0.381966) % 1, (view * 0.414214) % 1, (view * 0.732051) % 1)
        atom_phases = None
        kind = self.parameters.get("kind")
        if kind is not None:
            phase = 0.0
            offset = (0., 0., 0.)
            seed = int(self.parameters["seed"])
            if kind == "phase":
                if identities is None:
                    raise ValueError("per-atom phase needs canonical physical identities")
                keys = zip(identities["chain_indices"], identities["residue_indices"],
                           identities["atom_names"], strict=True)
                atom_phases = np.asarray([int.from_bytes(hashlib.sha256(
                    f"{seed}/{view}/{int(c)}/{int(r)}/{a}".encode()).digest()[:8], "big") /
                    2**64 * 2 * math.pi for c, r, a in keys])
            if kind == "grid":
                shift = np.random.default_rng(seed + view).uniform(size=3)
                offset = (float(shift[0]), float(shift[1]), float(shift[2]))
        return SurfaceBuilder(resolution, config.probe_radius, config.curvature_scales,
                              phase, offset, atom_phases)

    @staticmethod
    def rigid_transform(
        arrays: dict[str, np.ndarray], rotation: np.ndarray, translation: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Audit a rigidly transformed tensor representation without resampling or labels.

        Args:
            arrays: Base or view arrays; positions [*,3] and outward normals [M,3].
            rotation: Proper orthonormal rotation [3,3], acting on column vectors.
            translation: Finite displacement [3] in Å.

        Returns:
            Mapping with transformed positions/normals and tangent gradient components rotated
            into the normal-derived frame used by WISDOM; intrinsic scalar operators stay fixed.
            This is not regeneration through the globally oriented preprocessing sampler.

        Raises:
            ValueError: Non-rigid transform or malformed translation.
        """
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
            np.linalg.det(rotation), 1
        ) or translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("surface audit requires a finite proper rigid transform")
        result = dict(arrays)
        for key in ("atom_positions", "surface_positions"):
            if key in result:
                result[key] = (result[key] @ rotation.T + translation).astype(result[key].dtype)
        if "surface_normals" in result:
            result["surface_normals"] = (result["surface_normals"] @ rotation.T).astype(
                result["surface_normals"].dtype)
            if "diffusion_gradient_index" in result:
                old = np.asarray([SurfaceBuilder._tangent_basis(normal)
                                  for normal in arrays["surface_normals"]])
                new = np.asarray([SurfaceBuilder._tangent_basis(normal)
                                  for normal in result["surface_normals"]])
                transform = np.einsum("mij,mkj->mik", new, old @ rotation.T)
                rows = result["diffusion_gradient_index"][0]
                gradient = np.stack((arrays["diffusion_gradient_x"],
                                     arrays["diffusion_gradient_y"]), axis=1)
                changed = np.einsum("mij,mj->mi", transform[rows], gradient)
                result["diffusion_gradient_x"] = changed[:, 0].astype(gradient.dtype)
                result["diffusion_gradient_y"] = changed[:, 1].astype(gradient.dtype)
        return result
