"""Validated preprocessing configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Define the scientific settings accepted from LambdaForge YAML.

    Attributes:
        model_index: Zero-based coordinate model selected by Gemmi.
        chains: Global chain filter; a remote record selector takes precedence.
        include_hydrogens: Whether explicit source hydrogen atoms are retained.
        include_waters: Whether crystallographic water residues are retained.
        include_nonpolymer: Whether non-polymer residues such as ligands are retained.
        include_metals: Whether metal atoms/residues are retained.
        center_coordinates: Whether to subtract the filtered atomic centroid.
        atom_spatial_radius: Spatial atomic-neighbor cutoff in ångströms.
        atom_spatial_k_max: Maximum ranked spatial neighbors retained per atom.
        surface_resolution: Main surface sampling length scale in ångströms.
        probe_radius: Solvent probe radius added to van der Waals radii, in ångströms.
        surface_atom_radius: Surface-to-atom neighborhood cutoff in ångströms.
        surface_atom_k_max: Maximum nearest atoms retained per surface point.
        diffusion_spectral_modes_max: Maximum Laplace--Beltrami eigenmodes persisted.
        surface_neighbor_k_max: Maximum local surface neighbors persisted for V3 encoders.
        curvature_scales: Positive neighborhood-radius multipliers relative to
            ``surface_resolution``. One ``(H,K,C)`` curvature triplet is stored per multiplier.
    """

    # Structural filters determine which normalized domain objects exist.
    model_index       : int                          = 0
    chains            : tuple[str, ...] | list[str] = ()
    include_hydrogens : bool                         = False
    include_waters    : bool                         = False
    include_nonpolymer: bool                         = False
    include_metals    : bool                         = False
    center_coordinates: bool                         = True

    # Scientific radii are expressed in ångströms.
    atom_spatial_radius          : float                          = 6.0
    atom_spatial_k_max           : int                            = 32
    surface_resolution           : float                          = 1.0
    probe_radius                 : float                          = 1.4
    surface_atom_radius          : float                          = 6.0
    surface_atom_k_max           : int                            = 32
    diffusion_spectral_modes_max : int                            = 128
    surface_neighbor_k_max       : int                            = 24
    curvature_scales             : tuple[float, ...] | list[float] = (2.5, 5.0)

    def __post_init__(self) -> None:
        """Canonicalize chain values and reject invalid configuration domains.

        Frozen dataclass fields are normalized with ``object.__setattr__``. ``model_index`` must be
        non-negative and every physical radius must be strictly positive so KD-tree and surface
        operations have meaningful domains.

        Raises:
            ValueError: If the model index, a radius, or the curvature-scale collection violates
                its allowed domain.
        """
        # YAML may construct a list; internal code uses one immutable tuple representation.
        object.__setattr__(self, "chains", tuple(str(value) for value in self.chains))
        object.__setattr__(
            self,
            "curvature_scales",
            tuple(float(value) for value in self.curvature_scales),
        )

        if self.model_index < 0:
            raise ValueError("model_index cannot be negative")

        # All geometric cutoffs are strict positive length scales.
        for name in (
            "atom_spatial_radius",
            "surface_resolution",
            "probe_radius",
            "surface_atom_radius",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

        for name in (
            "atom_spatial_k_max",
            "surface_atom_k_max",
            "diffusion_spectral_modes_max",
            "surface_neighbor_k_max",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.curvature_scales:
            raise ValueError("curvature_scales must contain at least one scale")
        if any(value <= 0 for value in self.curvature_scales):
            raise ValueError("curvature_scales must contain only positive values")
        if len(set(self.curvature_scales)) != len(self.curvature_scales):
            raise ValueError("curvature_scales cannot contain duplicates")

    def scientific_dict(self) -> dict[str, Any]:
        """Build the canonical configuration payload that identifies scientific output.

        Returns:
            A JSON-compatible dictionary containing model selection, structural filters,
            coordinate policy, and graph/surface length scales. Operational settings are absent
            from this class because LambdaForge owns execution policy.
        """
        values = asdict(self)

        # JSON has no tuple type; a list provides a stable portable representation.
        values["chains"] = list(self.chains)
        values["curvature_scales"] = list(self.curvature_scales)
        return values
