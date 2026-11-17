"""Explicit pilot conditions for VF's electrostatic and neutral-probe reference fields."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PhysicsConfig:
    """Keep physical choices fingerprinted independently of execution and immutable base geometry.

    Conditions and repair thresholds are a declared pilot, not universal physical truth.
    Probe names identify actual
    AMBER14 residue/atom types; sigma and epsilon must be extracted from that exact XML, never
    inferred from WISDOM radii. Numerical tolerances are admission criteria, not accuracy claims.
    """

    ph: float = 7.0
    temperature: float = 298.15
    salt: float = 0.15
    protein_dielectric: float = 2.0
    solvent_dielectric: float = 78.5
    solvent_radius: float = 1.4
    nonlinear: bool = False
    forcefield: str = "amber14/protein.ff14SB.xml"
    probe_small: str = "ALA:HB1"
    probe_large: str = "ALA:CB"
    spacing: float = 1.0
    margin: float = 12.0
    coarse_factor: float = 1.7
    boundary_condition: str = "sdh"
    connectivity: int = 6
    bank_size: int = 16384
    seed: int = 2026
    boundary_width: float = 2.0
    exterior_width: float = 6.0
    overlap_fraction: float = 0.65
    repulsion_ceiling: float = 10000.0
    decoder_radius: float = 12.0
    lj_cutoff: float = 0.0
    lj_absolute_tolerance: float = 1e-5
    lj_relative_tolerance: float = 1e-5
    audit_queries: int = 24
    convergence: bool = True
    margin_convergence: bool = True
    margin_factor: float = 1.5
    refinement_factor: float = 0.75
    potential_absolute_tolerance: float = 0.25
    potential_relative_tolerance: float = 0.1
    potential_failure_policy: str = "mask"
    minimum_potential_coverage: float = 0.95
    minimum_region_coverage: float = 0.8
    heavy_atom_policy: str = "strict"
    repair_maximum_atoms: int = 32
    repair_maximum_fraction: float = 0.05
    repair_maximum_per_residue: int = 5
    repair_maximum_residue_fraction: float = 0.1
    repair_minimum_nonbonded_fraction: float = 0.65
    repair_exclusion_radius: float = 3.0

    def parameters(self) -> dict[str, object]:
        """Return JSON-shaped scientific choices for provenance and native checkpoint keys.

        Returns:
            Effective physical/numerical choices, without operational allocation limits. Strict
            policy omits inactive repair options to preserve historical strict checkpoint keys.
        """
        values = asdict(self)
        if self.heavy_atom_policy == "strict":
            values = {key: value for key, value in values.items()
                      if key != "heavy_atom_policy" and not key.startswith("repair_")}
        return values
