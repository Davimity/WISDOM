"""Opt-in grouped Hard-Concrete pressure without changing historical model defaults."""

import math
import torch

from torch import Tensor
from typing import Any, cast
from collections.abc import Mapping
from wisdom.models.gating.GateGroup import GateGroup
from wisdom.models.gating.HardConcreteGate import HardConcreteGate
from wisdom.models.gating.SemanticGateRegistry import SemanticGateRegistry


class GatePolicy:
    """Own V10f coefficients, initial probabilities and localized nominal-epoch schedule."""

    CHEMICAL = frozenset("atom." + name for name in (
        "atom_name", "residue_atom", "backbone_torsions", "sidechain_torsions",
        "covalent_statistics", "functional_groups", "atomic_sasa",
    ))
    SURFACE_DESCRIPTORS = ("mean", "gaussian", "curvedness", "shape_index")
    DENOMINATOR = 41

    def __init__(
        self,
        lambda_surface        : float = 0.04,
        lambda_chemical       : float = 0.04,
        lambda_other          : float = 0.04,
        initial_active_surface: float = 0.95,
        initial_active_chemical: float = 0.95,
        initial_active_other  : float = 0.95,
        warmup_fraction       : float = 0.0,
        ramp_fraction         : float = 0.0,
    ) -> None:
        """Declare a complete policy, independent of data and random-number consumption.

        Args:
            lambda_surface: Nonnegative final coefficient for pure surface curvature gates.
            lambda_chemical: Nonnegative final coefficient for the seven V10 chemical sources.
            lambda_other: Nonnegative final coefficient for every other historical path.
            initial_active_surface: Initial analytic P(z>0), strictly inside (0,1), for S.
            initial_active_chemical: Initial analytic P(z>0), strictly inside (0,1), for C.
            initial_active_other: Initial analytic P(z>0), strictly inside (0,1), for O.
            warmup_fraction: Fraction of nominal epochs forcing S/C open at zero direct cost.
            ramp_fraction: Following nominal fraction linearly increasing S/C costs; O is fixed.

        Raises:
            ValueError: Nonfinite/negative costs, invalid probabilities or overlapping horizons.
        """
        self.coefficients = dict(zip(
            GateGroup, (lambda_surface, lambda_chemical, lambda_other), strict=True))
        self.initial = dict(zip(GateGroup, (
            initial_active_surface, initial_active_chemical, initial_active_other), strict=True))
        self.warmup_fraction = warmup_fraction
        self.ramp_fraction   = ramp_fraction
        if any(not math.isfinite(x) or x < 0 for x in self.coefficients.values()):
            raise ValueError("grouped gate coefficients must be finite and nonnegative")
        if any(not 0 < x < 1 for x in self.initial.values()):
            raise ValueError("grouped gate initial probabilities must lie inside (0,1)")
        if not (0 <= warmup_fraction <= 1 and 0 <= ramp_fraction <= 1
                and warmup_fraction + ramp_fraction <= 1):
            raise ValueError("grouped warm-up/ramp fractions must fit the nominal horizon")
        self.effective = dict(self.coefficients)

    @classmethod
    def group(cls, name: str) -> GateGroup:
        """Classify one direct cost by meaning, not by the broad 'surface' module prefix.

        Args:
            name: Stable semantic gate identifier, including scale-qualified curvature gates.

        Returns:
            Exactly one S/C/O group; surface.atom_context belongs to O, never S.
        """
        if name in cls.CHEMICAL:
            return GateGroup.CHEMICAL
        if any(name.startswith(f"surface.curvature.{kind}.scale_")
               for kind in cls.SURFACE_DESCRIPTORS):
            return GateGroup.SURFACE
        return GateGroup.OTHER

    def initialize(self, registry: SemanticGateRegistry, baseline_count: int) -> None:
        """Set individual log_alpha deterministically without rebuilding any shared module.

        Args:
            registry: Fully assembled inventory, including fixed/absent chemical blocks.
            baseline_count: Historical inventory size; V10f requires five scales and 41 gates.

        Raises:
            ValueError: A different baseline would change the experiment's per-source pressure.
        """
        if baseline_count != self.DENOMINATOR:
            raise ValueError("grouped gate policy requires the historical 41-gate inventory")
        offset = HardConcreteGate.BETA * math.log(-HardConcreteGate.GAMMA / HardConcreteGate.ZETA)
        with torch.no_grad():
            for name, gate in zip(registry.names, registry.gates, strict=True):
                probability = self.initial[self.group(name)]
                cast(HardConcreteGate, gate).log_alpha.fill_(
                    math.log(probability / (1 - probability)) + offset)

    def schedule(self, registry: SemanticGateRegistry, epoch: int, epochs: int) -> str:
        """Apply localized all-on/ramp using t=(epoch-1)/nominal_epochs, not observed duration.

        Args:
            registry: Runtime registry whose localized override is saved with each checkpoint.
            epoch: One-based current epoch.
            epochs: Positive nominal horizon; patience never rescales this schedule.

        Returns:
            all_on, ramp or learned. Only S/C are forced during all_on; O remains learned.
        """
        fraction = (epoch - 1) / epochs
        warmup = fraction < self.warmup_fraction
        progress = fraction - self.warmup_fraction
        # Compare the endpoint before subtracting fractions: 0.03-0.01 can round below 0.02.

        finished = fraction >= self.warmup_fraction + self.ramp_fraction
        scale = (1.0 if finished else 0.0 if warmup else progress / self.ramp_fraction)
        registry.schedule_open = frozenset(
            name for name in registry.names if warmup and self.group(name) is not GateGroup.OTHER)
        self.effective = {group: value * (1.0 if group is GateGroup.OTHER else scale)
                          for group, value in self.coefficients.items()}
        return "all_on" if warmup else "ramp" if scale < 1.0 else "learned"

    def contributions(self, registry: SemanticGateRegistry) -> dict[GateGroup, Tensor]:
        """Compute weighted expected path costs with one shared denominator and no duplicates.

        Args:
            registry: Current learned probabilities with ancestor products. Fixed-design and
                structurally absent gates have zero direct cost; ancestors retain their own cost.

        Returns:
            Differentiable lambda_g * sum_{i in g} P(path_i active) / 41 per group.
        """
        zero = cast(HardConcreteGate, registry.gates[0]).log_alpha.new_zeros(())
        result = {group: zero for group in GateGroup}
        for name in registry.names:
            if name in registry.inactive_names or name in registry.fixed_values:
                continue
            group = self.group(name)
            result[group] = result[group] + registry.effective_probability(name)
        return {group: value * (self.effective[group] / self.DENOMINATOR)
                for group, value in result.items()}

    def penalty(self, registry: SemanticGateRegistry) -> Tensor:
        """Return the already-weighted loss term; never multiply it by gate_lambda again.

        Args:
            registry: Current semantic inventory and hierarchy.

        Returns:
            Total direct group contributions. Uniform learned coefficients use the exact
            historical mean * N/41 * lambda arithmetic, preserving value and gradients.
        """
        values = tuple(self.effective.values())
        if len(set(values)) == 1 and not registry.fixed_values:
            return registry.regularization() * (len(registry.names) / self.DENOMINATOR) * values[0]
        return torch.stack(tuple(self.contributions(registry).values())).sum()

    def manifest(
        self, registry: SemanticGateRegistry, dimensions: Mapping[str, int],
    ) -> dict[str, Any]:
        """Describe initial scientific controls and current path costs without reading targets.

        Args:
            registry: Full ordered inventory; dimensions must describe every available gate.
            dimensions: Source feature width, or latent branch width for a pathway gate.

        Returns:
            JSON-compatible formula, nominal policy, direct per-gate cost and expected budgets.
        """
        records = cast(list[dict[str, Any]], registry.summary()["gates"])
        for record in records:
            name = record["name"]
            group = self.group(name)
            record.update(group=group.value, dimensions=dimensions[name],
                          unit_cost=0 if name in registry.fixed_values
                          or name in registry.inactive_names else 1,
                          initial_probability=self.initial[group],
                          initialization_policy="analytic_activity_probability")
        return {
            "formula": "sum_g lambda_g * sum_{i in g} p_effective_i / 41; fixed costs excluded",
            "denominator": self.DENOMINATOR,
            "coefficients": self.coefficients,
            "initial_probabilities": self.initial,
            "warmup_fraction": self.warmup_fraction,
            "ramp_fraction": self.ramp_fraction,
            "gates": records,
            "structurally_absent_chemical_sources": sorted(self.CHEMICAL - set(registry.names)),
            "expected_penalty": {group.value: float(value.detach().cpu())
                                 for group, value in self.contributions(registry).items()},
        }
