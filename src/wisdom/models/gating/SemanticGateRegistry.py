"""Stable registry for globally shared semantic gates and their hierarchy."""

from collections.abc import Mapping, Sequence
from typing import cast

import torch

from torch import Tensor, nn

from wisdom.models.gating.HardConcreteGate import HardConcreteGate


class SemanticGateRegistry(nn.Module):
    """Own named gates, global overrides, normalized L0 cost, and reporting metadata."""

    def __init__(
        self,
        definitions: Sequence[tuple[str, str | None]],
    ) -> None:
        """Create gates in a stable declared order.

        Args:
            definitions: Ordered ``(name, parent)`` pairs. A parent must precede its child.

        Raises:
            ValueError: If names repeat or a parent is missing.
        """
        super().__init__()

        names = [name for name, _ in definitions]
        if len(names) != len(set(names)):
            raise ValueError("semantic gate names must be unique")

        known: set[str] = set()
        for name, parent in definitions:
            if parent is not None and parent not in known:
                raise ValueError(f"gate parent {parent!r} must precede child {name!r}")
            known.add(name)

        self.gates       = nn.ModuleList(HardConcreteGate() for _ in definitions)
        self.names       = tuple(names)
        self.parents     = {name: parent for name, parent in definitions}
        self._indices    = {name: index for index, name in enumerate(self.names)}
        self._override   = "learned"
        self._forced     : dict[str, float] = {}

    def sample(self, training: bool) -> dict[str, Tensor]:
        """Sample each semantic gate exactly once for one model forward.

        Args:
            training: Whether learned gates use stochastic reparameterized samples.

        Returns:
            Name-to-scalar mapping shared by all layers, atoms, points, and proteins in the batch.
        """
        values: dict[str, Tensor] = {}
        for name, raw_gate in zip(self.names, self.gates, strict=True):
            gate = cast(HardConcreteGate, raw_gate)
            if name in self._forced:
                values[name] = gate.log_alpha.new_tensor(self._forced[name], dtype=torch.float32)
            elif self._override == "all_on":
                values[name] = gate.log_alpha.new_ones((), dtype=torch.float32)
            else:
                values[name] = gate(training=training)
        return values

    def set_override(self, mode: str = "learned") -> None:
        """Select learned values or force every gate on for post-hoc evaluation.

        Args:
            mode: ``learned`` or ``all_on``.

        Raises:
            ValueError: If ``mode`` is unsupported.
        """
        if mode not in {"learned", "all_on"}:
            raise ValueError("gate override must be learned or all_on")
        self._override = mode

    def force(self, name: str, value: int | float | None) -> None:
        """Force one named gate to zero/one, or clear its post-hoc override.

        Args:
            name: Stable semantic gate name.
            value: ``0`` or ``1``; ``None`` restores the registry-level mode.

        Raises:
            KeyError: If the gate does not exist.
            ValueError: If a non-binary value is supplied.
        """
        if name not in self._indices:
            raise KeyError(name)
        if value is None:
            self._forced.pop(name, None)
            return
        if value not in {0, 1}:
            raise ValueError("a forced semantic gate must be zero or one")
        self._forced[name] = float(value)

    def probability_active(self, name: str) -> Tensor:
        """Return one gate's analytic nonzero probability.

        Args:
            name: Stable semantic gate name.

        Returns:
            Scalar float32 probability.
        """
        gate = cast(HardConcreteGate, self.gates[self._indices[name]])
        return gate.probability_active()

    def deterministic_value(self, name: str) -> Tensor:
        """Return one gate's deterministic stretched-sigmoid value.

        Args:
            name: Stable semantic gate name.

        Returns:
            Scalar float32 value in ``[0,1]``.
        """
        if name in self._forced:
            gate = cast(HardConcreteGate, self.gates[0])
            return gate.log_alpha.new_tensor(self._forced[name], dtype=torch.float32)
        if self._override == "all_on":
            gate = cast(HardConcreteGate, self.gates[0])
            return gate.log_alpha.new_ones((), dtype=torch.float32)
        gate = cast(HardConcreteGate, self.gates[self._indices[name]])
        return gate.deterministic_value()

    def effective_probability(self, name: str) -> Tensor:
        """Multiply a gate probability by every ancestor probability in its semantic path.

        Args:
            name: Stable semantic gate name.

        Returns:
            Scalar expected probability that the entire information path is active.
        """
        probability = self.probability_active(name)
        parent      = self.parents[name]
        while parent is not None:
            probability = probability * self.probability_active(parent)
            parent      = self.parents[parent]
        return probability

    def regularization(self) -> Tensor:
        """Return the unit-cost normalized expected L0 penalty in ``[0,1]``."""
        probabilities = [self.effective_probability(name) for name in self.names]
        return torch.stack(probabilities).mean()

    def summary(self) -> dict[str, object]:
        """Build JSON-compatible per-gate and family diagnostics.

        Returns:
            Mapping with fixed Hard-Concrete constants, ordered gate records, and family means.
        """
        records: list[dict[str, object]] = []
        families: dict[str, list[float]] = {}
        for name, raw_gate in zip(self.names, self.gates, strict=True):
            gate = cast(HardConcreteGate, raw_gate)
            probability = float(gate.probability_active().detach().cpu())
            deterministic = float(gate.deterministic_value().detach().cpu())
            effective = float(self.effective_probability(name).detach().cpu())
            family = (
                "surface_curvature"
                if name.startswith("surface.curvature")
                else name.split(".")[0]
            )
            families.setdefault(family, []).append(probability)
            records.append(
                {
                    "name":                       name,
                    "parent":                     self.parents[name],
                    "log_alpha":                  float(gate.log_alpha.detach().float().cpu()),
                    "probability_active":          probability,
                    "deterministic_value":         deterministic,
                    "effective_path_probability": effective,
                }
            )

        return {
            "hard_concrete": {
                "beta":  HardConcreteGate.BETA,
                "gamma": HardConcreteGate.GAMMA,
                "zeta":  HardConcreteGate.ZETA,
            },
            "gates": records,
            "families": {
                family: sum(values) / len(values)
                for family, values in families.items()
            },
        }

    def gate_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return learned ``log_alpha`` parameters for the no-decay optimizer group."""
        return tuple(cast(HardConcreteGate, gate).log_alpha for gate in self.gates)

    def definition(self) -> tuple[Mapping[str, str | None], ...]:
        """Return the stable ordered name/parent contract stored in checkpoints."""
        return tuple({"name": name, "parent": self.parents[name]} for name in self.names)
