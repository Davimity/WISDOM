"""Low-overhead training diagnostics for heterogeneous WISDOM components."""

import math

from torch import Tensor, nn
from typing import Any, cast
from torch.utils.hooks import RemovableHandle


class OptimizationDiagnostics:
    """Aggregate gradients, activations, logits, and gate saturation by training epoch."""

    def __init__(self, model: nn.Module) -> None:
        """Attach one atomic-encoder hook and retain the component references to inspect.

        Args:
            model: WISDOM model exposing atomic, transfer, surface, local-head, and gate modules.

        Raises:
            ValueError: If the model does not expose the required stability boundaries.
        """
        names = (
            "atomic_encoder",
            "surface_atom_transfer",
            "surface_encoder",
            "local_head",
            "semantic_gates",
        )
        if any(not hasattr(model, name) for name in names):
            raise ValueError("optimization diagnostics require the complete WISDOM backbone")

        self.model = cast(Any, model)
        self.atomic_statistics: tuple[float, float] | None = None
        self.sums  : dict[str, float] = {}
        self.counts: dict[str, int]   = {}
        atomic_encoder = self.model.atomic_encoder
        self.handle: RemovableHandle = atomic_encoder.register_forward_hook(self._capture_atomic)

    def begin_epoch(self) -> None:
        """Clear the previous epoch without removing the reusable forward hook."""
        self.sums.clear()
        self.counts.clear()

    def observe_forward(self, output: dict[str, Tensor]) -> None:
        """Accumulate detached atom/surface/logit and sampled-gate distribution summaries.

        Args:
            output: Current model output containing surface embeddings and local logits.

        Raises:
            ValueError: If expected diagnostic tensors are absent.
        """
        if "surface_embeddings" not in output or "surface_logits" not in output:
            raise ValueError("model output lacks optimization diagnostic tensors")
        surface = output["surface_embeddings"].detach().float()
        logits  = output["surface_logits"].detach().float()
        self._add("activation_surface_mean", float(surface.mean()))
        self._add("activation_surface_std", float(surface.std(unbiased=False)))
        self._add("surface_logit_mean", float(logits.mean()))
        self._add("surface_logit_std", float(logits.std(unbiased=False)))
        self._add("surface_logit_max", float(logits.max()))
        if self.atomic_statistics is not None:
            self._add("activation_atom_mean", self.atomic_statistics[0])
            self._add("activation_atom_std", self.atomic_statistics[1])

        semantic_gates = self.model.semantic_gates
        zero_fraction, one_fraction = semantic_gates.sampled_boundary_fractions()
        self._add("sampled_gate_zero_fraction", zero_fraction)
        self._add("sampled_gate_one_fraction", one_fraction)

    def observe_backward(self) -> None:
        """Accumulate global and component gradient L2 norms before optional clipping."""
        components = {
            "gradient_global_norm":   self.model,
            "gradient_atomic_norm":     self.model.atomic_encoder,
            "gradient_transfer_norm":   self.model.surface_atom_transfer,
            "gradient_surface_norm":    self.model.surface_encoder,
            "gradient_local_head_norm": self.model.local_head,
            "gradient_gate_norm":       self.model.semantic_gates,
        }
        for name, component in components.items():
            self._add(name, self._gradient_norm(component))

    def metrics(self) -> dict[str, float]:
        """Return arithmetic epoch means for every observed stability quantity."""
        return {
            name: value / self.counts[name]
            for name, value in self.sums.items()
            if self.counts[name]
        }

    def close(self) -> None:
        """Remove the atomic forward hook after training finishes."""
        self.handle.remove()

    def _capture_atomic(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor,
    ) -> None:
        """Retain detached atomic mean/std from the most recent model forward.

        Args:
            module: Atomic encoder that emitted ``output``.
            inputs: Positional encoder inputs; unused because output defines the statistic.
            output: Encoded atom states ``[N,H]``.
        """
        del module, inputs
        values = output.detach().float()
        self.atomic_statistics = (
            float(values.mean()),
            float(values.std(unbiased=False)),
        )

    def _add(self, name: str, value: float) -> None:
        """Accumulate one finite scalar under a stable metric name.

        Args:
            name: LambdaForge metric suffix.
            value: Detached numerical observation.
        """
        if not math.isfinite(value):
            return
        self.sums[name]   = self.sums.get(name, 0.0) + value
        self.counts[name] = self.counts.get(name, 0) + 1

    @staticmethod
    def _gradient_norm(module: nn.Module) -> float:
        """Compute the joint L2 norm of all available gradients in one component.

        Args:
            module: Model or component whose parameter gradients are inspected.

        Returns:
            Euclidean norm, or zero when no parameter received a gradient.
        """
        squared = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float()
                squared += float(gradient.square().sum())
        return math.sqrt(squared)
