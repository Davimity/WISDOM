"""Epoch-bounded gate evidence, distinct final-state and selected-checkpoint audits."""

import json
import torch
import hashlib
import inspect

from typing import Any
from pathlib import Path
from collections.abc import Mapping
from wisdom.models.gating.GateGroup import GateGroup
from wisdom.models.gating.GatePolicy import GatePolicy


class GateDiagnostics:
    """Accumulate small device counters, never per-batch files or retained autograd graphs."""

    def __init__(self, model: Any, work: Any, context: Mapping[str, Any]) -> None:
        """Create an explicit audit artifact and immutable initialization manifest.

        Args:
            model: Fully initialized model with a non-null grouped gate policy, on its device.
            work: Public LF Work supplying output declaration and metric recording.
            context: Effective code/data/seed/model/training contract; never contains local GT.
        """
        self.model = model
        self.work  = work
        self.root  = Path(work.outputs.directory("gate-audit", role="report"))
        self.count = 0
        self.values = next(model.parameters()).new_zeros((len(model.semantic_gates.names), 5))
        model.gate_policy.schedule(model.semantic_gates, 1, int(context["epochs"]))
        widths = self.dimensions(model)
        manifest = model.gate_policy.manifest(model.semantic_gates, widths)
        manifest["context"] = dict(context)
        manifest["code_files"] = {
            cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
            for cls in (GatePolicy, GateDiagnostics, type(model.semantic_gates),
                        type(model.semantic_gates.gates[0]),
                        *(cls for cls in type(model).__mro__
                          if cls.__module__.startswith("wisdom.models")))}
        manifest["code_commit"] = None
        manifest["code_commit_source"] = (
            "Native LF git metadata when available; packaged workers may have no Git checkout. "
            "Preparation HEAD/dirty status: experiments/v10/gate_study.json. "
            "Code digests identify actual files, not an assumed clean HEAD.")
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.snapshot("initial", 0)

    @staticmethod
    def dimensions(model: Any) -> dict[str, int]:
        """Describe source widths separately from latent pathway widths.

        Args:
            model: Assembled model whose chemical tables and branch widths define this inventory.

        Returns:
            Gate-to-feature count: graph branches use their internal latent width, context its
            transferred width, and additive transfer scorers one scalar score per candidate.
        """
        widths = {name: 1 for name in model.semantic_gates.names}
        widths.update({
            "atom.residue_type": model.residue_type_embedding.embedding_dim,
            "atom.hybridization": model.hybridization_embedding.embedding_dim,
            "atom.atom_role": model.atom_role_embedding.embedding_dim,
        })
        widths["surface.atom_context"] = model.hidden_dim
        for name in ("atomic_graph.spatial", "atomic_graph.covalent"):
            widths[name] = model.atomic_encoder.hidden_dim
        representation = model.atomic_representation
        if representation is not None:
            for name in representation.sources:
                widths["atom." + name] = (representation.tables[name].embedding_dim
                    if name in representation.tables else representation.WIDTHS[name])
            if representation.context_projection is not None:
                widths["atom.external_context"] = representation.context_projection.out_features
        return widths

    def observe(self, gradient_scale: float = 1.0) -> None:
        """Accumulate pre-step gradient and actual training-sample statistics on device.

        Args:
            gradient_scale: AMP scale if gradients are still scaled; one after unscale_.
                A nonfinite gradient remains visible as unavailable JSON evidence, not zero.
        """
        registry = self.model.semantic_gates

        # Stack detached scalars once: telemetry must not retain batch graphs or synchronize CUDA.

        samples = torch.stack([registry._last_sample[name] for name in registry.names])
        gradients = torch.stack([gate.log_alpha.grad.detach()
            if gate.log_alpha.grad is not None else samples.new_zeros(())
            for gate in registry.gates]).float() / gradient_scale
        nonfinite = (~torch.isfinite(gradients)).float()
        self.values += torch.stack(((samples == 0).float(), (samples == 1).float(),
                                    gradients, gradients.square(), nonfinite), 1)
        self.count += 1

    def snapshot(self, stage: str, epoch: int) -> dict[str, Any]:
        """Write readable bounded per-gate records and LF group curves at explicit boundaries.

        Args:
            stage: initial/epoch/final/selected; selected describes restored G-chosen inference.
            epoch: Current or selected one-based epoch; initialization uses zero.

        Returns:
            Complete summary including overrides. Epoch rows append JSONL; other stages write
            distinct files. Epoch counters reset only after recording training evidence.
        """
        registry = self.model.semantic_gates
        policy   = self.model.gate_policy
        summary  = registry.summary()
        records  = summary["gates"]
        counters = self.values.detach().cpu().tolist()
        runtime  = registry.inference_state()

        # Separate true stochastic draws from forced controls and nominal warm-up all-on values.

        for index, record in enumerate(records):
            record["group"] = GatePolicy.group(record["name"]).value
            stochastic = (record["available_to_model"] and record["fixed_value"] is None
                and record["posthoc_override"] is None and not record["scheduled_open"]
                and runtime["override"] == "learned")
            record["stochastic_sampling_enabled"] = stochastic
            if stage == "epoch":
                zero, one, gradient, squared, nonfinite = counters[index]
                record.update(
                    sampled_zero_fraction=zero / max(1, self.count),
                    sampled_one_fraction=one / max(1, self.count),
                    gradient_mean=gradient / max(1, self.count) if nonfinite == 0 else None,
                    gradient_rms=(squared / max(1, self.count)) ** .5 if nonfinite == 0 else None,
                    gradient_nonfinite_batches=nonfinite,
                )

        # Report direct closure separately from an inactive ancestor and from a fixed-off design.

        groups = {}
        costs = policy.contributions(registry)
        for group in GateGroup:
            members = [r for r in records if r["group"] == group.value and r["available_to_model"]]
            n = max(1, len(members))
            values = [r["inference_value"] for r in members]
            groups[group.value] = {
                "available_count": len(members),
                "expected_active_count": sum(r["effective_path_probability"] for r in members),
                "closed_count": sum(x == 0 for x in values),
                "learned_closed_count": sum(r["inference_value"] == 0
                    and r["fixed_value"] is None and r["posthoc_override"] is None
                    and not r["scheduled_open"] for r in members),
                "fixed_closed_count": sum(r["fixed_value"] == 0 for r in members),
                "path_closed_count": sum(r["effective_inference_value"] == 0 for r in members),
                "closed_fraction": sum(x == 0 for x in values) / n,
                "open_fraction": sum(x == 1 for x in values) / n,
                "partial_fraction": sum(0 < x < 1 for x in values) / n,
                "coefficient": policy.effective[group],
                "penalty": float(costs[group].detach().cpu()),
            }
            if stage == "epoch":
                groups[group.value].update(
                    sampled_zero_fraction=sum(r["sampled_zero_fraction"] for r in members) / n,
                    sampled_one_fraction=sum(r["sampled_one_fraction"] for r in members) / n,
                    gradient_nonfinite_batches=sum(
                        r["gradient_nonfinite_batches"] for r in members),
                )
                finite_gradients = [r["gradient_rms"] for r in members]
                if all(value is not None for value in finite_gradients):
                    groups[group.value]["gradient_rms_norm"] = sum(
                        value ** 2 for value in finite_gradients) ** .5
                sampled = [r for r in members if r["stochastic_sampling_enabled"]]
                groups[group.value]["stochastic_gate_count"] = len(sampled)
                for boundary in ("zero", "one"):
                    groups[group.value][f"stochastic_sampled_{boundary}_fraction"] = (
                        sum(r[f"sampled_{boundary}_fraction"] for r in sampled) / len(sampled)
                        if sampled else None)
            for name, value in groups[group.value].items():
                if value is not None:
                    self.work.metrics.log(f"gate_{stage}_{group.value}_{name}", float(value),
                                          **({"step": epoch} if stage == "epoch" else {}))
        summary.update(stage=stage, epoch=epoch, groups=groups,
                       inference_state=runtime)
        if stage == "selected":
            summary["checkpoint_artifact"] = "best-model"

        # Publish one compact record per epoch, not one document or trace per minibatch.

        if stage == "epoch":
            with (self.root / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(summary, allow_nan=False) + "\n")
            self.values.zero_()
            self.count = 0
        else:
            (self.root / (stage + ".json")).write_text(json.dumps(summary, indent=2) + "\n")
        return summary
