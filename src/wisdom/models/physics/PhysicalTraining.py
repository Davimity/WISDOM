"""VF supervision, atomic-only pretraining and explicit encoder transfer within Training."""

import json
import time
import torch
import random
import numpy as np

from typing import Any
from typing import cast
from pathlib import Path
from torch import Tensor
from collections.abc import Callable, Mapping
from wisdom.evaluation.PhysicalMetricSuite import PhysicalMetricSuite
from wisdom.models.physics.PhysicalObjective import PhysicalObjective
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive


class PhysicalTraining:
    """Keep physical targets separate from predictor inputs and functional model selection."""

    ATOMIC_PREFIXES = ("atomic_number_embedding.", "residue_type_embedding.",
                       "atom_role_embedding.", "hybridization_embedding.", "atomic_encoder.",
                       "atomic_representation.")
    CONTINUATION = "physics/continuation.pt"

    def __init__(self, work: Any, options: Mapping[str, Any],
                 move_batch: Callable[[Mapping[str, Any], torch.device], dict[str, Any]]) -> None:
        """Bind the existing Work services, objective and host-to-device input selector.

        Args:
            work: Active public Training Work; no nested runner or scheduling is introduced.
            options: mode=joint/pretrain; objective={} uses PhysicalObjective defaults;
                decoder={} uses FieldDecoder defaults. freeze_encoder=false, no_messages=false,
                validation_interval=1, evaluation_queries=4096, gradient_diagnostics=true,
                pretraining_task=fields/masked_element and mask_fraction=0.15.
            move_batch: Training's existing model-input-only device transfer function. Physical
                banks and local functional targets must stay outside its returned mapping.

        Raises:
            ValueError: Unknown mode/task or a physical-only run with no optimization signal.
        """
        self.work = work
        self.options = dict(options)
        self.mode = str(options.get("mode", "joint"))
        self.task = str(options.get("pretraining_task", "fields"))
        if self.mode not in {"joint", "pretrain"} or self.task not in {"fields", "masked_element"}:
            raise ValueError("VF mode must be joint/pretrain and task fields/masked_element")
        if self.mode == "joint" and options.get("no_messages", False):
            raise ValueError("VF no_messages is a physical-only diagnostic; use the explicit "
                             "atomic_representation.atomic_message_control for functional controls")
        self.objective = PhysicalObjective(options.get("objective", {}))
        if self.mode == "pretrain" and self.task == "fields" and not self.objective.weights.any():
            raise ValueError("VF physical pretraining requires a positive field-loss weight")
        self.move_batch = move_batch
        self.statistics: dict[str, Any] = {}
        self.training_groups: list[str] = []
        self.training_sources: list[str] = []
        self.validation_groups: set[str] = set()
        self.validation_sources: set[str] = set()

    def prepare(self, model: Any, datasets: Mapping[str, Any],
                checkpoint: Path | None = None) -> None:
        """Fit train-only transforms, check split independence and optionally transfer weights.

        Args:
            model: WISDOM with an explicitly enabled FieldDecoder.
            datasets: Selected train/val views with explicit existing leakage-group partitions.
            checkpoint: Optional researcher-selected native file; never auto-select another seed.

        Raises:
            ValueError: Missing group provenance, cross-split groups, external representations in
                the principal VF protocol, or incompatible atomic checkpoint semantics/shapes.
        """
        groups = {}
        for split in ("train", "val"):
            groups[split] = {record[5] for record in datasets[split].records}
            if not groups[split] or "unspecified" in groups[split]:
                raise ValueError("VF requires explicit existing leakage groups on train/validation")
        if groups["train"] & groups["val"]:
            raise ValueError("VF physical train and validation share a leakage group")
        self.training_groups = sorted(groups["train"])
        self.validation_groups = groups["val"]
        self.training_sources = sorted({ProteinArchive.sha256(record[0])
                                        for record in datasets["train"].records})
        self.validation_sources = {ProteinArchive.sha256(record[0])
                                    for record in datasets["val"].records}
        # Complete radius neighborhoods are a stored scientific contract, not a top-K budget.
        # Changing Gaussian support alone must not masquerade as changing candidate topology.

        for split in ("train", "val"):
            for paths in datasets[split].structural_inputs.paths.values():
                with np.load(paths["physical_queries"], allow_pickle=False) as stored:
                    header = json.loads(str(stored["metadata_json"].item()))
                if header["parameters"]["decoder_radius"] != model.physics_decoder.radius:
                    raise ValueError("VF decoder radius must match the complete stored query "
                                     "candidate radius; explicitly re-enrich to change support")
        representation = model.atomic_representation
        if representation is not None and (representation.external_mode.value != "none" or
            getattr(representation, "atomic_external_model", None) is not None):
            raise ValueError("principal VF must disable ESM/SaProt/SLAE inputs, not delete assets")
        training_paths = datasets["train"].structural_inputs.paths.values()
        self.statistics = self.objective.fit(paths["physical_queries"] for paths in training_paths)
        self.work.outputs.file("physical-normalization", filename="physical-normalization.json",
            role="diagnostic").write_json({**self.statistics,
                "partitions": "inherited train/validation leakage groups; functional labels unused",
                "options": self.options})
        if checkpoint is not None:
            self.load_encoder(model, checkpoint)
        if self.options.get("freeze_encoder", False):
            for name, parameter in model.named_parameters():
                if name.startswith(self.ATOMIC_PREFIXES) or self.atomic_gate(model, name):
                    parameter.requires_grad_(False)
            model.freeze_physical_encoder = True

    @staticmethod
    def atomic_gate(model: Any, name: str) -> bool:
        """Identify registry parameters belonging to atomic features or relations.

        Args:
            model: WISDOM with stable semantic gate names.
            name: Complete state/parameter name.

        Returns:
            Whether this is an atomic gate, excluding curvature, transfer and surface gates.
        """
        if not name.startswith("semantic_gates.gates."):
            return False
        index = int(name.split(".")[2])
        return model.semantic_gates.names[index].startswith(("atom.", "atomic_graph.", "edge."))

    def load_encoder(self, model: Any, path: Path) -> None:
        """Transfer only compatible atomic weights; keep functional/surface initialization paired.

        Args:
            model: Destination WISDOM model with the same atomic configuration and gate names.
            path: Trusted native tensor checkpoint with physical-encoder parameters and signature.

        Raises:
            ValueError: Different atomic configuration, shapes, missing states or nonfinite weights.
        """
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if self.validation_groups.intersection(saved["training_groups"]) or (
            self.validation_sources.intersection(saved["training_sources"])
        ):
            raise ValueError("VF checkpoint pretraining overlaps the functional "
                             "validation partition")
        if saved.get("atomic_signature") != self.signature(model):
            raise ValueError("VF encoder checkpoint differs from the requested atomic "
                             "configuration")
        destination = model.state_dict()
        expected = {name for name in destination
                    if name.startswith(self.ATOMIC_PREFIXES) or self.atomic_gate(model, name)}
        source = saved["atomic_state"]
        if source.keys() != expected:
            raise ValueError("VF encoder checkpoint has missing or extra atomic tensors")
        for name, value in source.items():
            if value.shape != destination[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"VF invalid transferred atomic tensor: {name}")
            destination[name] = value.to(destination[name].device)
        model.load_state_dict(destination)
        if self.options.get("transfer_decoder", True):
            if any(saved["normalization"][key] != self.statistics[key]
                   for key in ("scale", "mean", "std")):
                raise ValueError("VF decoder transfer requires matching training-only "
                                 "normalization; "
                                 "set transfer_decoder=false to initialize new physical heads")
            model.physics_decoder.load_state_dict(saved["decoder_state"])
        self.work.log(f"VF: restored {len(source)} atomic tensors from an explicit checkpoint; "
                      "functional/surface heads were not transferred")

    @staticmethod
    def signature(model: Any) -> dict[str, Any]:
        """Record architecture choices that tensor dimensions alone cannot distinguish.

        Args:
            model: Instantiated scalar WISDOM model.

        Returns:
            Exact atomic operator, feature/gate layout and available scalar operator settings.
        """
        encoder = model.atomic_encoder
        return {"type": model.atomic_encoder_type.value,
            "atom_spatial_k": model.atom_spatial_k,
            "radial_encoding": getattr(getattr(encoder, "radial_encoding", None), "value", None),
            "gates": list(model.semantic_gates.names),
            "encoder": {key: getattr(encoder, key) for key in (
                "hidden_dim", "output_dim", "distance_scale", "spatial_cutoff",
                "covalent_cutoff", "radial_count") if hasattr(encoder, key)},
            "state_shapes": {name: list(value.shape) for name, value in model.state_dict().items()
                if name.startswith(PhysicalTraining.ATOMIC_PREFIXES)}}

    def prediction(self, model: Any, batch: Mapping[str, Any], device: torch.device,
                   epoch: int = 0, full: bool = False, maximum: int | None = None
                   ) -> tuple[Tensor, dict[str, Tensor]]:
        """Encode atoms once and decode geometry-only queries without a surface forward.

        Args:
            model: Active WISDOM/FieldDecoder.
            batch: CPU collated proteins with physical sidecars and retained atomic geometry.
            device: Granted logical CUDA/CPU device.
            epoch: Deterministic training query-sampling epoch.
            full: Use every fixed-bank query; ordinary validation samples a fixed query set.
            maximum: Optional fixed validation sample size; None uses the training budget.

        Returns:
            Normalized [Q,3] predictions and separate detached reference dictionaries.
        """
        geometry, references = self.objective.queries(batch, device, epoch, full, maximum)
        if not len(geometry["positions"]):
            # Empty physical domains need neither an atomic forward nor device graph copies.
            # Keep a differentiable zero for joint loss composition, never a scored observation.

            zero = model.physics_decoder.global_projection.weight.sum() * 0.0
            return zero.expand(0, 3), references
        inputs = self.move_batch(batch, device)
        if self.options.get("no_messages", False):
            inputs["atom_edge_is_spatial"] = torch.zeros_like(inputs["atom_edge_is_spatial"])
            inputs["atom_edge_is_covalent"] = torch.zeros_like(inputs["atom_edge_is_covalent"])
        states, _ = model.encode_atoms(inputs)
        predictions = model.physics_decoder(states, inputs["atom_positions"], inputs["atom_batch"],
                                             geometry)
        return predictions, references

    def evaluate(self, model: Any, loader: Any, device: torch.device, epoch: int = 0,
                 detailed: bool = False) -> dict[str, float]:
        """Evaluate held-out physical proteins, never functional test or local binding targets.

        Args:
            model: Current/restored physical encoder and decoder.
            loader: Explicit validation partition; no random split is created.
            device: Granted logical model device.
            epoch: Native metric step and detailed audit filename index.
            detailed: Evaluate every bank query and publish per-protein/region original-unit rows.

        Returns:
            Normalized loss and available macro-protein errors/correlations with support counts.
        """
        model.eval()
        began = time.perf_counter()
        suite = PhysicalMetricSuite(float(self.options.get("sign_band", .1)))
        field_losses: dict[str, list[float]] = {name: [] for name in self.objective.FIELDS}
        batches = 0
        unavailable = 0
        with torch.inference_mode():
            for batch in loader:
                # One protein per validation batch bounds full-bank decoding. This metric pass
                # never chooses a functional checkpoint or accesses surface ground truth.

                prediction, references = self.prediction(model, batch, device, full=detailed,
                    maximum=int(self.options.get("evaluation_queries", 4096)))
                selected = [name in self.objective.fields for name in self.objective.FIELDS]
                if not bool(references["valid"][:, selected].any()):
                    unavailable += len(batch["identifier"])
                    continue
                losses = self.objective.loss(prediction, references)
                for name, values in field_losses.items():
                    if name in losses:
                        values.append(float(losses[name]))
                batches += 1
                decoded = self.objective.decode(prediction).cpu().numpy()
                for owner, identifier in enumerate(batch["identifier"]):
                    selected = references["owners"].cpu().numpy() == owner
                    suite.add(identifier, decoded[selected],
                        references["targets"].cpu().numpy()[selected],
                        references["valid"].cpu().numpy()[selected],
                        references["region"].cpu().numpy()[selected])
        if not batches:
            raise ValueError("VF validation has no valid references for the selected fields")
        means = {name: float(np.mean(values)) for name, values in field_losses.items() if values}
        missing = [name for i, name in enumerate(self.objective.FIELDS)
                   if self.objective.weights[i] > 0 and name not in means]
        if missing:
            raise ValueError(f"VF validation lacks positively weighted fields: {missing}")
        total = sum(float(self.objective.weights[i]) * means.get(name, 0.0)
                    for i, name in enumerate(self.objective.FIELDS))
        metrics = {"physics_loss": total, **suite.metrics()}
        metrics["physics_unavailable_proteins"] = float(unavailable)
        metrics["physics_unweighted_loss"] = sum(means.values())
        metrics["physics_evaluation_seconds"] = time.perf_counter() - began
        for name, value in metrics.items():
            self.work.metrics.log(name, value, step=epoch, split="val")
        if detailed:
            self.work.outputs.file("physical-evaluation", filename="physical-evaluation.json",
                role="report").write_json({"metrics": metrics, "rows": suite.rows,
                    "units": ["kBT/e", "kJ/mol", "kJ/mol"],
                    "checkpoint_policy": "physical loss for pretraining; global G for joint"})
        self.work.log(f"VF physical validation: loss={metrics['physics_loss']:.5g}, "
                      f"proteins={batches}, seconds={metrics['physics_evaluation_seconds']:.1f}")
        return metrics

    def gradients(self, model: Any, functional: Tensor, physical: Tensor, epoch: int) -> None:
        """Log shared-encoder gradient norms and cosine without changing training gradients.

        Differentiate the retained forward graph with autocast disabled, as for ordinary AMP
        backward. CUDA's sparse surface derivative backward requires FP32 even when surrounding
        dense layers use BF16/FP16. This protects direct callers too; no forward is recomputed,
        no parameter .grad is populated, and the optimizer can still backpropagate the joint loss.

        Args:
            model: Shared atomic encoder with trainable parameters.
            functional: Protein-label loss; never local binding targets.
            physical: Weighted normalized auxiliary field loss.
            epoch: Native LF curve step; diagnostic does not alter either gradient.

        Raises:
            RuntimeError: Autograd cannot differentiate the existing loss graph. Unknown failures
                propagate rather than disabling the diagnostic or replacing gradients with zeros.
        """
        parameters = tuple(p for p in model.atomic_encoder.parameters() if p.requires_grad)
        if not parameters:
            return
        with torch.autocast(device_type=parameters[0].device.type, enabled=False):
            vectors = [torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                       for loss in (functional, physical)]
        square = [sum(float(g.detach().float().square().sum()) for g in vector if g is not None)
                  for vector in vectors]
        for name, norm in zip(("functional", "physical"), square, strict=True):
            self.work.metrics.log(f"physics_{name}_gradient_norm", norm**.5, step=epoch,
                                  split="train")
        if all(norm > 0 for norm in square):
            dot = sum(float((left.detach().float() * right.detach().float()).sum())
                for left, right in zip(*vectors, strict=True)
                if left is not None and right is not None)
            self.work.metrics.log("physics_gradient_cosine", dot / (square[0] * square[1])**.5,
                                  step=epoch, split="train")

    def pretrain(self, model: Any, loaders: Mapping[str, Any], device: torch.device,
                 epochs: int, patience: int, minimum_delta: float, learning_rate: float,
                 weight_decay: float, clip_norm: float | None,
                 generator: torch.Generator) -> dict[str, Any]:
        """Run physical-only optimization inside the ordinary native Training lifecycle.

        Args:
            model: WISDOM whose atomic encoder/decoder are the only modules executed.
            loaders: Explicit train and val loaders; functional labels are never consumed.
            device: Granted logical device, respecting CUDA_VISIBLE_DEVICES.
            epochs: Maximum completed optimizer epochs.
            patience: Physical-validation non-improvement budget.
            minimum_delta: Required improvement in normalized physical validation loss.
            learning_rate: AdamW learning rate.
            weight_decay: AdamW decay, excluding semantic gate parameters.
            clip_norm: Optional finite global gradient norm ceiling.
            generator: Existing loader RNG, restored together with optimizer and model state.

        Returns:
            Best physical validation epoch/loss and published transferable checkpoint. A separate
            native continuation preserves completed epochs; it never adopts an incompatible Run.
        """
        optimizer = torch.optim.AdamW([
            {"params": [p for name, p in model.named_parameters() if p.requires_grad and
                        not name.startswith("semantic_gates.")], "weight_decay": weight_decay},
            {"params": [p for p in model.semantic_gates.parameters() if p.requires_grad],
             "weight_decay": 0.0}], lr=learning_rate)
        for label, module in (("atomic", model.atomic_encoder), ("decoder", model.physics_decoder)):
            self.work.metrics.log(f"physics_{label}_parameters",
                                  float(sum(p.numel() for p in module.parameters())))
            self.work.metrics.log(f"physics_{label}_trainable_parameters",
                                  float(sum(p.numel() for p in module.parameters()
                                            if p.requires_grad)))
        first, best_epoch, stale, best_loss = 1, 0, 0, float("inf")
        patience_reference = float("inf")
        best_state: dict[str, Tensor] = {}
        if self.work.checkpoints.exists(self.CONTINUATION):
            saved = torch.load(self.work.checkpoints.file(self.CONTINUATION),
                               map_location="cpu", weights_only=True)
            model.load_state_dict(saved["state"])
            optimizer.load_state_dict(saved["optimizer"])
            first, best_epoch, stale = saved["epoch"] + 1, saved["best_epoch"], saved["stale"]
            best_loss, best_state = saved["best_loss"], saved["best_state"]
            patience_reference = saved["patience_reference"]
            generator.set_state(saved["loader_rng"])
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            numpy = saved["numpy_rng"]
            np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
            if saved["cuda_rng"]:
                if len(saved["cuda_rng"]) != torch.cuda.device_count():
                    raise ValueError("VF continuation requires the same granted logical GPU count")
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            self.work.log(f"VF: restored completed physical epoch {first - 1}")
        final_epoch = first if self.task == "fields" and stale >= patience else epochs + 1
        for epoch in range(first, final_epoch):
            began = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            losses: list[float] = []
            gradient_measured = False
            for batch in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                if self.task == "masked_element":
                    inputs = self.move_batch(batch, device)
                    original = inputs["atomic_numbers"]
                    mask = torch.rand(len(original), device=device) < float(
                        self.options.get("mask_fraction", .15))
                    mask[0] = True
                    # Hide the queried element and its atom-local chemistry. Neighbor topology
                    # and unmasked context remain a non-physical reconstruction signal.

                    for name in ("atomic_numbers", "residue_type_ids", "atom_role_ids",
                        "atom_hybridization_ids", "formal_charges", "atom_aromaticity",
                        "atom_hbond_donor", "atom_hbond_acceptor", "residue_hydropathy",
                        "residue_polarity"):
                        inputs[name] = inputs[name].clone()
                        inputs[name][mask] = 0
                    inputs["atomic_feature_mask"] = mask
                    states, _ = model.encode_atoms(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        model.physics_decoder.element_head(states[mask]), original[mask])
                else:
                    prediction, reference = self.prediction(model, batch, device, epoch)
                    selected = (self.objective.weights > 0).tolist()
                    if not bool(reference["valid"][:, selected].any()):
                        # No optimizer/weight-decay update and no artificial zero-loss observation
                        # is allowed when the entire batch lacks physical supervision.

                        continue
                    loss = self.objective.loss(prediction, reference)["total"]
                if not torch.isfinite(loss):
                    raise ValueError("VF optimization produced a nonfinite physical/control loss")
                loss.backward()  # type: ignore[no-untyped-call]
                if not gradient_measured:
                    norm = sum(float(p.grad.detach().float().square().sum())
                               for p in model.atomic_encoder.parameters() if p.grad is not None)**.5
                    self.work.metrics.log("physics_encoder_gradient_norm", norm,
                                          step=epoch, split="train")
                    gradient_measured = True
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm,
                                                  error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = self.evaluate(model, loaders["val"], device, epoch)
            # Masked-category controls use a fixed update budget, not selection by untrained
            # physical heads. Their final epoch transfers; physical errors stay diagnostic.

            value = validation["physics_loss"] if self.task == "fields" else -float(epoch)
            if value < best_loss:
                best_loss, best_epoch = value, epoch
                best_state = {key: tensor.detach().cpu().clone()
                              for key, tensor in model.state_dict().items()}
            if value < patience_reference - minimum_delta:
                patience_reference, stale = value, 0
            else:
                stale += 1
            numpy_state = cast(Any, np.random.get_state())
            payload = {"state": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_epoch": best_epoch, "best_loss": best_loss,
                "patience_reference": patience_reference,
                "best_state": best_state, "stale": stale, "loader_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}
            self.work.checkpoints.file(self.CONTINUATION,
                build=lambda path, state=payload: torch.save(state, path), step=epoch,
                validate=lambda path, expected=epoch: torch.load(
                    path, map_location="cpu", weights_only=True)["epoch"] == expected)
            self.work.metrics.log("physics_loss", float(np.mean(losses)), step=epoch, split="train")
            self.work.metrics.log("epoch_seconds", time.perf_counter() - began, step=epoch)
            if device.type == "cuda":
                self.work.metrics.log("physics_peak_allocated_gib",
                    torch.cuda.max_memory_allocated(device) / 1024**3, step=epoch)
            self.work.progress.update(completed=epoch, total=epochs, message="VF physical training")
            self.work.log(f"VF epoch={epoch}/{epochs} physical_train={np.mean(losses):.5g} "
                          f"physical_val={validation['physics_loss']:.5g} "
                          f"patience={stale}/{patience}")
            if self.work.stop_requested or (self.task == "fields" and stale >= patience):
                break
        model.load_state_dict(best_state)
        metrics = self.evaluate(model, loaders["val"], device, best_epoch, detailed=True)
        atomic = {key: value.cpu() for key, value in model.state_dict().items()
                  if key.startswith(self.ATOMIC_PREFIXES) or self.atomic_gate(model, key)}
        destination = Path(self.work.outputs.file("physical-encoder",
                            filename="physical-encoder.pt", role="model"))
        torch.save({"atomic_state": atomic, "atomic_signature": self.signature(model),
            "training_groups": self.training_groups, "training_sources": self.training_sources,
            "normalization": self.statistics, "options": self.options,
            "decoder_state": model.physics_decoder.state_dict(), "best_epoch": best_epoch,
            "physical_metrics": metrics, "functional_labels_used": False}, destination)
        return {"physical_checkpoint": str(destination), "best_epoch": best_epoch,
                "metrics": metrics, "functional_training": False}
