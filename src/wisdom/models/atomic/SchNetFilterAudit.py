"""Bounded, target-free inspection of SchNet filters and initialization/capacity contracts."""

import json
import torch
import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from matplotlib.figure import Figure


class SchNetFilterAudit:
    """Inspect the actual LF filters without fitting, ranking or interpreting them as energies."""

    @staticmethod
    def configuration(model: Any) -> dict[str, Any]:
        """Describe static capacities and hash all weights/buffers outside the changed encoder.

        Args:
            model: Constructed WISDOM model, before optimizer updates for initialization hashes.

        Returns:
            JSON-compatible encoder/filter parameter counts, actual widths and radial settings,
            plus a common-module SHA-256. Equal hashes establish exact common initialization;
            equal seeds alone do not. Neither count estimates activations or effective capacity.
        """
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            if name.startswith("atomic_encoder."):
                continue
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        encoder = model.atomic_encoder
        filters = [operator.filter for relation in ("covalent", "spatial")
                   for operator in getattr(encoder, relation, ()) if hasattr(operator, "filter")]
        result: dict[str, Any] = {
            "shared_initialization_sha256": digest.hexdigest(),
            "atomic_encoder_total_parameter_count": sum(p.numel() for p in encoder.parameters()),
            "atomic_encoder_trainable_parameter_count": sum(
                p.numel() for p in encoder.parameters() if p.requires_grad),
            "atomic_filter_parameter_count": sum(
                p.numel() for f in filters for p in f.parameters()),
            "atomic_hidden_dim": int(getattr(encoder, "hidden_dim", model.hidden_dim)),
            "atomic_output_dim": int(getattr(encoder, "output_dim", model.hidden_dim)),
        }
        if model.atomic_encoder_type.value.startswith("schnet"):
            basis = encoder.covalent[0].radial_basis
            result["radial_encoding"] = encoder.radial_encoding.value
            result["radial_channels"] = basis.num_radial
            result["reference_radius_angstrom"] = basis.radius
            result["centers_angstrom"] = (
                basis.centers.detach().cpu().tolist() if hasattr(basis, "centers") else None)
            result["gamma_inverse_angstrom_squared"] = (
                0.5 / basis.width**2 if hasattr(basis, "width") else None)
            result["edge_metadata"] = (
                "five_metadata_sources_unavailable"
                if model.atomic_encoder_type.value == "schnet_radial" else "full_semantic_gates")
            width = encoder.hidden_dim
            result["filter_dense_macs_per_directed_edge_per_layer"] = {
                relation: width * (basis.num_radial + attributes) + width**2
                for relation, attributes in (("covalent", 3), ("spatial", 4))}
            result["output_projection_macs_per_atom"] = (
                width * encoder.output_dim if width != encoder.output_dim else 0)
            result["mac_scope"] = (
                "Multiply-accumulates in filter Linear layers and optional output projection only; "
                "excludes basis evaluation, biases, activations, messages, updates, backward and "
                "surface processing. Not a whole-model FLOP estimate.")
        return result

    @staticmethod
    def write(model: Any, destination: Path) -> dict[str, Any]:
        """Sample actual learned filters on a fixed distance grid after best-checkpoint restore.

        Args:
            model: Restored SchNet model. Evaluation gate samples are deterministic; no RNG draws,
                training, task targets, molecule coordinates or DataLoader access is required.
            destination: Existing LF-owned output directory for JSON, compact NPZ and PNG curves.

        Returns:
            Metadata and per-layer/profile amplitude, finite-difference sensitivity and channel
            correlation summaries. Full [121,H] responses before/after support/relation gates are
            saved in NPZ. PNG shows only the first eight latent channels to keep plots readable.

        Raises:
            ValueError: The source model is not a SchNet variant.
            OSError: Output files cannot be written.
        """
        if not model.atomic_encoder_type.value.startswith("schnet"):
            raise ValueError("radial filter inspection requires a SchNet checkpoint")
        encoder = model.atomic_encoder
        device = next(encoder.parameters()).device
        distances = torch.linspace(0, encoder.spatial_cutoff, 121, device=device)
        gates = model.semantic_gates.sample(training=False)
        arrays: dict[str, Any] = {"distance_angstrom": distances.cpu().numpy()}
        result = SchNetFilterAudit.configuration(model)
        # This digest describes the restored weights, not their pretraining initialization.

        result["shared_checkpoint_sha256"] = result.pop("shared_initialization_sha256")
        result["interpretation"] = (
            "Latent weakly supervised filters, not physical energy potentials.")
        result["gate_values"] = {name: float(value.detach().cpu()) for name, value in gates.items()}
        result["response_stages"] = (
            "before_support already includes current radial/metadata source gates; "
            "after_support_relation_gate additionally includes LF support and the relation gate. "
            "Sensitivity uses adjacent samples (0.05 Å); no derivatives of physical energy.")
        result["profiles"] = []
        figure = Figure(figsize=(12, max(4, 4 * encoder.layers)), layout="constrained")
        axes = figure.subplots(encoder.layers, 2, squeeze=False)

        # Attribute tuples follow the exact forward normalization. Covalent profiles distinguish
        # an intra-residue single bond and a peptide-like cross-residue bond; spatial profiles
        # distinguish same-chain nonlocal neighbors and inter-chain neighbors. No mixture average.

        profiles: dict[str, dict[str, tuple[float, ...]]] = {
            "covalent": {"intra_residue_single": (1 / 3, 1.0),
                         "cross_residue_single": (1 / 3, 0.0)},
            "spatial": {"same_chain_nonlocal": (0.0, 1.0, 1.0),
                        "different_chain": (0.0, 0.0, 0.0)},
        }
        for column, relation in enumerate(("covalent", "spatial")):
            fields = ("bond_order", "same_residue") if relation == "covalent" else (
                "same_residue", "same_chain", "residue_separation")
            for layer, operator in enumerate(getattr(encoder, relation)):
                for profile, values in profiles[relation].items():
                    prefix = f"{relation}_layer{layer}_{profile}"
                    with torch.no_grad():
                        radial, support = operator.radial_inputs(
                            distances, distance_weights=gates[f"edge.{relation}.distance"])
                        attributes = [(distances / encoder.distance_scale).clamp(max=2)
                                      * gates[f"edge.{relation}.distance"]]
                        attributes.extend(torch.full_like(distances, value)
                                          * gates[f"edge.{relation}.{field}"]
                                          for field, value in zip(fields, values, strict=True))
                        raw = operator.filter(torch.cat((radial, torch.stack(attributes, 1)), 1))
                        filtered = raw * support[:, None] * gates[f"atomic_graph.{relation}"]
                    raw_values = raw.float().cpu().numpy()
                    filtered_values = filtered.float().cpu().numpy()
                    arrays[prefix + "_before_support"] = raw_values
                    arrays[prefix + "_after_support_relation_gate"] = filtered_values
                    arrays[prefix + "_radial_input"] = radial.float().cpu().numpy()
                    arrays[prefix + "_support"] = support.float().cpu().numpy()
                    slopes = np.diff(filtered_values, axis=0) / float(distances[1] - distances[0])
                    centered = filtered_values - filtered_values.mean(0)
                    lengths = np.linalg.norm(centered, axis=0)
                    varying = lengths > 1e-10
                    normalized = centered[:, varying] / lengths[varying]
                    correlation = normalized.T @ normalized
                    off_diagonal = ~np.eye(correlation.shape[0], dtype=bool)
                    result["profiles"].append({
                        "relation": relation, "layer": layer, "profile": profile,
                        "metadata_normalized_before_gates": dict(zip(fields, values, strict=True)),
                        "filter_rms": float(np.sqrt(np.mean(filtered_values**2))),
                        "sensitivity_rms_per_angstrom": float(np.sqrt(np.mean(slopes**2))),
                        "varying_channels": int(varying.sum()),
                        "mean_absolute_channel_correlation": (
                            float(np.abs(correlation[off_diagonal]).mean())
                            if off_diagonal.any() else None),
                    })
                    if profile == next(iter(profiles[relation])):
                        axes[layer, column].plot(
                            arrays["distance_angstrom"], filtered_values[:, :8])
                        axes[layer, column].set(
                            title=f"{relation}, layer {layer}: {profile}", xlabel="Distance (Å)",
                            ylabel="Filter after support and relation gate")
        np.savez_compressed(destination / "radial-filters.npz", **arrays)
        figure.savefig(destination / "radial-filters.png", dpi=120)
        (destination / "radial-filters.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
