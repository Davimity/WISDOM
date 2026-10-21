"""Bounded synthetic V8d cost audit; never submits a Work or reads a biological/test dataset."""

# ruff: noqa: I001

import json
import time
import yaml
import torch
import argparse
import statistics

from pathlib import Path
from inspect import signature
from wisdom.Training import _create_model
from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.WisdomV2 import WisdomV2
from wisdom.models.ProteinTaskLoss import ProteinTaskLoss
from wisdom.models.WeakSurfaceLoss import WeakSurfaceLoss


def synthetic_batch(
    atoms: int,
    points: int,
    device: torch.device,
    curvature_scales: int = 5,
) -> dict:
    """Construct two unequal synthetic proteins with shared fixed sparse physical-like tensors.

    Args:
        atoms: First protein atom count, at least 16; the second has seven more.
        points: First protein point count, at least 32; the second has sixteen more.
        device: Explicit CPU/CUDA target for the disposable microbenchmark.
        curvature_scales: Persisted scale count; five reproduces the audited V8b constructor.

    Returns:
        Model inputs plus global binary labels. No local GT, real protein or dense pair matrix.
        The DCT modes are orthonormal toy operators, not a biologically validated NPZ.
    """
    sizes = ((atoms, points), (atoms + 7, points + 16))
    total_atoms = sum(size[0] for size in sizes)
    total_points = sum(size[1] for size in sizes)
    fields = {
        name: []
        for name in (
            "atom_edge_index",
            "atom_edge_is_covalent",
            "atom_edge_distance",
            "surface_atom_neighbors",
            "surface_atom_distances",
            "surface_batch",
            "atom_batch",
        )
    }
    operators, boundaries = [], [0]
    offset = 0
    for owner, (nodes, samples) in enumerate(sizes):
        indices = torch.arange(nodes, device=device)
        positions = torch.stack((indices % 8, indices // 8 % 8, indices // 64), dim=1).float()
        source = torch.cat((indices[:-1], indices[:-8]))
        target = torch.cat((indices[1:], indices[8:]))
        fields["atom_edge_index"].append(torch.stack((source, target)) + offset)
        fields["atom_edge_is_covalent"].append(torch.arange(len(source), device=device) < nodes - 1)
        fields["atom_edge_distance"].append((positions[source] - positions[target]).norm(dim=1))
        candidates = (
            torch.arange(samples, device=device)[:, None] + torch.arange(16, device=device)[None, :]
        ) % nodes
        fields["surface_atom_neighbors"].append(candidates + offset)
        fields["surface_atom_distances"].append(
            torch.linspace(1.0, 3.9, 16, device=device).expand(samples, -1)
        )
        fields["surface_batch"].append(
            torch.full((samples,), owner, device=device, dtype=torch.long)
        )
        fields["atom_batch"].append(torch.full((nodes,), owner, device=device, dtype=torch.long))
        modes = min(32, samples)
        frequency = torch.arange(modes, device=device).float()
        phase = torch.arange(samples, device=device).float() + 0.5
        phi = (
            torch.cos(torch.pi * phase[:, None] * frequency[None, :] / samples)
            * (2 / samples) ** 0.5
        )
        phi[:, 0] /= 2**0.5
        diagonal = torch.arange(samples, device=device)
        operators.append(
            {
                "mass": torch.ones(samples, device=device),
                "eigenvalues": frequency.square() / samples**2,
                "eigenvectors": phi,
                "gradient_index": torch.stack((diagonal, diagonal)),
                "gradient_x": torch.zeros(samples, device=device),
                "gradient_y": torch.zeros(samples, device=device),
            }
        )
        boundaries.append(boundaries[-1] + samples)
        offset += nodes
    edges = torch.cat(fields["atom_edge_index"], dim=1)
    lengths = torch.cat(fields["surface_atom_distances"])
    result = {
        "atomic_numbers": torch.full((total_atoms,), 6, device=device, dtype=torch.long),
        "residue_type_ids": torch.ones(total_atoms, device=device, dtype=torch.long),
        "atom_edge_index": edges,
        "atom_edge_is_spatial": torch.ones(edges.shape[1], device=device, dtype=torch.bool),
        "atom_edge_is_covalent": torch.cat(fields["atom_edge_is_covalent"]),
        "atom_edge_distance": torch.cat(fields["atom_edge_distance"]),
        "atom_edge_bond_order": torch.ones(edges.shape[1], device=device),
        "atom_edge_same_residue": torch.ones(edges.shape[1], device=device),
        "atom_edge_same_chain": torch.ones(edges.shape[1], device=device),
        "atom_edge_residue_separation": torch.ones(edges.shape[1], device=device),
        "surface_curvatures": torch.zeros(total_points, curvature_scales, 3, device=device),
        "surface_atom_neighbors": torch.cat(fields["surface_atom_neighbors"]),
        "surface_atom_distances": lengths,
        "surface_atom_normal_offsets": lengths * 0.5,
        "surface_atom_tangential_distances": lengths * (0.75**0.5),
        "surface_atom_mask": torch.ones_like(lengths, dtype=torch.bool),
        "surface_area_weights": torch.ones(total_points, device=device),
        "surface_batch": torch.cat(fields["surface_batch"]),
        "atom_batch": torch.cat(fields["atom_batch"]),
        "surface_ptr": torch.tensor(boundaries, device=device),
        "surface_operators": operators,
    }
    for name in ("atom_role_ids", "atom_hybridization_ids"):
        result[name] = torch.zeros(total_atoms, device=device, dtype=torch.long)
    for name in (
        "formal_charges",
        "atom_aromaticity",
        "atom_hbond_donor",
        "atom_hbond_acceptor",
        "residue_hydropathy",
        "residue_polarity",
    ):
        result[name] = torch.zeros(total_atoms, device=device)
    return result


def synchronize(device: torch.device) -> None:
    """Finish only this process's device stream before measuring a timed boundary.

    Args:
        device: Explicit benchmark target; CPU performs no synchronization.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    """Measure finite synthetic steps with the authored model/head/prior configuration.

    Returns:
        Prints JSON with exact parameter counts and median forward/loss, backward and optimizer
        times. CUDA includes allocated peak VRAM; CPU VRAM, epoch timing and FLOPs remain null.
        No comparison here estimates a production epoch or constitutes predictive evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/v8/wisdom_v8d.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--modes", nargs="+", choices=[f"d{i}" for i in range(9)], default=["d0", "d3", "d5", "d6"]
    )
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--curvature-scales", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    if (
        args.atoms < 16
        or args.points < 32
        or args.curvature_scales < 1
        or not 1 <= args.repeats <= 100
    ):
        parser.error("atoms>=16, points>=32, curvature-scales>=1 and 1<=repeats<=100 are required")
    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    inputs = synthetic_batch(args.atoms, args.points, device, args.curvature_scales)
    target = torch.tensor([0.0, 1.0], device=device)
    records = []
    for pooling in ("max", "attention"):
        candidate = next(
            item for item in config["sweep"]["candidates"] if item["pooling_type"] == pooling
        )
        for mode in args.modes:
            torch.manual_seed(151)
            parameters = (
                config["with"]
                | candidate
                | {
                    "feedback_propagation": mode,
                    "checkpoint_surface": args.checkpoint,
                    "checkpoint_post_feedback": args.checkpoint,
                    "curvature_features": args.curvature_scales * 3,
                }
            )
            accepted = signature(WisdomV1).parameters.keys() | signature(WisdomV2).parameters.keys()
            model, _ = _create_model(
                2, {name: value for name, value in parameters.items() if name in accepted}
            )
            model.to(device).train()
            # Use the same protein-level losses and weak priors, never a local synthetic target.

            task = ProteinTaskLoss(
                parameters["head_type"],
                parameters.get("head_loss_balance", "legacy_sum"),
                parameters.get("surface_loss_fraction", 0.5),
            )
            weak = WeakSurfaceLoss(
                negative_weight=parameters["negative_surface_lambda"],
                regional_positive_weight=parameters["regional_positive_lambda"],
                regional_ranking_weight=parameters["regional_ranking_lambda"],
            )
            gates = list(model.semantic_gates.gate_parameters())
            gate_ids = {id(parameter) for parameter in gates}
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [p for p in model.parameters() if id(p) not in gate_ids],
                        "weight_decay": parameters["weight_decay"],
                    },
                    {"params": gates, "weight_decay": 0.0},
                ],
                lr=parameters["learning_rate"],
            )
            samples = []
            peaks = []
            amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
            for repetition in range(args.repeats + 1):
                optimizer.zero_grad(set_to_none=True)
                synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    out = model(**inputs)
                    loss = (
                        task(out, target)["total"]
                        + parameters["gate_lambda"] * model.gate_regularization()
                    )
                    loss = (
                        loss
                        + weak(
                            out["surface_logits"],
                            inputs["surface_area_weights"],
                            inputs["surface_batch"],
                            target,
                            inputs["surface_operators"],
                            inputs["surface_ptr"],
                        )["total"]
                    )
                synchronize(device)
                forwarded = time.perf_counter()
                loss.backward()
                synchronize(device)
                backwarded = time.perf_counter()
                optimizer.step()
                synchronize(device)
                finished = time.perf_counter()
                if device.type == "cuda":
                    peaks.append(torch.cuda.max_memory_allocated(device) / 2**30)
                if repetition:
                    samples.append(
                        (
                            forwarded - started,
                            backwarded - forwarded,
                            finished - backwarded,
                            finished - started,
                        )
                    )
                del out, loss
            records.append(
                {
                    "pooling": pooling,
                    "mode": mode,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "post_parameters": sum(
                        p.numel() for p in model.post_feedback_propagation.parameters()
                    )
                    if model.post_feedback_propagation is not None
                    else 0,
                    **{
                        name: statistics.median(sample[index] for sample in samples)
                        for index, name in enumerate(
                            ("forward_loss_s", "backward_s", "optimizer_s", "step_s")
                        )
                    },
                    "peak_allocated_gib": max(peaks) if peaks else None,
                }
            )
            del model, optimizer, task, weak
    print(
        json.dumps(
            {
                "scope": "synthetic microbenchmark; NOT production feasibility or results",
                "device": str(device),
                "atoms": [args.atoms, args.atoms + 7],
                "points": [args.points, args.points + 16],
                "measured_repeats": args.repeats,
                "checkpointing": args.checkpoint,
                "curvature_scales": args.curvature_scales,
                "median_epoch_seconds": None,
                "flops": None,
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
