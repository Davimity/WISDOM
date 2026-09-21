"""Reproducible before/after benchmarks for exact WISDOM runtime refactors.

Run the representative CUDA suite with::

    python benchmarks/benchmark_runtime.py --device cuda --output runtime-benchmark.json

The legacy functions in this file intentionally preserve the pre-refactor execution order. They
are benchmark references, not alternative trainable architectures.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import time
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from wisdom.models.DeltaConvSurfaceEncoder import DeltaConvSurfaceEncoder
from wisdom.models.DiffusionBlock import DiffusionBlock
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.GatedAtomicEncoder import GatedAtomicEncoder
from wisdom.models.SurfaceAtomTransfer import SurfaceAtomTransfer
from wisdom.models.WisdomV1 import WisdomV1


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work so wall-clock regions have real boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    operation : Callable[[], Tensor],
    device    : torch.device,
    iterations: int,
    backward  : bool,
) -> dict[str, float]:
    """Measure a warmed operation, optional backward, and CUDA allocator peaks."""
    for _ in range(3):
        result = operation()
        if backward:
            result.square().mean().backward()
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    for _ in range(iterations):
        result = operation()
        if backward:
            result.square().mean().backward()
    synchronize(device)
    elapsed = time.perf_counter() - started

    return {
        "milliseconds": 1_000.0 * elapsed / iterations,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0
        ),
    }


def atomic_inputs(
    atoms     : int,
    degree    : int,
    input_dim : int,
    device    : torch.device,
) -> tuple[Tensor, ...]:
    """Create deterministic compact undirected atom pairs and symmetric edge attributes."""
    generator = torch.Generator().manual_seed(2026)
    pairs = {
        (min(source, (source + offset) % atoms), max(source, (source + offset) % atoms))
        for source in range(atoms)
        for offset in range(1, degree // 2 + 1)
        if source != (source + offset) % atoms
    }
    edges = torch.tensor(sorted(pairs), dtype=torch.long).T.contiguous()
    count = edges.shape[1]

    return (
        torch.randn(atoms, input_dim, generator=generator).to(device),
        edges.to(device),
        (torch.arange(count) % 3 != 0).to(device),
        (torch.arange(count) % 5 == 0).to(device),
        (1.0 + 5.0 * torch.rand(count, generator=generator)).to(device),
        (torch.arange(count) % 3 == 0).float().to(device),
        (torch.arange(count) % 4 == 0).to(device),
        (torch.arange(count) % 7 != 0).to(device),
        (torch.arange(count) % 24).float().to(device),
    )


def gates(reference: Tensor) -> dict[str, Tensor]:
    """Return deterministic non-trivial gates for isolated atomic benchmarking."""
    names = (
        "atomic_graph.spatial",
        "atomic_graph.covalent",
        "edge.spatial.distance",
        "edge.spatial.same_residue",
        "edge.spatial.same_chain",
        "edge.spatial.residue_separation",
        "edge.covalent.distance",
        "edge.covalent.bond_order",
        "edge.covalent.same_residue",
    )
    return {
        name: reference.new_tensor(0.3 + 0.6 * index / (len(names) - 1))
        for index, name in enumerate(names)
    }


def legacy_atomic_forward(
    encoder  : GatedAtomicEncoder,
    arguments: tuple[Tensor, ...],
    active_gates: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Evaluate the former duplicated-directed, per-edge H-by-H projection path."""
    features, compact_edges, spatial, covalent, distance, bond, residue, chain, separation = (
        arguments
    )
    if active_gates is None:
        active_gates = gates(features)
    hidden       = encoder.input_projection(features)

    directed = torch.cat((compact_edges, compact_edges.flip(0)), dim=1)
    source, target = directed
    spatial  = torch.cat((spatial, spatial))
    covalent = torch.cat((covalent, covalent))
    distance = torch.cat((distance, distance))
    bond     = torch.cat((bond, bond))
    residue  = torch.cat((residue, residue))
    chain    = torch.cat((chain, chain))
    separation = torch.cat((separation, separation))

    spatial_attributes = torch.stack(
        (
            (distance / encoder.distance_scale).clamp(max=2.0)
            * active_gates["edge.spatial.distance"],
            residue.float() * active_gates["edge.spatial.same_residue"],
            chain.float() * active_gates["edge.spatial.same_chain"],
            (separation / encoder.residue_scale).clamp(max=2.0)
            * active_gates["edge.spatial.residue_separation"],
        ),
        dim=1,
    )
    covalent_attributes = torch.stack(
        (
            (distance / encoder.distance_scale).clamp(max=2.0)
            * active_gates["edge.covalent.distance"],
            bond.clamp(min=0.0, max=3.0) / 3.0 * active_gates["edge.covalent.bond_order"],
            residue.float() * active_gates["edge.covalent.same_residue"],
        ),
        dim=1,
    )

    layers = zip(
        encoder.spatial_messages,
        encoder.covalent_messages,
        encoder.spatial_conditioners,
        encoder.covalent_conditioners,
        encoder.normalizations,
        strict=True,
    )
    for spatial_message, covalent_message, spatial_score, covalent_score, norm in layers:
        updates = []
        for mask, message, score, attributes in (
            (spatial, spatial_message, spatial_score, spatial_attributes),
            (covalent, covalent_message, covalent_score, covalent_attributes),
        ):
            selected_source = source[mask]
            selected_target = target[mask]
            selected_values = attributes[mask]
            scale = score(selected_values) - score(torch.zeros_like(selected_values))
            messages = message(hidden[selected_source]) * (1.0 + scale)
            aggregate = torch.zeros_like(hidden).index_add_(0, selected_target, messages)
            degree = torch.bincount(selected_target, minlength=len(hidden)).to(hidden.dtype)
            updates.append(aggregate / degree.clamp_min(1.0).unsqueeze(1))

        hidden = norm(
            hidden
            + active_gates["atomic_graph.spatial"] * updates[0]
            + active_gates["atomic_graph.covalent"] * updates[1]
        )
        hidden = torch.nn.functional.silu(hidden)
    return hidden


def optimized_atomic_forward(
    encoder  : GatedAtomicEncoder,
    arguments: tuple[Tensor, ...],
) -> Tensor:
    """Evaluate the optimized compact-edge factorization with identical learned weights."""
    return encoder(*arguments, gates(arguments[0]))


def legacy_surface_forward(
    transfer  : SurfaceAtomTransfer,
    embeddings: Tensor,
    neighbors : Tensor,
    distances : Tensor,
    offsets   : Tensor,
    tangential: Tensor,
    mask      : Tensor,
    active_gates: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Evaluate the former repeated-normalization and explicit weighted-tensor reduction."""
    if active_gates is None:
        active_gates = {
            "transfer.orientation": embeddings.new_ones(()),
            "transfer.atom_content": embeddings.new_ones(()),
        }
    outputs = []
    for start in range(0, len(neighbors), transfer.chunk_size):
        stop        = min(start + transfer.chunk_size, len(neighbors))
        atom_ids    = neighbors[start:stop].clamp_min(0)
        valid       = mask[start:stop] & (distances[start:stop] <= transfer.radius)
        gathered    = embeddings[atom_ids]
        normalized  = (distances[start:stop] / transfer.radius).unsqueeze(-1)
        orientation = torch.stack(
            (
                distances[start:stop] / transfer.radius,
                offsets[start:stop] / transfer.radius,
                tangential[start:stop] / transfer.radius,
            ),
            dim=-1,
        )
        base = transfer.distance_scorer(normalized).squeeze(-1)
        orientation_score = (
            transfer.orientation_scorer(orientation)
            - transfer.orientation_scorer(torch.zeros_like(orientation))
        ).squeeze(-1)
        content = torch.cat((normalized, gathered), dim=-1)
        content_score = (
            transfer.content_scorer(content)
            - transfer.content_scorer(
                torch.cat((normalized, torch.zeros_like(gathered)), dim=-1)
            )
        ).squeeze(-1)
        scores = (
            base
            + active_gates["transfer.orientation"] * orientation_score
            + active_gates["transfer.atom_content"] * content_score
        ).masked_fill(
            ~valid,
            torch.finfo(base.dtype).min,
        )
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        outputs.append(torch.sum(weights.unsqueeze(-1) * gathered, dim=1))
    return torch.cat(outputs)


def legacy_diffusion_forward(
    block   : DiffusionBlock,
    features: Tensor,
    operator: Mapping[str, Tensor],
    gradients: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Evaluate the former two sparse and two dense tangent-gradient launches."""
    values         = features.float()
    mass           = operator["mass"].float()
    eigenvalues    = operator["eigenvalues"].float()
    eigenvectors   = operator["eigenvectors"].float()
    gradient_x, gradient_y = (
        gradients
        if gradients is not None
        else DiffusionSurfaceEncoder.sparse_gradients(operator, len(features))
    )

    coefficients = eigenvectors.T @ (mass[:, None] * values)
    attenuation  = torch.exp(-eigenvalues[:, None] * block.diffusion_times.float()[None, :])
    diffused     = eigenvectors @ (attenuation * coefficients)
    grad_x       = block.sparse_multiply(gradient_x, values)
    grad_y       = block.sparse_multiply(gradient_y, values)
    mixed_x      = block.gradient_mixing(grad_x)
    mixed_y      = block.gradient_mixing(grad_y)
    invariant    = torch.tanh(grad_x * mixed_x + grad_y * mixed_y)
    return values + block.mlp(torch.cat((values, diffused, invariant), dim=1))


def optimized_diffusion_forward(
    block   : DiffusionBlock,
    features: Tensor,
    operator: Mapping[str, Tensor],
) -> Tensor:
    """Evaluate the experimental exact stacked path, including sparse construction cost."""
    point_count = len(features)
    index       = operator["gradient_index"]
    y_index     = torch.stack((index[0] + point_count, index[1]))
    gradient = torch.sparse_coo_tensor(
        torch.cat((index, y_index), dim=1),
        torch.cat((operator["gradient_x"], operator["gradient_y"])),
        (2 * point_count, point_count),
        check_invariants=False,
    ).coalesce()

    values         = features.float()
    mass           = operator["mass"].float()
    eigenvalues    = operator["eigenvalues"].float()
    eigenvectors   = operator["eigenvectors"].float()
    coefficients   = eigenvectors.T @ (mass[:, None] * values)
    attenuation    = torch.exp(-eigenvalues[:, None] * block.diffusion_times.float()[None, :])
    diffused       = eigenvectors @ (attenuation * coefficients)
    gradients      = block.sparse_multiply(gradient, values)
    mixed_gradients = block.gradient_mixing(gradients)
    grad_x, grad_y = gradients.split(point_count, dim=0)
    mixed_x, mixed_y = mixed_gradients.split(point_count, dim=0)
    invariant = torch.tanh(grad_x * mixed_x + grad_y * mixed_y)
    return values + block.mlp(torch.cat((values, diffused, invariant), dim=1))


def legacy_deltaconv_forward(
    encoder : DeltaConvSurfaceEncoder,
    features: Tensor,
    operator: Mapping[str, Tensor],
) -> Tensor:
    """Evaluate the former separate-axis DeltaConv gradient and divergence launches."""
    values = features.float()
    mass   = operator["mass"].float()
    gradient_x, gradient_y = DiffusionSurfaceEncoder.sparse_gradients(operator, len(features))

    vector_x = torch.zeros_like(values)
    vector_y = torch.zeros_like(values)
    for mixing, update in zip(
        encoder.vector_mixing,
        encoder.scalar_updates,
        strict=True,
    ):
        vector_x = vector_x + mixing(DiffusionBlock.sparse_multiply(gradient_x, values))
        vector_y = vector_y + mixing(DiffusionBlock.sparse_multiply(gradient_y, values))
        divergence = (
            DiffusionBlock.sparse_multiply(
                gradient_x.transpose(0, 1),
                mass[:, None] * vector_x,
            )
            + DiffusionBlock.sparse_multiply(
                gradient_y.transpose(0, 1),
                mass[:, None] * vector_y,
            )
        ) / mass[:, None]
        norm   = torch.sqrt(vector_x.square() + vector_y.square() + 1.0e-8)
        values = values + update(torch.cat((values, divergence, norm), dim=1))
    return values


def install_legacy_paths(model: WisdomV1) -> None:
    """Bind the preserved pre-refactor operations to one benchmark-only model copy."""

    def atomic_forward(
        encoder                : GatedAtomicEncoder,
        atom_features          : Tensor,
        edge_index             : Tensor,
        edge_is_spatial        : Tensor,
        edge_is_covalent       : Tensor,
        edge_distance          : Tensor,
        edge_bond_order        : Tensor,
        edge_same_residue      : Tensor,
        edge_same_chain        : Tensor,
        edge_residue_separation: Tensor,
        active_gates           : Mapping[str, Tensor],
    ) -> Tensor:
        """Adapt the legacy atomic reference to the model's unchanged call contract."""
        arguments = (
            atom_features,
            edge_index,
            edge_is_spatial,
            edge_is_covalent,
            edge_distance,
            edge_bond_order,
            edge_same_residue,
            edge_same_chain,
            edge_residue_separation,
        )
        return legacy_atomic_forward(encoder, arguments, active_gates)

    def surface_forward(
        transfer             : SurfaceAtomTransfer,
        atom_embeddings      : Tensor,
        neighbors            : Tensor,
        distances            : Tensor,
        normal_offsets       : Tensor,
        tangential_distances : Tensor,
        mask                 : Tensor,
        active_gates         : Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Adapt the legacy transfer reference to the model's unchanged call contract."""
        return legacy_surface_forward(
            transfer,
            atom_embeddings,
            neighbors,
            distances,
            normal_offsets,
            tangential_distances,
            mask,
            active_gates,
        )

    def diffusion_forward(
        encoder   : DiffusionSurfaceEncoder,
        features  : Tensor,
        operators : list[Mapping[str, Tensor]],
        surface_ptr: Tensor,
    ) -> Tensor:
        """Adapt the former separate-gradient implementation to the encoder call contract."""
        outputs = []
        for protein_index, operator in enumerate(operators):
            start     = int(surface_ptr[protein_index])
            stop      = int(surface_ptr[protein_index + 1])
            local     = features[start:stop]
            gradients = DiffusionSurfaceEncoder.sparse_gradients(operator, stop - start)
            for block in encoder.blocks:
                local = legacy_diffusion_forward(block, local, operator, gradients)
            outputs.append(local)
        return torch.cat(outputs)

    model.atomic_encoder.forward = types.MethodType(  # type: ignore[method-assign]
        atomic_forward,
        model.atomic_encoder,
    )
    model.surface_atom_transfer.forward = types.MethodType(  # type: ignore[method-assign]
        surface_forward,
        model.surface_atom_transfer,
    )
    model.surface_encoder.forward = types.MethodType(  # type: ignore[method-assign]
        diffusion_forward,
        model.surface_encoder,
    )


def model_batch(
    hidden_dim: int,
    device    : torch.device,
) -> dict[str, Any]:
    """Create a four-protein medium batch with realistic sparse topology and operator sizes."""
    protein_count       = 4
    atoms_per_protein   = 256
    points_per_protein  = 1_024
    neighbors_per_point = 16
    modes               = 64
    generator           = torch.Generator().manual_seed(2029)

    local_atomic = atomic_inputs(atoms_per_protein, 16, hidden_dim, device)
    local_edges  = local_atomic[1]
    edges = torch.cat(
        tuple(local_edges + protein * atoms_per_protein for protein in range(protein_count)),
        dim=1,
    )
    spatial, covalent, distance, bond, residue, chain, separation = (
        value.repeat(protein_count) for value in local_atomic[2:]
    )
    atom_count    = protein_count * atoms_per_protein
    surface_count = protein_count * points_per_protein

    surface_batch = torch.arange(protein_count).repeat_interleave(points_per_protein).to(device)
    surface_ptr   = torch.arange(0, surface_count + 1, points_per_protein)
    atom_batch    = torch.arange(protein_count).repeat_interleave(atoms_per_protein)
    local_neighbors = torch.randint(
        atoms_per_protein,
        (surface_count, neighbors_per_point),
        generator=generator,
    )
    local_neighbors += atom_batch.repeat_interleave(points_per_protein // atoms_per_protein)[
        :, None
    ] * atoms_per_protein
    surface_distances = torch.rand(
        surface_count,
        neighbors_per_point,
        generator=generator,
    ) * 5.5
    surface_offsets = (
        torch.rand(surface_count, neighbors_per_point, generator=generator) - 0.5
    ) * surface_distances
    surface_tangent = torch.sqrt(
        (surface_distances.square() - surface_offsets.square()).clamp_min(0.0)
    )

    operators = []
    for _ in range(protein_count):
        row = torch.arange(points_per_protein).repeat_interleave(8)
        offset = torch.arange(8).repeat(points_per_protein)
        column = (row + offset - 4).remainder(points_per_protein)
        index  = torch.stack((row, column)).to(device)
        gradient = torch.randn(index.shape[1], generator=generator) * 0.1
        eigenvectors = (
            torch.randn(points_per_protein, modes, generator=generator)
            / points_per_protein**0.5
        )
        operators.append(
            {
                "mass": torch.ones(points_per_protein, device=device),
                "eigenvalues": torch.linspace(0.0, 10.0, modes, device=device),
                "eigenvectors": eigenvectors.to(device),
                "gradient_index": index,
                "gradient_x": gradient.to(device),
                "gradient_y": gradient.roll(3).to(device),
            }
        )

    atomic_numbers = torch.randint(1, 20, (atom_count,), generator=generator).to(device)
    return {
        "atomic_numbers": atomic_numbers,
        "residue_type_ids": torch.randint(0, 21, (atom_count,), generator=generator).to(device),
        "atom_role_ids": torch.zeros(atom_count, dtype=torch.long, device=device),
        "atom_hybridization_ids": torch.zeros(atom_count, dtype=torch.long, device=device),
        "formal_charges": torch.zeros(atom_count, device=device),
        "atom_aromaticity": torch.zeros(atom_count, device=device),
        "atom_hbond_donor": torch.zeros(atom_count, device=device),
        "atom_hbond_acceptor": torch.zeros(atom_count, device=device),
        "residue_hydropathy": torch.zeros(atom_count, device=device),
        "residue_polarity": torch.zeros(atom_count, device=device),
        "atom_edge_index": edges,
        "atom_edge_is_spatial": spatial,
        "atom_edge_is_covalent": covalent,
        "atom_edge_distance": distance,
        "atom_edge_bond_order": bond,
        "atom_edge_same_residue": residue,
        "atom_edge_same_chain": chain,
        "atom_edge_residue_separation": separation,
        "surface_curvatures": torch.randn(
            surface_count, 2, 3, generator=generator
        ).to(device),
        "surface_atom_neighbors": local_neighbors.to(device),
        "surface_atom_distances": surface_distances.to(device),
        "surface_atom_normal_offsets": surface_offsets.to(device),
        "surface_atom_tangential_distances": surface_tangent.to(device),
        "surface_atom_mask": torch.ones(
            surface_count,
            neighbors_per_point,
            dtype=torch.bool,
            device=device,
        ),
        "surface_area_weights": torch.ones(surface_count, device=device),
        "surface_batch": surface_batch,
        "surface_operators": operators,
        "surface_ptr": surface_ptr,
    }


def atomic_suite(
    device    : torch.device,
    iterations: int,
) -> list[dict[str, Any]]:
    """Benchmark atomic forward and forward/backward for three realistic widths and sizes."""
    sizes = {
        "small":  (256, 12),
        "medium": (768, 20),
        "large":  (1_536, 28),
    }
    rows: list[dict[str, Any]] = []
    for size, (atom_count, degree) in sizes.items():
        for hidden_dim in (64, 128, 256):
            arguments = atomic_inputs(atom_count, degree, 48, device)
            model = GatedAtomicEncoder(
                48,
                hidden_dim,
                layers=2,
                dropout=0.0,
                distance_scale=6.0,
                message_chunk_size=65_536,
            ).to(device)
            edge_count = arguments[1].shape[1]

            for backward in (False, True):
                before = functools.partial(legacy_atomic_forward, model, arguments)
                after  = functools.partial(optimized_atomic_forward, model, arguments)
                for implementation, operation in (
                    ("before", before),
                    ("after", after),
                ):
                    model.zero_grad(set_to_none=True)
                    result = measure(operation, device, iterations, backward)
                    rows.append(
                        {
                            "component": (
                                "atomic_forward_backward" if backward else "atomic_forward"
                            ),
                            "implementation": implementation,
                            "size": size,
                            "hidden_dim": hidden_dim,
                            "atoms": atom_count,
                            "undirected_edges": edge_count,
                            "directed_degree": 2.0 * edge_count / atom_count,
                            **result,
                        }
                    )
    return rows


def edge_storage_suite(device: torch.device) -> list[dict[str, Any]]:
    """Report exact compact versus duplicated edge bytes transferred per representative graph."""
    rows = []
    for size, atom_count, degree in (
        ("small", 256, 12),
        ("medium", 768, 20),
        ("large", 1_536, 28),
    ):
        arguments = atomic_inputs(atom_count, degree, 48, device)
        compact_bytes = sum(value.numel() * value.element_size() for value in arguments[1:])
        rows.append(
            {
                "component": "atomic_edge_storage",
                "size": size,
                "atoms": atom_count,
                "undirected_edges": arguments[1].shape[1],
                "before_bytes": 2 * compact_bytes,
                "after_bytes": compact_bytes,
            }
        )
    return rows


def surface_suite(device: torch.device, iterations: int) -> list[dict[str, Any]]:
    """Benchmark the exact atom-to-surface contraction for all HPO widths."""
    generator = torch.Generator().manual_seed(2027)
    rows      : list[dict[str, Any]] = []
    for hidden_dim in (64, 128, 256):
        atom_count    = 1_024
        surface_count = 4_096
        width         = 16
        embeddings = torch.randn(atom_count, hidden_dim, generator=generator).to(device)
        neighbors  = torch.randint(
            atom_count, (surface_count, width), generator=generator
        ).to(device)
        distances  = (torch.rand(surface_count, width, generator=generator) * 5.5).to(device)
        random_offsets = torch.rand(surface_count, width, generator=generator).to(device)
        offsets        = (random_offsets - 0.5) * distances
        tangential = torch.sqrt((distances.square() - offsets.square()).clamp_min(0.0))
        mask       = (torch.rand(surface_count, width, generator=generator) > 0.1).to(device)
        transfer   = SurfaceAtomTransfer(hidden_dim, chunk_size=8_192).to(device)
        arguments  = (embeddings, neighbors, distances, offsets, tangential, mask)

        before = functools.partial(legacy_surface_forward, transfer, *arguments)
        after  = functools.partial(transfer, *arguments)
        for backward in (False, True):
            for implementation, operation in (("before", before), ("after", after)):
                transfer.zero_grad(set_to_none=True)
                result = measure(operation, device, iterations, backward)
                rows.append(
                    {
                        "component": (
                            "surface_transfer_forward_backward"
                            if backward
                            else "surface_transfer_forward"
                        ),
                        "implementation": implementation,
                        "size": "medium",
                        "hidden_dim": hidden_dim,
                        "atoms": atom_count,
                        "surface_points": surface_count,
                        "neighbors_per_point": width,
                        **result,
                    }
                )
    return rows


def diffusion_suite(device: torch.device, iterations: int) -> list[dict[str, Any]]:
    """Benchmark separate versus stacked tangent operators including construction cost."""
    generator = torch.Generator().manual_seed(2028)
    rows      : list[dict[str, Any]] = []
    point_count = 2_048
    modes       = 128
    stencil     = 8
    row = torch.arange(point_count).repeat_interleave(stencil)
    offsets = torch.arange(stencil).repeat(point_count)
    column  = (row + offsets - stencil // 2).remainder(point_count)
    index   = torch.stack((row, column)).to(device)
    values  = torch.randn(index.shape[1], generator=generator) * 0.1
    eigenvectors = torch.randn(point_count, modes, generator=generator) / point_count**0.5
    operator = {
        "mass": torch.ones(point_count, device=device),
        "eigenvalues": torch.linspace(0.0, 10.0, modes, device=device),
        "eigenvectors": eigenvectors.to(device),
        "gradient_index": index,
        "gradient_x": values.to(device),
        "gradient_y": values.roll(3).to(device),
    }
    for hidden_dim in (64, 128, 256):
        features = torch.randn(point_count, hidden_dim, generator=generator).to(device)
        block    = DiffusionBlock(hidden_dim, dropout=0.0).to(device)
        before   = functools.partial(legacy_diffusion_forward, block, features, operator)
        after    = functools.partial(optimized_diffusion_forward, block, features, operator)
        for backward in (False, True):
            for implementation, operation in (("before", before), ("after", after)):
                block.zero_grad(set_to_none=True)
                result = measure(operation, device, iterations, backward)
                rows.append(
                    {
                        "component": (
                            "diffusion_forward_backward" if backward else "diffusion_forward"
                        ),
                        "implementation": implementation,
                        "size": "medium",
                        "hidden_dim": hidden_dim,
                        "surface_points": point_count,
                        "spectral_modes": modes,
                        "gradient_entries": index.shape[1],
                        **result,
                    }
                )
    return rows


def deltaconv_suite(device: torch.device, iterations: int) -> list[dict[str, Any]]:
    """Benchmark exact separate versus stacked DeltaConv differential operators."""
    generator   = torch.Generator().manual_seed(2030)
    rows       : list[dict[str, Any]] = []
    point_count = 2_048
    stencil     = 8
    row         = torch.arange(point_count).repeat_interleave(stencil)
    offsets     = torch.arange(stencil).repeat(point_count)
    column      = (row + offsets - stencil // 2).remainder(point_count)
    index       = torch.stack((row, column)).to(device)
    gradient    = torch.randn(index.shape[1], generator=generator) * 0.1
    operator = {
        "mass": torch.ones(point_count, device=device),
        "gradient_index": index,
        "gradient_x": gradient.to(device),
        "gradient_y": gradient.roll(3).to(device),
    }

    for hidden_dim in (64, 128, 256):
        features  = torch.randn(point_count, hidden_dim, generator=generator).to(device)
        optimized = DeltaConvSurfaceEncoder(hidden_dim, layers=2, dropout=0.0).to(device)
        reference = copy.deepcopy(optimized)
        pointer   = torch.tensor([0, point_count])

        before = functools.partial(legacy_deltaconv_forward, reference, features, operator)
        after  = functools.partial(optimized, features, [operator], pointer)
        for backward in (False, True):
            for implementation, operation in (("before", before), ("after", after)):
                result = measure(operation, device, iterations, backward)
                rows.append(
                    {
                        "component": (
                            "deltaconv_forward_backward" if backward else "deltaconv_forward"
                        ),
                        "implementation": implementation,
                        "size": "medium",
                        "hidden_dim": hidden_dim,
                        "surface_points": point_count,
                        "gradient_entries": index.shape[1],
                        **result,
                    }
                )
    return rows


def model_suite(device: torch.device, iterations: int) -> list[dict[str, Any]]:
    """Benchmark complete V1 inference and optimizer steps with shared initial parameters."""
    rows: list[dict[str, Any]] = []
    for hidden_dim in (64, 128, 256):
        batch = model_batch(hidden_dim, device)
        optimized = WisdomV1(
            hidden_dim=hidden_dim,
            embedding_dim=16,
            residue_embedding_dim=16,
            atomic_layers=2,
            projection_depth=1,
            surface_layers=2,
            dropout=0.0,
            curvature_features=6,
            atom_spatial_k=16,
            surface_atom_k=16,
            diffusion_spectral_modes=64,
        ).to(device)
        legacy = copy.deepcopy(optimized)
        install_legacy_paths(legacy)

        # Deterministic evaluation proves that the complete composition stays locally equivalent.

        optimized.eval()
        legacy.eval()
        with torch.inference_mode():
            expected = legacy(**batch)
            actual   = optimized(**batch)
        maximum_error = float((expected["logits"] - actual["logits"]).abs().max())

        for implementation, model in (("before", legacy), ("after", optimized)):
            model.eval()
            forward = functools.partial(model, **batch)
            result  = measure(forward, device, max(2, iterations // 2), backward=False)
            rows.append(
                {
                    "component": "v1_forward",
                    "implementation": implementation,
                    "size": "medium",
                    "hidden_dim": hidden_dim,
                    "proteins": 4,
                    "proteins_per_second": 4_000.0 / result["milliseconds"],
                    "maximum_logit_error": maximum_error,
                    **result,
                }
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
            targets   = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
            model.train()

            def train_step(
                current_model: WisdomV1 = model,
                current_optimizer: torch.optim.Optimizer = optimizer,
                current_batch: Mapping[str, Any] = batch,
                current_targets: Tensor = targets,
            ) -> Tensor:
                """Execute one complete BCE plus Hard-Concrete regularization optimizer step."""
                current_optimizer.zero_grad(set_to_none=True)
                output = current_model(**current_batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["logits"], current_targets
                ) + 0.001 * current_model.gate_regularization()
                loss.backward()
                current_optimizer.step()
                return loss.detach()

            result = measure(train_step, device, max(2, iterations // 2), backward=False)
            rows.append(
                {
                    "component": "v1_train_step",
                    "implementation": implementation,
                    "size": "medium",
                    "hidden_dim": hidden_dim,
                    "proteins": 4,
                    "proteins_per_second": 4_000.0 / result["milliseconds"],
                    **result,
                }
            )
    return rows


def profile_atomic(device: torch.device, output: Path) -> None:
    """Capture one medium H=128 before/after operator profile sorted by device time."""
    arguments = atomic_inputs(768, 20, 48, device)
    model     = GatedAtomicEncoder(48, 128, 2, 0.0, 6.0, 65_536).to(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    tables = []
    for name, operation in (
        ("before", lambda: legacy_atomic_forward(model, arguments)),
        ("after", lambda: optimized_atomic_forward(model, arguments)),
    ):
        with torch.profiler.profile(
            activities=activities,
            profile_memory=True,
            record_shapes=True,
        ) as profiler:
            result = operation()
            result.square().mean().backward()
            synchronize(device)
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        table = profiler.key_averages().table(sort_by=sort_key, row_limit=20)
        tables.append(f"## {name}\n\n```text\n{table}\n```\n")
    output.write_text("\n".join(tables), encoding="utf-8")


def main() -> None:
    """Parse benchmark controls and persist machine-readable measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runtime-benchmark.json"))
    parser.add_argument("--profile", type=Path)
    arguments = parser.parse_args()

    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")
    device = torch.device(arguments.device)
    rows = [
        *atomic_suite(device, arguments.iterations),
        *surface_suite(device, arguments.iterations),
        *diffusion_suite(device, arguments.iterations),
        *deltaconv_suite(device, arguments.iterations),
        *model_suite(device, arguments.iterations),
        *edge_storage_suite(device),
    ]
    report = {
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "measurements": rows,
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if arguments.profile is not None:
        profile_atomic(device, arguments.profile)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
