from __future__ import annotations

import copy

import torch
from torch import Tensor

from wisdom.models.GatedAtomicEncoder import GatedAtomicEncoder
from wisdom.models.SurfaceAtomTransfer import SurfaceAtomTransfer


def _atomic_reference(
    encoder  : GatedAtomicEncoder,
    features : Tensor,
    edges    : Tensor,
    spatial  : Tensor,
    covalent : Tensor,
    distance : Tensor,
    bond     : Tensor,
    residue  : Tensor,
    chain    : Tensor,
    separation: Tensor,
    gates    : dict[str, Tensor],
) -> Tensor:
    """Evaluate the former directed per-edge matrix projection exactly."""
    hidden = encoder.input_projection(features)
    directed_edges = torch.cat((edges, edges.flip(0)), dim=1)
    source, target = directed_edges

    spatial_mask  = torch.cat((spatial, spatial))
    covalent_mask = torch.cat((covalent, covalent))
    distance      = torch.cat((distance, distance))
    bond          = torch.cat((bond, bond))
    residue       = torch.cat((residue, residue))
    chain         = torch.cat((chain, chain))
    separation    = torch.cat((separation, separation))

    spatial_attributes = torch.stack(
        (
            (distance / encoder.distance_scale).clamp(max=2.0)
            * gates["edge.spatial.distance"],
            residue.float() * gates["edge.spatial.same_residue"],
            chain.float() * gates["edge.spatial.same_chain"],
            (separation.float() / encoder.residue_scale).clamp(max=2.0)
            * gates["edge.spatial.residue_separation"],
        ),
        dim=1,
    )
    covalent_attributes = torch.stack(
        (
            (distance / encoder.distance_scale).clamp(max=2.0)
            * gates["edge.covalent.distance"],
            bond.float().clamp(min=0.0, max=3.0) / 3.0
            * gates["edge.covalent.bond_order"],
            residue.float() * gates["edge.covalent.same_residue"],
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
        deltas = []
        for mask, message, score, attributes in (
            (spatial_mask, spatial_message, spatial_score, spatial_attributes),
            (covalent_mask, covalent_message, covalent_score, covalent_attributes),
        ):
            selected_source     = source[mask]
            selected_target     = target[mask]
            selected_attributes = attributes[mask]
            scale = score(selected_attributes) - score(torch.zeros_like(selected_attributes))
            messages = message(hidden[selected_source]) * (1.0 + scale)
            aggregate = torch.zeros_like(hidden).index_add(0, selected_target, messages)
            degree = torch.bincount(selected_target, minlength=len(hidden)).to(hidden.dtype)
            deltas.append(aggregate / degree.clamp_min(1.0).unsqueeze(1))

        hidden = norm(
            hidden
            + gates["atomic_graph.spatial"] * deltas[0]
            + gates["atomic_graph.covalent"] * deltas[1]
        )
        hidden = torch.nn.functional.silu(hidden)
    return hidden


def test_factorized_atomic_messages_preserve_outputs_and_all_gradients() -> None:
    """Compact factorization agrees with directed per-edge projections and parameter gradients."""
    torch.manual_seed(17)
    reference = GatedAtomicEncoder(9, 12, 2, 0.0, 6.0, 3).eval()
    optimized = copy.deepcopy(reference)

    features = torch.randn(7, 9)
    edges = torch.tensor(
        [[0, 0, 1, 1, 2, 3, 4, 5], [1, 2, 2, 4, 3, 5, 6, 6]],
        dtype=torch.long,
    )
    spatial   = torch.tensor([True, True, True, False, True, True, True, True])
    covalent  = torch.tensor([True, False, True, True, False, True, False, True])
    distance  = torch.linspace(1.0, 6.0, edges.shape[1])
    bond      = covalent.float()
    residue   = torch.tensor([True, True, False, True, False, False, True, True])
    chain     = torch.tensor([True, True, True, True, True, False, False, False])
    separation = torch.arange(edges.shape[1]).float()
    gates = {
        "atomic_graph.spatial": edges.new_tensor(0.8, dtype=torch.float32),
        "atomic_graph.covalent": edges.new_tensor(0.6, dtype=torch.float32),
        "edge.spatial.distance": edges.new_tensor(0.7, dtype=torch.float32),
        "edge.spatial.same_residue": edges.new_tensor(0.9, dtype=torch.float32),
        "edge.spatial.same_chain": edges.new_tensor(0.5, dtype=torch.float32),
        "edge.spatial.residue_separation": edges.new_tensor(0.4, dtype=torch.float32),
        "edge.covalent.distance": edges.new_tensor(0.3, dtype=torch.float32),
        "edge.covalent.bond_order": edges.new_tensor(0.8, dtype=torch.float32),
        "edge.covalent.same_residue": edges.new_tensor(0.6, dtype=torch.float32),
    }

    reference_features = features.clone().requires_grad_(True)
    optimized_features = features.clone().requires_grad_(True)
    expected = _atomic_reference(
        reference,
        reference_features,
        edges,
        spatial,
        covalent,
        distance,
        bond,
        residue,
        chain,
        separation,
        gates,
    )
    actual = optimized(
        optimized_features,
        edges,
        spatial,
        covalent,
        distance,
        bond,
        residue,
        chain,
        separation,
        gates,
    )

    expected.square().mean().backward()
    actual.square().mean().backward()

    assert torch.allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)
    assert torch.allclose(
        optimized_features.grad,
        reference_features.grad,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    for expected_parameter, actual_parameter in zip(
        reference.parameters(), optimized.parameters(), strict=True
    ):
        assert expected_parameter.grad is not None
        assert actual_parameter.grad is not None
        assert torch.allclose(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=3.0e-5,
            atol=3.0e-6,
        )

    # Identical AdamW states must therefore produce the same one-step parameters. This checks the
    # complete optimization boundary rather than treating gradient agreement as sufficient.

    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    optimized_optimizer = torch.optim.AdamW(optimized.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    reference_optimizer.step()
    optimized_optimizer.step()
    for expected_parameter, actual_parameter in zip(
        reference.parameters(), optimized.parameters(), strict=True
    ):
        assert torch.allclose(
            actual_parameter,
            expected_parameter,
            rtol=3.0e-5,
            atol=3.0e-6,
        )


def test_batched_surface_transfer_preserves_weighted_sum_gradients() -> None:
    """The allocation-free batched contraction matches the explicit weighted tensor sum."""
    torch.manual_seed(23)
    transfer = SurfaceAtomTransfer(7, radius=6.0, chunk_size=64).eval()
    atoms     = torch.randn(11, 7, requires_grad=True)
    neighbors = torch.randint(0, 11, (13, 5))
    distances = torch.rand(13, 5) * 5.5
    offsets   = (torch.rand(13, 5) - 0.5) * distances
    tangent   = torch.sqrt((distances.square() - offsets.square()).clamp_min(0.0))
    mask      = torch.rand(13, 5) > 0.15

    output = transfer(atoms, neighbors, distances, offsets, tangent, mask)

    normalized = (distances / transfer.radius).unsqueeze(-1)
    orientation = torch.stack(
        (distances / transfer.radius, offsets / transfer.radius, tangent / transfer.radius),
        dim=-1,
    )
    gathered = atoms[neighbors]
    base = transfer.distance_scorer(normalized).squeeze(-1)
    orientation_score = (
        transfer.orientation_scorer(orientation)
        - transfer.orientation_scorer(torch.zeros_like(orientation))
    ).squeeze(-1)
    content = torch.cat((normalized, gathered), dim=-1)
    content_score = (
        transfer.content_scorer(content)
        - transfer.content_scorer(torch.cat((normalized, torch.zeros_like(gathered)), dim=-1))
    ).squeeze(-1)
    scores = (base + orientation_score + content_score).masked_fill(
        ~mask,
        torch.finfo(base.dtype).min,
    )
    weights = torch.softmax(scores, dim=1)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )
    expected = torch.sum(weights.unsqueeze(-1) * gathered, dim=1)

    output.square().mean().backward(retain_graph=True)
    actual_gradient = atoms.grad.detach().clone()
    atoms.grad       = None
    expected.square().mean().backward()

    assert torch.allclose(output, expected, rtol=1.0e-6, atol=1.0e-6)
    assert torch.allclose(actual_gradient, atoms.grad, rtol=1.0e-5, atol=1.0e-6)
