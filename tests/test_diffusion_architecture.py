from __future__ import annotations

import numpy as np
import pytest
import torch
from test_wisdom_v1 import _model_inputs, _sample

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.models.DiffusionBlock import DiffusionBlock
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.SurfaceAtomTransfer import SurfaceAtomTransfer
from wisdom.models.WisdomV3 import WisdomV3
from wisdom.preprocessing.structure.DiffusionOperatorBuilder import DiffusionOperatorBuilder
from wisdom.preprocessing.structure.SurfaceAtomNeighborhoodBuilder import (
    SurfaceAtomNeighborhoodBuilder,
)


def _point_cloud() -> tuple[np.ndarray, np.ndarray]:
    """Return a non-symmetric oriented point cloud with non-degenerate low modes."""
    generator = np.random.default_rng(41)
    directions = generator.normal(size=(40, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = 3.0 + 0.35 * directions[:, 0] + 0.15 * directions[:, 1] * directions[:, 2]
    return directions * radii[:, None], directions


def _operator_tensors(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert one persisted operator mapping to the trainable tensor contract."""
    return {
        "mass": torch.from_numpy(arrays["diffusion_mass"]),
        "eigenvalues": torch.from_numpy(arrays["diffusion_eigenvalues"]),
        "eigenvectors": torch.from_numpy(arrays["diffusion_eigenvectors"]),
        "gradient_index": torch.from_numpy(arrays["diffusion_gradient_index"]).long(),
        "gradient_x": torch.from_numpy(arrays["diffusion_gradient_x"]),
        "gradient_y": torch.from_numpy(arrays["diffusion_gradient_y"]),
    }


def test_spectral_diffusion_preserves_constants_and_smooths_high_modes() -> None:
    """Heat diffusion retains constants and attenuates high frequencies more strongly."""
    points, normals = _point_cloud()
    arrays   = DiffusionOperatorBuilder(1.0, spectral_modes=24, max_neighbors=10).build(
        points,
        normals,
    )
    operator = _operator_tensors(arrays)
    pointer  = torch.tensor([0, len(points)])

    constant = torch.ones(len(points))
    preserved = DiffusionSurfaceEncoder.diffuse_scalar(
        constant,
        [operator],
        pointer,
        time=3.0,
    )
    assert torch.allclose(preserved, constant, rtol=2.0e-3, atol=2.0e-3)

    phi      = operator["eigenvectors"]
    high     = phi[:, -1]
    near_zero = DiffusionSurfaceEncoder.diffuse_scalar(high, [operator], pointer, time=1.0e-8)
    smoothed  = DiffusionSurfaceEncoder.diffuse_scalar(high, [operator], pointer, time=4.0)

    assert torch.allclose(near_zero, high, rtol=2.0e-4, atol=2.0e-4)
    assert torch.linalg.vector_norm(smoothed) < torch.linalg.vector_norm(high)


def test_sparse_gradients_annihilate_a_constant_field() -> None:
    """Both fitted tangent derivatives have zero row sum within numerical tolerance."""
    points, normals = _point_cloud()
    arrays   = DiffusionOperatorBuilder(1.0, spectral_modes=16, max_neighbors=10).build(
        points,
        normals,
    )
    operator = _operator_tensors(arrays)
    gradient_x, gradient_y = DiffusionSurfaceEncoder.sparse_gradients(operator, len(points))
    constant = torch.ones(len(points), 1)

    assert torch.allclose(
        torch.sparse.mm(gradient_x, constant),
        torch.zeros_like(constant),
        atol=2e-5,
    )
    assert torch.allclose(
        torch.sparse.mm(gradient_y, constant),
        torch.zeros_like(constant),
        atol=2e-5,
    )


def test_diffusion_block_times_are_positive_and_receive_gradients() -> None:
    """Softplus-constrained physical times and every learned block path support autograd."""
    points, normals = _point_cloud()
    arrays   = DiffusionOperatorBuilder(1.0, spectral_modes=16, max_neighbors=10).build(
        points,
        normals,
    )
    operator = _operator_tensors(arrays)
    gradient_x, gradient_y = DiffusionSurfaceEncoder.sparse_gradients(operator, len(points))
    features = torch.randn(len(points), 6, requires_grad=True)
    block    = DiffusionBlock(6, initial_time=1.0)

    output = block(
        features,
        operator["mass"],
        operator["eigenvalues"],
        operator["eigenvectors"],
        gradient_x,
        gradient_y,
    )
    output.square().mean().backward()

    assert torch.all(block.diffusion_times > 0.0)
    assert block.raw_times.grad is not None
    assert torch.isfinite(block.raw_times.grad).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_sparse_diffusion_disables_bfloat16_autocast_only_for_sparse_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA-unsupported sparse operation stays FP32 inside a mixed-precision forward."""
    index    = torch.tensor([[0, 1], [0, 1]])
    operator = torch.sparse_coo_tensor(
        index,
        torch.ones(2),
        (2, 2),
        check_invariants=False,
    ).coalesce()
    values   = torch.randn(2, 3, requires_grad=True)

    autocast_states: list[bool] = []
    original_sparse_mm          = torch.sparse.mm

    def observed_sparse_mm(matrix: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        """Record the CPU autocast state and operand dtypes at the protected operation."""
        autocast_states.append(torch.is_autocast_enabled("cpu"))
        assert matrix.dtype == torch.float32
        assert dense.dtype == torch.float32
        return original_sparse_mm(matrix, dense)

    monkeypatch.setattr(torch.sparse, "mm", observed_sparse_mm)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = DiffusionBlock.sparse_multiply(operator, values)

    result.sum().backward()

    assert autocast_states == [False]
    assert result.dtype == torch.float32
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_diffusion_result_is_rigid_motion_invariant_after_operator_recomputation() -> None:
    """Intrinsic heat/gradient features agree after rotating and translating the point cloud."""
    points, normals = _point_cloud()
    angle = 0.71
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed_points  = points @ rotation.T + np.asarray([7.0, -4.0, 2.0])
    transformed_normals = normals @ rotation.T

    builder = DiffusionOperatorBuilder(1.0, spectral_modes=20, max_neighbors=10)
    original    = _operator_tensors(builder.build(points, normals))
    transformed = _operator_tensors(builder.build(transformed_points, transformed_normals))
    features = torch.from_numpy(
        (np.square(normals[:, :1]) + 0.3).astype(np.float32)
    ).repeat(1, 4)
    encoder     = DiffusionSurfaceEncoder(4, layers=1, dropout=0.0).eval()
    pointer     = torch.tensor([0, len(points)])

    with torch.no_grad():
        first  = encoder(features, [original], pointer)
        second = encoder(features, [transformed], pointer)

    assert torch.allclose(first, second, rtol=2.0e-3, atol=2.0e-3)


def test_diffusion_result_follows_surface_point_permutations() -> None:
    """Reindexing every operator consistently only reindexes the encoded surface output."""
    points, normals = _point_cloud()
    arrays   = DiffusionOperatorBuilder(1.0, spectral_modes=20, max_neighbors=10).build(
        points,
        normals,
    )
    operator = _operator_tensors(arrays)
    features = torch.randn(len(points), 4)
    encoder  = DiffusionSurfaceEncoder(4, layers=1, dropout=0.0).eval()
    pointer  = torch.tensor([0, len(points)])

    permutation = torch.randperm(len(points), generator=torch.Generator().manual_seed(19))
    inverse     = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(points))
    permuted_operator = {
        "mass": operator["mass"][permutation],
        "eigenvalues": operator["eigenvalues"],
        "eigenvectors": operator["eigenvectors"][permutation],
        "gradient_index": inverse[operator["gradient_index"]],
        "gradient_x": operator["gradient_x"],
        "gradient_y": operator["gradient_y"],
    }

    with torch.no_grad():
        original = encoder(features, [operator], pointer)
        permuted = encoder(features[permutation], [permuted_operator], pointer)

    assert torch.allclose(permuted, original[permutation], rtol=1.0e-5, atol=1.0e-5)


def test_surface_atom_transfer_is_rigid_invariant_and_chunk_equivalent() -> None:
    """Precomputed local geometry and chunking do not change learned atom transfer."""
    atoms = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [0.5, 2.0, 0.0]])
    points = np.asarray([[0.5, 0.5, 1.0], [1.0, 1.0, 1.5], [0.2, 0.3, 1.2]])
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (3, 1))
    angle = 0.43
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    builder = SurfaceAtomNeighborhoodBuilder(radius=6.0, max_neighbors=3)
    first  = builder.build(atoms, points, normals)
    second = builder.build(
        atoms @ rotation.T + 3.0,
        points @ rotation.T + 3.0,
        normals @ rotation.T,
    )
    embeddings = torch.randn(3, 5)
    chunked     = SurfaceAtomTransfer(5, chunk_size=1).eval()
    complete    = SurfaceAtomTransfer(5, chunk_size=100).eval()
    complete.load_state_dict(chunked.state_dict())

    def transfer(module: SurfaceAtomTransfer, arrays: dict[str, np.ndarray]) -> torch.Tensor:
        """Evaluate one transfer fixture from persisted geometry arrays."""
        return module(
            embeddings,
            torch.from_numpy(arrays["surface_atom_neighbors"]).long(),
            torch.from_numpy(arrays["surface_atom_distances"]),
            torch.from_numpy(arrays["surface_atom_normal_offsets"]),
            torch.from_numpy(arrays["surface_atom_tangential_distances"]),
            torch.from_numpy(arrays["surface_atom_mask"]),
        )

    expected = transfer(complete, first)
    assert torch.allclose(transfer(chunked, first), expected, atol=1.0e-6)
    assert torch.allclose(transfer(chunked, second), expected, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize(
    "encoder_type",
    ["diffusion", "dmasif", "deltaconv", "ptv3", "point_mamba"],
)
def test_every_v3_surface_encoder_supports_forward_and_backward(encoder_type: str) -> None:
    """The fair v3 alternatives preserve point count, protein boundaries, and gradients."""
    batch = dict(WisdomCollator()((_sample(3, 5, 1.0), _sample(2, 4, 0.0))))
    model = WisdomV3(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=1,
        surface_layers=1,
        dropout=0.0,
        curvature_features=6,
        surface_encoder_type=encoder_type,
        surface_patch_size=4,
    )
    inputs = _model_inputs(batch)
    inputs.update(
        {
            "surface_positions": batch["surface_positions"],
            "surface_normals": batch["surface_normals"],
            "surface_neighbors": batch["surface_neighbors"],
            "surface_neighbor_mask": batch["surface_neighbor_mask"],
        }
    )

    output = model(**inputs)
    loss   = torch.nn.functional.binary_cross_entropy_with_logits(output["logits"], batch["target"])
    loss.backward()

    assert output["surface_logits"].shape == (9,)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["surface_logits"]).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.surface_encoder.parameters()
    )
