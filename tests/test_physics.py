"""VF numerical contracts and native Training integration; fixtures are not scientific datasets."""

import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.cache import WorkCache
from lambdaforge.work.checkpoints import CheckpointCollection
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from test_structural_enrichment import _complete_base, _work
from test_wisdom_v1 import _sample

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.evaluation.PhysicalMetricSuite import PhysicalMetricSuite
from wisdom.models.physics.FieldDecoder import FieldDecoder
from wisdom.models.physics.PhysicalObjective import PhysicalObjective
from wisdom.models.physics.PhysicalTraining import PhysicalTraining
from wisdom.models.surface.DiffusionBlock import DiffusionBlock
from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.WisdomV2 import WisdomV2
from wisdom.preprocessing.common.StructuralEnrichment import StructuralEnrichment
from wisdom.preprocessing.common.structure.assets.physics.FieldQueries import FieldQueries
from wisdom.preprocessing.common.structure.assets.physics.MolecularPreparation import (
    MolecularPreparation,
)
from wisdom.preprocessing.common.structure.assets.physics.PhysicalFields import PhysicalFields
from wisdom.preprocessing.common.structure.assets.physics.PhysicalReferenceUnavailable import (
    PhysicalReferenceUnavailable,
)
from wisdom.preprocessing.common.structure.assets.physics.PhysicsConfig import PhysicsConfig
from wisdom.preprocessing.common.structure.assets.physics.ProbeInteraction import ProbeInteraction
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars
from wisdom.Training import _create_model, _device_batch, _model_inputs


def test_lj_units_masks_and_openmm_reference():
    """A true mixed-pair minimum is -epsilon; masked overlaps are unavailable, not giant labels."""
    pytest.importorskip("openmm")
    interaction = ProbeInteraction(np.zeros((1, 3)), np.array([0.3]), np.array([0.4]))
    radius = 0.35 * 2 ** (1 / 6) * 10  # Å; mixed sigma=.35 nm, mixed epsilon=.6 kJ/mol.
    queries = np.array([[radius, 0, 0], [0, 0, 0], [20, 0, 0]])
    values, valid = interaction.evaluate(queries, (0.4, 0.9))
    assert values[0] == pytest.approx(-0.6)
    assert valid.tolist() == [True, False, True]
    assert values[1] == 0
    np.testing.assert_allclose(
        interaction.reference(queries[valid], (0.4, 0.9)), values[valid], rtol=1e-12, atol=1e-12
    )
    short, _ = interaction.evaluate(queries[valid], (0.4, 0.9), cutoff=10)
    assert short[-1] == 0 and values[-1] < 0


def test_blocked_full_lj_matches_scalar_reduction_exactly():
    """Vectorizing bounded query blocks changes neither pair order, overlaps nor float64 sums."""
    rng = np.random.default_rng(17)
    atoms = rng.normal(size=(47, 3)) * 10
    sigma = rng.uniform(.2, .4, len(atoms))
    epsilon = rng.uniform(.1, .8, len(atoms))
    epsilon[::5] = 0
    points = np.vstack((atoms[:4], rng.normal(size=(199, 3)) * 20))
    interaction = ProbeInteraction(atoms, sigma, epsilon)
    values, valid = interaction.evaluate(points, (.35, .45))
    mixed_sigma = .5 * (sigma + .35)
    mixed_depth = np.sqrt(epsilon * .45)
    expected = np.zeros(len(points))
    available = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        distance = np.linalg.norm(point - atoms, axis=1) * .1
        active = mixed_depth > 0
        if np.any(distance[active] < .65 * mixed_sigma[active]):
            available[i] = False
            continue
        ratio6 = (mixed_sigma[active] / distance[active]) ** 6
        expected[i] = np.sum(4 * mixed_depth[active] * (ratio6 * ratio6 - ratio6))
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(valid, available)


def test_exact_vdw_gap_is_not_nearest_atom_only():
    """The nearest center need not define the nearest sphere when radii differ."""
    axis = np.arange(-4, 5, dtype=float)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    atoms = np.array([[0.0, 0, 0], [3.0, 0, 0]])
    radii = np.array([0.5, 2.5])
    result = FieldQueries.domains(points, atoms, radii, (9, 9, 9), 0.2)
    expected = np.minimum(
        np.linalg.norm(points - atoms[0], axis=1) - radii[0],
        np.linalg.norm(points - atoms[1], axis=1) - radii[1],
    )
    np.testing.assert_allclose(result["distance"], expected, atol=1e-6)
    assert not result["accessible"][np.all(points == 0, axis=1)].any()
    assert result["accessible"][np.all(points == [-4, -4, -4], axis=1)].all()


@pytest.mark.parametrize("policy", ["uniform", "boundary", "stratified"])
def test_queries_are_unique_deterministic_and_redistribute_missing_regions(policy):
    """Absent closed cavities cannot produce synthetic or duplicated training positions."""
    regions = np.array([0] * 20 + [1] * 20 + [3] * 60 + [-1] * 10)
    valid = np.ones((110, 3), dtype=bool)
    valid[5] = False
    first = FieldQueries.sample(regions, valid, 60, 20, policy)
    np.testing.assert_array_equal(first, FieldQueries.sample(regions, valid, 60, 20, policy))
    assert len(first) == len(set(first)) == 60
    assert 5 not in first and np.all(regions[first] >= 0)
    assert len(FieldQueries.sample(regions, valid, 1000, 1)) == 99


def _queries_sample(identifier="tiny", offset=0):
    """Return tiny physical targets with a distinct local query/atom index namespace."""
    sample = dict(_sample(3, 4, 1.0))
    sample["identifier"] = identifier
    sample.update(
        {
            "tier": "fixture",
            "leakage_group": identifier,
            "global_phenotype": "unspecified",
            "local_phenotype": "unspecified",
        }
    )
    sample["atom_positions"] = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    sample.update(
        {
            "physical_queries__positions": torch.tensor([[2.0, 0, 0], [0.0, 2, 0]]),
            "physical_queries__targets": torch.tensor([[1.0, -2, 4], [3.0, 2, -1]]) + offset,
            "physical_queries__valid": torch.tensor([[True, False, True], [True, True, True]]),
            "physical_queries__region": torch.tensor([0, 1]),
            "physical_queries__offsets": torch.tensor([0, 3, 6]),
            "physical_queries__atom_ids": torch.tensor([0, 1, 2, 0, 1, 2]),
        }
    )
    sample["physical_queries__targets"][~sample["physical_queries__valid"]] = 0
    return sample


def test_train_only_scales_and_geometry_only_device_inputs():
    """Physical targets never enter the predictor's device mapping; masks are per channel."""
    sample = _queries_sample()
    objective = PhysicalObjective({"queries": 1})
    audit = objective.fit([sample])
    assert audit["valid_train_queries"] == [2, 1, 2]
    assert audit["mean"][0] == 2
    batch = WisdomCollator()([sample, _queries_sample("other", 20)])
    device = torch.device("cpu")
    geometry, references = objective.queries(batch, device)
    assert set(geometry) == {"positions", "owners", "pair_query", "pair_atom", "protein_count"}
    inputs = _device_batch(batch, device)
    assert "physical_queries" not in inputs.get("structural_assets", {})
    assert "targets" not in geometry and len(references["targets"]) == 2
    torch.testing.assert_close(
        batch["atom_batch"][geometry["pair_atom"]], geometry["owners"][geometry["pair_query"]]
    )
    objective.loss(torch.zeros((2, 3), requires_grad=True), references)["total"].backward()
    with pytest.raises(ValueError, match="not both"):
        PhysicalObjective({"weight": 1, "weights": {"potential": 1}})
    assert PhysicalObjective({"weights": {"potential": 1}}).weights.tolist() == [1, 0, 0]


@pytest.mark.parametrize("mode", ["three", "conditioned"])
def test_decoder_rigid_invariance_ownership_and_atomic_gradient(mode):
    """Distances, sums and means are rigid invariant; normal heads backpropagate into atoms."""
    sample = _queries_sample()
    batch = WisdomCollator()([sample])
    objective = PhysicalObjective({})
    objective.fit([sample])
    geometry, references = objective.queries(batch, torch.device("cpu"), full=True)
    decoder = FieldDecoder(8, hidden_dim=6, radial_count=5, output_mode=mode, chunk_size=2)
    states = torch.randn(3, 8, requires_grad=True)
    prediction = decoder(states, batch["atom_positions"], batch["atom_batch"], geometry)
    rotation = torch.tensor(
        Rotation.from_euler("xyz", [35, 70, 20], degrees=True).as_matrix(), dtype=torch.float32
    )
    transformed = {**geometry, "positions": geometry["positions"] @ rotation.T + 5}
    other = decoder(
        states, batch["atom_positions"] @ rotation.T + 5, batch["atom_batch"], transformed
    )
    torch.testing.assert_close(prediction, other, atol=1e-6, rtol=1e-5)
    objective.loss(prediction, references)["total"].backward()
    assert states.grad is not None and float(states.grad.abs().sum()) > 0
    invalid = {**geometry, "owners": torch.ones_like(geometry["owners"])}
    with pytest.raises(ValueError, match="ownership"):
        decoder(states, batch["atom_positions"], batch["atom_batch"], invalid)
    empty = {
        **geometry,
        "pair_query": torch.empty(0, dtype=torch.long),
        "pair_atom": torch.empty(0, dtype=torch.long),
    }
    assert torch.isfinite(
        decoder(states, batch["atom_positions"], batch["atom_batch"], empty)
    ).all()


def test_optional_heads_leave_matched_functional_initialization_unchanged():
    """Enabling an unused VF branch must not change the no-physics functional control."""
    torch.manual_seed(19)
    plain = WisdomV1(hidden_dim=8, embedding_dim=4, curvature_features=6)
    torch.manual_seed(19)
    physical = WisdomV1(
        hidden_dim=8, embedding_dim=4, curvature_features=6, physics_options={"hidden_dim": 5}
    )
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(value, physical.state_dict()[key], rtol=0, atol=0)
    plain.eval()
    physical.eval()
    batch = WisdomCollator()([_queries_sample()])
    inputs = _model_inputs(batch)
    torch.testing.assert_close(plain(**inputs)["logits"], physical(**inputs)["logits"])


def test_undefined_correlations_remain_unavailable_with_support():
    """Constant physical fields do not acquire a made-up zero Pearson correlation."""
    suite = PhysicalMetricSuite()
    target = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    suite.add("constant", target, target, np.ones_like(target, dtype=bool), np.array([0, 1]))
    result = suite.metrics()
    assert result["physics_potential_rmse"] == 0
    assert result["physics_potential_sign_accuracy"] == 1
    assert "physics_potential_pearson" not in result
    with pytest.raises(ValueError, match="non-finite"):
        suite.add(
            "bad", target * np.inf, target, np.ones_like(target, dtype=bool), np.array([0, 1])
        )


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_joint_gradient_diagnostic_keeps_sparse_backward_fp32_and_preserves_optimizer_gradients(
    device_type, monkeypatch
):
    """Real AMP/sparse autograd must survive the VF diagnostic and the subsequent joint backward.

    CPU verifies the precision boundary independently of CUDA availability. CUDA exercises the
    exact addmm_sparse_cuda regression with a tiny graph, never a cluster or dataset training job.
    """
    if device_type == "cuda" and (not torch.cuda.is_available() or
                                 not torch.cuda.is_bf16_supported()):
        pytest.skip("The granted CUDA device does not support BF16")
    device = torch.device(device_type)
    encoder = torch.nn.Linear(3, 3).to(device)
    model = SimpleNamespace(atomic_encoder=encoder)
    service = PhysicalTraining(SimpleNamespace(metrics=SimpleNamespace(log=Mock())), {}, Mock())
    inputs = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3) / 12
    operator = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device=device),
        torch.tensor([1., 2., 3., 4.], device=device), (4, 4), check_invariants=True,
    ).coalesce()
    with torch.autocast(device_type, dtype=torch.bfloat16):
        states = encoder(inputs)
        functional = DiffusionBlock.sparse_multiply(operator, states).square().mean()
        physical = .1 * states.float().square().mean()
    assert states.dtype == torch.bfloat16
    parameters = tuple(encoder.parameters())
    expected = [torch.autograd.grad(loss, parameters, retain_graph=True)
                for loss in (functional, physical)]
    expected_norms = [sum(float(g.detach().float().square().sum()) for g in vector)**.5
                      for vector in expected]
    backward_autocast_states = []
    native_grad = torch.autograd.grad

    def measured_grad(*args, **kwargs):
        """Record the actual backward precision context without mocking its calculations."""
        backward_autocast_states.append(torch.is_autocast_enabled(device_type))
        return native_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", measured_grad)
    with torch.autocast(device_type, dtype=torch.bfloat16):
        service.gradients(model, functional, physical, epoch=1)
    assert backward_autocast_states == [False, False]
    assert all(p.grad is None for p in parameters)
    logged = {call.args[0]: call.args[1] for call in service.work.metrics.log.call_args_list}
    assert logged["physics_functional_gradient_norm"] == pytest.approx(expected_norms[0])
    assert logged["physics_physical_gradient_norm"] == pytest.approx(expected_norms[1])
    assert np.isfinite(logged["physics_gradient_cosine"])
    (functional + physical).backward()
    for parameter, left, right in zip(parameters, *expected, strict=True):
        torch.testing.assert_close(parameter.grad, left + right, rtol=.02, atol=.01)
        assert torch.isfinite(parameter.grad).all()


def test_physical_split_overlap_and_external_inputs_are_rejected():
    """An existing asset is preserved on disk, but the principal VF encoder cannot read SLAE."""
    work = SimpleNamespace(outputs=Mock())
    service = PhysicalTraining(work, {}, _device_batch)
    record = (None, None, 1, "id", "tier", "g")
    datasets = {split: SimpleNamespace(records=[record]) for split in ("train", "val")}
    with pytest.raises(ValueError, match="share a leakage"):
        service.prepare(None, datasets)


def test_transfer_restores_atomic_weights_only_and_rejects_leakage(tmp_path):
    """A reviewed physical checkpoint cannot replace surface heads or overlap validation."""
    torch.manual_seed(3)
    source = WisdomV1(hidden_dim=8, embedding_dim=4, curvature_features=6, physics_options={})
    torch.manual_seed(7)
    destination = WisdomV1(hidden_dim=8, embedding_dim=4, curvature_features=6, physics_options={})
    service = PhysicalTraining(
        SimpleNamespace(log=Mock()), {"transfer_decoder": False}, _device_batch
    )
    surface_before = {
        name: value.clone()
        for name, value in destination.state_dict().items()
        if name.startswith("surface_projection.")
    }
    atomic = {
        name: value.clone()
        for name, value in source.state_dict().items()
        if name.startswith(PhysicalTraining.ATOMIC_PREFIXES)
        or PhysicalTraining.atomic_gate(source, name)
    }
    path = tmp_path / "encoder.pt"
    payload = {
        "atomic_state": atomic,
        "atomic_signature": service.signature(source),
        "training_groups": ["train"],
        "training_sources": ["source-train"],
    }
    torch.save(payload, path)
    service.validation_groups = {"validation"}
    service.validation_sources = {"source-validation"}
    service.load_encoder(destination, path)
    for name, value in atomic.items():
        torch.testing.assert_close(destination.state_dict()[name], value, rtol=0, atol=0)
    for name, value in surface_before.items():
        torch.testing.assert_close(destination.state_dict()[name], value, rtol=0, atol=0)
    service.validation_groups = {"train"}
    with pytest.raises(ValueError, match="overlaps"):
        service.load_encoder(destination, path)


def test_numerical_confidence_masks_unstable_voxels_not_whole_proteins(tmp_path):
    """Numerical uncertainty is an unavailable target, while strict and low-coverage cases fail."""
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = PhysicsConfig(margin_convergence=False, minimum_potential_coverage=0.8)
    producer.work = SimpleNamespace(log=Mock())
    nominal = np.ones((2, 2, 2))
    refined = nominal.copy()
    refined[0, 0, 0] = 5
    producer.solve = Mock(return_value=(refined, np.zeros(3), np.ones(3), "test-input"))
    points = np.stack(np.indices((2, 2, 2)), axis=-1).reshape(-1, 3)
    geometry = {"distance": np.ones(8), "region": np.zeros(8, dtype=int)}
    valid, audit, error = producer.audit_potential(
        tmp_path, np.zeros((1, 3)), nominal, np.zeros(3), np.ones(3), points, geometry
    )
    assert valid.tolist() == [False] + [True] * 7
    assert audit["coverage"] == 0.875 and audit["passed"]
    assert error[0] == 4 and nominal[0, 0, 0] == 1
    producer.config = replace(producer.config, potential_failure_policy="fail")
    with pytest.raises(ValueError, match="coverage policy"):
        producer.audit_potential(
            tmp_path, np.zeros((1, 3)), nominal, np.zeros(3), np.ones(3), points, geometry
        )


@pytest.mark.parametrize("cause", ["interruption", "allocation"])
@pytest.mark.parametrize("policy", ["strict", "bounded"])
def test_native_reference_checkpoints_survive_another_protein_failure(tmp_path, cause, policy):
    """Compatible retry or a raised allocation guard never rebuilds a completed protein."""
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = PhysicsConfig(heavy_atom_policy=policy)
    producer.maximum_voxels = 8000000
    producer.reference_parameters = producer.config.parameters()
    producer.reference_parameters.pop("bank_size")
    producer.report = {}
    producer.preparation = SimpleNamespace(provenance={"version": "test"})
    producer.work = SimpleNamespace(
        log=Mock(),
        checkpoints=CheckpointCollection(tmp_path / "cp"),
        cache=WorkCache(tmp_path / "cache"),
        map=lambda items, function, **kwargs: [function(item) for item in items],
    )
    bases = {name: tmp_path / (name + ".npz") for name in ("first", "second")}
    for name, path in bases.items():
        path.write_bytes(name.encode())
    calls = []
    failed = False

    def reference(base, output, **kwargs):
        """Simulate a missing reference or operational refusal at the native file boundary."""
        nonlocal failed
        calls.append(base.stem)
        if base.stem == "second" and not failed:
            if cause == "allocation" and producer.maximum_voxels < 24000000:
                raise ValueError("unit grid exceeds operational allocation ceiling")
            failed = True
            raise RuntimeError("unit interruption")
        output.write_bytes(b"complete-reference")

    producer.write_reference = reference
    def queries(base, output, reference):
        """Write a tiny native query file with explicit support metadata for the audit."""
        with output.open("wb") as stream:
            np.savez(stream, valid=np.ones((1, 3), dtype=bool), metadata_json=np.asarray("{}"))

    producer.write_queries = queries
    producer.validate_reference = lambda path, **kwargs: (
        Path(path).read_bytes() == b"complete-reference"
    )
    producer.validate_queries = lambda path, **kwargs: Path(path).is_file()
    with pytest.raises(RuntimeError, match="1/2 proteins"):
        producer.generate(bases, workers=1)
    assert producer.report["completed"] == 1
    assert producer.work.checkpoints.load_json("physics/last-report.json")["failed"] == 1
    if cause == "allocation":
        producer.maximum_voxels = 24000000
        failed = True  # The operational refusal is resolved; no synthetic interruption follows.
    assert len(producer.generate(bases, workers=1)) == 2
    assert calls == ["first", "second", "second"]
    producer.config = replace(producer.config, bank_size=1024)
    assert len(producer.generate(bases, workers=1)) == 2
    assert calls == ["first", "second", "second"]  # Query count does not rerun chemical solvers.
    # A fresh Run can restore the same reference from the native cache without owning the earlier
    # Run's checkpoint directory. No source/config/tool key or scientific validation is bypassed.

    producer.work.checkpoints = CheckpointCollection(tmp_path / "new-cp")
    assert len(producer.generate(bases, workers=1)) == 2
    assert calls == ["first", "second", "second"]


@pytest.mark.parametrize("stem,spacing,margin,fixed_delta,expected", [
    ("potential", 1., None, None, [385, 193, 129]),
    ("refined", .75, None, None, [513, 257, 161]),
    ("larger-box", 1., 18., np.array([373.487, 171.211, 113.867]) / [384, 192, 128],
     [417, 225, 161]),
])
def test_elongated_apbs_grids_keep_exact_scientific_inputs_when_guard_increases(
    tmp_path, stem, spacing, margin, fixed_delta, expected
):
    """Only admit the previously refused grid; do not coarsen it or change solver physics.

    Bounding extents follow 8GLV_Ey with 2 Å padding per side for preparation. The mocked
    OpenDX reader avoids a real APBS solve or allocating millions of voxels in this unit test.
    """
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = PhysicsConfig(heavy_atom_policy="bounded")
    producer.maximum_voxels = 8000000
    producer.preparation = SimpleNamespace(apbs="apbs")
    producer.work = SimpleNamespace(log=Mock(), tools=SimpleNamespace(run=Mock(
        return_value=SimpleNamespace(stdout="", stderr=""))))
    producer.grid_module = SimpleNamespace(Grid=Mock(return_value=SimpleNamespace(
        grid=np.ones((2, 2, 2)), origin=np.zeros(3), delta=np.ones(3))))
    positions = np.array([[0., 0, 0], [349.487, 147.211, 89.867]])
    settings = producer.config.parameters()
    voxels = int(np.prod(expected))
    with pytest.raises(ValueError, match=f"{stem} grid.*{voxels:,}.*accept-operational"):
        producer.solve(tmp_path, positions, spacing, stem, margin, fixed_delta)
    producer.work.tools.run.assert_not_called()
    assert not (tmp_path / f"{stem}.in").exists()
    (tmp_path / f"{stem}.dx").write_text("Synthetic reader fixture, not physical evidence.")
    producer.maximum_voxels = 24000000
    _, _, _, text = producer.solve(tmp_path, positions, spacing, stem, margin, fixed_delta)
    assert f" dime {' '.join(map(str, expected))}\n" in text
    assert "lpbe\n" in text and " pdie 2.0\n sdie 78.5\n" in text
    assert producer.config.parameters() == settings
    assert len(producer.work.tools.run.call_args_list) == 1
    producer.maximum_voxels = voxels  # The exact count, not a strictly larger ceiling, is valid.
    assert producer.solve(tmp_path, positions, spacing, stem, margin, fixed_delta)[3] == text


def test_unavailable_chemistry_has_no_fabricated_queries(tmp_path):
    """Unsupported observed chemistry preserves base bytes and yields a validated empty bank."""
    base = tmp_path / "base.npz"
    base.write_bytes(b"unit base identity, never rewritten")
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = PhysicsConfig()
    producer.reference_parameters = producer.config.parameters()
    producer.reference_parameters.pop("bank_size")
    producer.work = SimpleNamespace(log=Mock(), temp_dir=tmp_path)
    producer.preparation = SimpleNamespace(provenance={"unit": True},
        prepare=Mock(side_effect=PhysicalReferenceUnavailable("reconstructed OXT")))
    reference, queries = tmp_path / "ref.npz", tmp_path / "queries.npz"
    producer.write_reference(base, reference)
    assert producer.validate_reference(reference, base, producer.reference_parameters)
    producer.write_queries(base, queries, reference)
    assert producer.validate_queries(queries, base, producer.config.parameters())
    values, header = StructuralAsset.read(queries, base, producer.QUERIES)
    assert header["availability"] == "unavailable" and "OXT" in header["reason"]
    assert values["positions"].shape == (0, 3) and not values["valid"].any()
    assert base.read_bytes() == b"unit base identity, never rewritten"
    # A reason alone cannot qualify fabricated measurements as an unavailable bank.

    values["targets"] = np.ones((1, 3), dtype=np.float32)
    StructuralAsset.write(queries, base, values, producer.QUERIES,
                          producer.config.parameters(), header)
    with pytest.raises(ValueError, match="must be empty"):
        producer.validate_queries(queries, base)


def test_empty_physical_batch_does_not_create_targets_or_sparse_pairs():
    """A fully unavailable batch has no physical loss samples, not positive/negative labels."""
    sample = _queries_sample()
    for key, value in list(sample.items()):
        if key.startswith("physical_queries__"):
            sample[key] = torch.zeros(1, dtype=torch.long) if key.endswith("offsets") else value[:0]
    batch = WisdomCollator()([sample])
    objective = PhysicalObjective({})
    geometry, references = objective.queries(batch, torch.device("cpu"))
    assert geometry["positions"].shape == (0, 3)
    assert geometry["pair_query"].shape == geometry["pair_atom"].shape == (0,)
    assert references["targets"].shape == (0, 3)
    with pytest.raises(ValueError, match="no valid potential"):
        objective.fit([sample])


def test_mask_policy_discards_unqualified_potential_only(tmp_path):
    """Failing coverage masks the complete potential without claiming numerical convergence."""
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = PhysicsConfig(margin_convergence=False)
    producer.work = SimpleNamespace(log=Mock())
    nominal = np.ones((2, 2, 2))
    producer.solve = Mock(return_value=(nominal * 5, np.zeros(3), np.ones(3), "unit-input"))
    points = np.stack(np.indices((2, 2, 2)), axis=-1).reshape(-1, 3)
    valid, audit, error = producer.audit_potential(tmp_path, np.zeros((1, 3)), nominal,
        np.zeros(3), np.ones(3), points,
        {"distance": np.ones(8), "region": np.zeros(8, dtype=int)})
    assert not valid.any() and not audit["passed"] and audit["valid_queries"] == 0
    assert np.all(error == 4) and audit["candidate_coverage"] == 0


@pytest.mark.parametrize("diagnostic,known", [
    ("CRITICAL:Found gap in biomolecule structure for atom HG", True),
    ("ERROR:This PDB file is missing too many heavy atoms", True),
    ("CRITICAL:Too few atoms present to reconstruct or cap residue ALA A 1", True),
    ("Traceback: ImportError: missing module", False),
])
def test_only_known_chemistry_failures_are_unavailable(tmp_path, diagnostic, known):
    """Native command failures are not blanket-converted into apparently successful data."""
    preparation = MolecularPreparation.__new__(MolecularPreparation)
    preparation.config = PhysicsConfig()
    preparation.pdb2pqr = "pdb2pqr"
    preparation.work = SimpleNamespace(tools=SimpleNamespace(run=Mock(return_value=
        SimpleNamespace(returncode=1, stdout="", stderr=diagnostic, name="vf-protonation"))))
    base = tmp_path / "base.npz"
    np.savez(base, atom_positions=np.zeros((1, 3)), atomic_numbers=np.array([6]),
        atom_names=np.array(["CA"]), residue_names=np.array(["ALA"]),
        residue_indices=np.zeros(1, dtype=int), chain_indices=np.zeros(1, dtype=int),
        metadata_json=np.asarray("{}"))
    expected = PhysicalReferenceUnavailable if known else RuntimeError
    with pytest.raises(expected):
        preparation.prepare(base, tmp_path)


def test_partial_field_support_cannot_improve_validation_loss_by_missingness():
    """Loss averages use each field's actual protein support, never missing-as-zero values."""
    work = SimpleNamespace(metrics=SimpleNamespace(log=Mock()), log=Mock())
    service = PhysicalTraining(work, {"objective": {"weight": 1, "loss": "mse"}}, Mock())
    loader = [{"identifier": [name]} for name in ("all", "lj_only", "unavailable")]
    masks = [torch.tensor([[True, True, True]]), torch.tensor([[False, True, True]]),
             torch.empty((0, 3), dtype=torch.bool)]
    references = [{"targets": torch.zeros(mask.shape), "valid": mask,
                   "region": torch.zeros(len(mask), dtype=torch.long),
                   "owners": torch.zeros(len(mask), dtype=torch.long)} for mask in masks]
    service.prediction = Mock(side_effect=[(torch.ones(ref["targets"].shape), ref)
                                           for ref in references])
    result = service.evaluate(Mock(), loader, torch.device("cpu"))
    assert result["physics_loss"] == pytest.approx(3.0)
    assert result["physics_unavailable_proteins"] == 1
    assert result["physics_potential_proteins"] == 1
    assert result["physics_lj1_proteins"] == result["physics_lj2_proteins"] == 2
    # A globally absent positively weighted validation field cannot become a free zero term.

    service.prediction = Mock(return_value=(torch.ones((1, 3)), references[1]))
    with pytest.raises(ValueError, match=r"lacks positively weighted fields.*potential"):
        service.evaluate(Mock(), loader[:1], torch.device("cpu"))


def test_empty_physical_forward_skips_the_atomic_encoder():
    """An unavailable-only protein must not pay an atomic forward just to get no targets."""
    sample = _queries_sample()
    for key, value in list(sample.items()):
        if key.startswith("physical_queries__"):
            sample[key] = torch.zeros(1, dtype=torch.long) if key.endswith("offsets") else value[:0]
    move = Mock(side_effect=AssertionError("must not transfer an unavailable graph"))
    service = PhysicalTraining(SimpleNamespace(), {}, move)
    model = Mock(physics_decoder=FieldDecoder(8))
    prediction, references = service.prediction(model, WisdomCollator()([sample]),
                                                torch.device("cpu"))
    assert prediction.shape == references["targets"].shape == (0, 3)
    assert prediction.requires_grad
    model.encode_atoms.assert_not_called()


def test_failed_potential_and_audited_lj_can_share_a_validated_reference(tmp_path):
    """Partial-channel storage is audited without inventing APBS convergence or dropping LJ."""
    base = tmp_path / "base.npz"
    atoms = np.zeros((1, 3))
    np.savez(base, atom_positions=atoms, atom_names=np.array(["CA"]),
             residue_indices=np.zeros(1, int), chain_indices=np.zeros(1, int))
    config = PhysicsConfig()
    settings = config.parameters()
    settings.pop("bank_size")
    shape = (2, 2, 2)
    valid = np.ones((*shape, 3), dtype=bool)
    valid[..., 0] = False
    arrays = {"potential": np.ones(shape, dtype=np.float32),
        "targets": np.zeros((*shape, 3), dtype=np.float32), "valid": valid,
        "delta": np.ones(3), "origin": np.ones(3),
        "distance": np.ones(shape), "region": np.zeros(shape, dtype=np.int8),
        "accessible": np.ones(shape, dtype=bool), "lj_energy": np.zeros((*shape, 2)),
        "lj_overlap_valid": np.ones((*shape, 2), dtype=bool), "heavy_mapping": np.zeros(1, int),
        "prepared_positions": atoms}
    audit = {"passed": False, "coverage": 0.0, "valid_queries": 0,
        "region_coverage": {"0": 0.0}, "candidate_coverage": 0.5,
        "candidate_region_coverage": {"0": 0.5}, "candidate_valid_queries": 4,
        "comparisons": [{"name": "refined"}, {"name": "larger-box"}]}
    metadata = {"algorithm": PhysicalFields.ALGORITHM, "units": ["kBT/e", "kJ/mol", "kJ/mol"],
        "potential_convergence": audit, "lj_audits": [{"fixture": True}, {"fixture": True}]}
    path = tmp_path / "reference.npz"
    StructuralAsset.write(path, base, arrays, PhysicalFields.REFERENCE, settings, metadata)
    assert PhysicalFields.validate_reference(path, base, settings)
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = config
    queries = tmp_path / "queries.npz"
    producer.write_queries(base, queries, path)
    assert PhysicalFields.validate_queries(queries, base, config.parameters())
    bank, _ = StructuralAsset.read(queries, base, PhysicalFields.QUERIES)
    assert not bank["valid"][:, 0].any() and bank["valid"][:, 1:].all()
    audit["candidate_coverage"] = 1.0
    audit["candidate_region_coverage"] = {"0": 1.0}
    StructuralAsset.write(path, base, arrays, PhysicalFields.REFERENCE, settings, metadata)
    with pytest.raises(ValueError, match="convergence audit"):
        PhysicalFields.validate_reference(path, base, settings)


def test_physical_publication_preserves_slae_assets_membership_and_labels(
    tmp_path, pdb_path, monkeypatch
):
    """Test publication ownership with sentinel assets, without claiming they are real SLAE/VF."""
    dataset = tmp_path / "source"
    base = _complete_base(dataset, pdb_path)
    slae = dataset / "slae.npz"
    slae.write_bytes(b"unit sentinel: existing immutable SLAE representation")
    reference, queries = dataset / "reference.npz", dataset / "queries.npz"
    reference.write_bytes(b"unit sentinel: separately validated physical reference")
    queries.write_bytes(b"unit sentinel: separately validated physical queries")
    member = DatasetMember(
        "tiny",
        assets={
            "universal_npz": DatasetAsset(path=base.name),
            "external_atomic_slae": DatasetAsset(path=slae.name),
        },
        targets={"dna_binding": 1},
        partitions={"split": "train", "leakage_group": "group"},
        metadata={"scientific_fixture": True},
    )
    DatasetIndex.write(dataset / "index.jsonl", [member])
    producer = SimpleNamespace(
        report={"completed": 1},
        generate=Mock(
            return_value={"tiny": {"physical_reference": reference, "physical_queries": queries}}
        ),
    )
    module = __import__(
        "wisdom.preprocessing.common.StructuralEnrichment", fromlist=["PhysicalFields"]
    )
    monkeypatch.setattr(module, "PhysicalFields", Mock(return_value=producer))
    monkeypatch.setattr(StructuralSidecars, "validate_assets", Mock())
    work = _work(tmp_path / "work")
    work.checkpoints = CheckpointCollection(tmp_path / "work" / "checkpoints")
    original = {path: path.read_bytes() for path in (base, slae)}
    StructuralEnrichment.run(
        work,
        dataset,
        "unit-only-physics",
        "1",
        atomic_context=False,
        extended_surface_atom=False,
        physics_options={},
        workers=1,
    )
    published = work.outputs.dataset.call_args.kwargs["members"][0]
    assert published["targets"] == dict(member.targets)
    assert published["partitions"] == dict(member.partitions)
    assert published["metadata"] == dict(member.metadata)
    assert published["id"] == "tiny"
    assert set(published["assets"]) == {
        "universal_npz",
        "external_atomic_slae",
        "physical_reference",
        "physical_queries",
    }
    for path, content in original.items():
        assert path.read_bytes() == content


def _physical_dataset(root: Path, pdb_path: Path) -> Path:
    """Create a native index with synthetic physical targets; never publish it in the Registry."""
    base = _complete_base(root / "base", pdb_path)
    with np.load(base, allow_pickle=False) as stored:
        atoms = stored["atom_positions"]
    origin = atoms[0].astype(float) + np.array([0, 0, 4])
    positions = (origin + np.array([[0, 0, 0], [1, 0, 0]])).astype(np.float32)
    nearby = cKDTree(atoms).query_ball_point(positions, 12, return_sorted=True)
    offsets = np.r_[0, np.cumsum([len(row) for row in nearby])]
    queries = root / "queries.npz"
    targets = np.array([[1.0, -2.0, 4.0], [3.0, 2.0, -1.0]], dtype=np.float32)
    StructuralAsset.write(
        queries,
        base,
        {
            "positions": positions.astype(np.float32),
            "targets": targets,
            "valid": np.ones((2, 3), dtype=bool),
            "region": np.array([0, 1], dtype=np.int8),
            "distance": np.ones(2, dtype=np.float32),
            "accessible": np.ones(2, dtype=bool),
            "grid_index": np.array([0, 1]),
            "offsets": offsets,
            "atom_ids": np.concatenate(nearby).astype(np.int32),
        },
        "physical_queries",
        PhysicsConfig().parameters(),
        {
            "algorithm": PhysicalFields.ALGORITHM,
            "physical_sources_are_targets_only": True,
            "units": ["kBT/e", "kJ/mol", "kJ/mol"],
            "reference_sha256": "fixture",
            "grid_shape": [2, 1, 1],
            "grid_origin": origin.tolist(),
            "grid_delta": [1, 1, 1],
        },
    )
    members = [
        DatasetMember(
            f"{split}-{label}",
            assets={
                "universal_npz": DatasetAsset(path=str(base.relative_to(root))),
                "physical_queries": DatasetAsset(path="queries.npz"),
            },
            targets={"dna_binding": label},
            partitions={"split": split, "leakage_group": split + str(label)},
        )
        for split in ("train", "validation")
        for label in (0, 1)
    ]
    DatasetIndex.write(root / "index.jsonl", members)
    return root


@pytest.mark.parametrize(
    "mode,control,precision",
    [
        ("pretrain", {}, "float32"),
        ("pretrain", {"no_messages": True}, "float32"),
        ("pretrain", {"freeze_encoder": True}, "float32"),
        ("pretrain", {"pretraining_task": "masked_element"}, "float32"),
        ("joint", {}, "float32"),
        ("joint", {}, "bfloat16"),
    ],
)
def test_native_training_vf_is_executable_and_preserves_selection_policy(
    tmp_path, pdb_path, monkeypatch, request, mode, control, precision
):
    """The real public Work fits fields or joint losses, with no functional/local GT leakage."""
    if precision == "bfloat16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            pytest.skip("The granted CUDA device does not support BF16")
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    request.addfinalizer(lambda: torch.set_num_threads(previous))
    dataset = _physical_dataset(tmp_path / "data", pdb_path)
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "run": "wisdom.Training.Training",
                "with": {
                    "dataset": {"file": str(dataset)},
                    "model_version": 2,
                    "architecture_implementation": "controlled",
                    "head_implementation": "controlled",
                    "atomic_encoder_type": "schnet",
                    "atomic_hidden_dim": 8,
                    "atomic_output_dim": 8,
                    "hidden_dim": 8,
                    "embedding_dim": 4,
                    "atomic_layers": 1,
                    "surface_layers": 1,
                    "diffusion_spectral_modes": 8,
                    "batch_size": 2,
                    "epochs": 2,
                    "patience": 2,
                    "precision": precision,
                    "data_workers": 0,
                    "surface_metrics": False,
                    "visualization": {"mode": "none"},
                    "evaluate_test": False,
                    "physics": {
                        **control,
                        "mode": mode,
                        "objective": {"queries": 2},
                        "decoder": {"hidden_dim": 8, "radial_count": 4},
                    },
                },
            },
            sort_keys=False,
        )
    )
    runner = WorkRunner(execution_root=tmp_path / "runs")
    if mode == "joint":
        measured = []
        native_diagnostic = PhysicalTraining.gradients

        def inspect_backward_boundary(self, model, functional, physical, epoch):
            """Verify Training invokes real diagnostics outside forward AMP on the toy dataset."""
            assert functional.device.type == ("cuda" if precision == "bfloat16" else "cpu")
            measured.append(torch.is_autocast_enabled(functional.device.type))
            return native_diagnostic(self, model, functional, physical, epoch)

        monkeypatch.setattr(PhysicalTraining, "gradients", inspect_backward_boundary)
    if mode == "pretrain":
        evaluate = PhysicalTraining.evaluate
        calls = []

        def interrupted(self, *args, **kwargs):
            """Interrupt epoch two; native recovery must reuse committed physical epoch one."""
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("unit interruption after a complete physical epoch")
            return evaluate(self, *args, **kwargs)

        monkeypatch.setattr(PhysicalTraining, "evaluate", interrupted)
        first = runner.run(WorkConfig.from_yaml(config))
        assert first.status == "failed"
        continuation = next((tmp_path / "runs").rglob("continuation.pt"))
        assert torch.load(continuation, weights_only=True)["epoch"] == 1
        monkeypatch.setattr(PhysicalTraining, "evaluate", evaluate)
    result = runner.run(WorkConfig.from_yaml(config))
    assert result.status == "succeeded", result
    if mode == "joint":
        assert measured and not any(measured)
    normalization = json.loads(
        next((tmp_path / "runs").rglob("physical-normalization.json")).read_text()
    )
    assert normalization["valid_train_queries"] == [4, 4, 4]
    evaluation = json.loads(next((tmp_path / "runs").rglob("physical-evaluation.json")).read_text())
    assert evaluation["metrics"]["physics_potential_proteins"] == 2
    if mode == "pretrain":
        checkpoint = torch.load(
            next((tmp_path / "runs").rglob("physical-encoder.pt")),
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["functional_labels_used"] is False
        assert all(
            name.startswith(PhysicalTraining.ATOMIC_PREFIXES) or name.startswith("semantic_gates.")
            for name in checkpoint["atomic_state"]
        )
        assert not list((tmp_path / "runs").rglob("best-model.pt"))
    else:
        checkpoint = torch.load(
            next((tmp_path / "runs").rglob("best-model.pt")), map_location="cpu", weights_only=True
        )
        assert "physical_normalization" in checkpoint
        assert checkpoint["best_epoch"] if "best_epoch" in checkpoint else checkpoint["epoch"] >= 1


@pytest.mark.parametrize("name", ["wisdom_vf1.yaml", "wisdom_vf_joint.yaml"])
def test_authored_vf_candidates_construct_the_actual_encoder_and_heads(name):
    """Check scientific constructors as well as the LF schema, without submitting any Runs."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "experiments/vf" / name).read_text())
    assert config["seeds"] == [1103, 1109, 1117, 1123]
    assert "steps" not in config
    for candidate in config["sweep"]["space"]["physics"]["values"]:
        accepted = set(inspect.signature(WisdomV1).parameters) | set(
            inspect.signature(WisdomV2).parameters
        )
        parameters = {
            **{key: value for key, value in config["with"].items() if key in accepted},
            "curvature_features": 15,  # Production schema uses the 41-gate grouped baseline.
            "physics_options": candidate.get("decoder", {}),
        }
        model, _ = _create_model(2, parameters)
        assert model.physics_decoder is not None
        assert model.atomic_encoder_type.value == "schnet"
        assert model.atomic_representation.external_model is None
        assert model.atomic_representation.atomic_external_model is None


def test_vf_pilot_pretraining_and_joint_atomic_checkpoints_are_compatible():
    """The physical checkpoint must match the entire atomic signature of its functional target."""
    root = Path(__file__).resolve().parents[1] / "experiments/vf"
    accepted = set(inspect.signature(WisdomV1).parameters) | set(
        inspect.signature(WisdomV2).parameters
    )
    signatures = []
    for name in ("wisdom_vf1.yaml", "wisdom_vf_joint.yaml"):
        options = yaml.safe_load((root / name).read_text())["with"]
        parameters = {key: value for key, value in options.items() if key in accepted}
        model, _ = _create_model(2, {**parameters, "curvature_features": 15,
                                     "physics_options": {}})
        signatures.append(PhysicalTraining.signature(model))
        assert (options["atomic_layers"], options["projection_depth"],
                options["surface_layers"]) == (1, 1, 6)
    assert signatures[0] == signatures[1]
