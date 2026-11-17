"""Physical residue communication, free-space geometry and frozen isolated-view contracts."""

# ruff: noqa: I001

import json
import copy
import torch
import numpy as np
import pytest
import yaml
import hashlib

from pathlib import Path
from functools import partial
from types import SimpleNamespace
from lambdaforge.work.checkpoints import CheckpointCollection
from test_representation_campaign import chemical_batch
from test_v8_architectures import model
from test_structural_enrichment import _complete_base, _work
from wisdom.Training import _model_inputs
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.models.residue.ResidueGraphEncoder import ResidueGraphEncoder
from wisdom.models.surface.SurfaceEnvironmentInput import SurfaceEnvironmentInput
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.evaluation.SurfaceMapAudit import SurfaceMapAudit
from wisdom.preprocessing.common.structure.assets.surface.SurfaceView import SurfaceView
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars
from wisdom.preprocessing.common.structure.assets.external.AtomicProteinExtractor import (
    AtomicProteinExtractor,
)
from wisdom.preprocessing.common.structure.assets.external.SLAEExtraction import SLAEExtraction
from wisdom.preprocessing.common.structure.assets.graphs.ResidueGraph import ResidueGraph
from wisdom.preprocessing.common.structure.assets.surface.AccessibleSpace import AccessibleSpace
from wisdom.preprocessing.common.structure.assets.surface.SurfaceEnvironment import (
    SurfaceEnvironment,
)


def residue_member(count=2, points=6):
    """Construct two physical CA nodes and explicit per-point residue candidates."""
    return {"backbone_valid": torch.ones((count, 4), dtype=torch.bool),
        "ca_edge_index": torch.tensor([[0], [1]]), "ca_edge_rank": torch.ones((2, 1)),
        "ca_edge_relation": torch.tensor([1]), "ca_edge_distance": torch.tensor([3.8]),
        "surface_ptr": torch.arange(points+1)*2,
        "surface_residue": torch.tensor([0, 1]*points),
        "surface_distance": torch.ones(2*points),
        "metadata_json": json.dumps({"parameters": {"ca_radius": 12., "ca_k_max": 32,
                                                    "surface_radius": 12.}})}


@pytest.mark.parametrize("topology", ["descending", "roundtrip", "parallel"])
@pytest.mark.parametrize("mode", ["pointwise", "null", "sequence", "contact", "hybrid"])
def test_residue_routes_recover_baseline_then_receive_gradients(topology, mode):
    """Zero output, nonzero alpha recovers the old predictor and permits subsequent learning."""
    options = dict(sources=["atom_name", "residue_atom"], external_model="esm2",
                   external_mode="context", context_dim=4)
    torch.manual_seed(211)
    baseline = model("baseline", atomic_encoder_type="schnet", atomic_representation=options).eval()
    torch.manual_seed(211)
    candidate = model("baseline", atomic_encoder_type="schnet", atomic_representation={**options,
        "residue_graph": {"mode": mode, "topology": topology, "atomic_dim": 8,
                          "hidden_dim": 8}}).eval()
    data = chemical_batch()
    sizes = data["surface_ptr"].diff().tolist()
    data["structural_assets"]["residue_graph"] = {
        "members": [residue_member(points=n) for n in sizes]}
    baseline.atomic_representation.fit([data])
    candidate.atomic_representation.fit([data])
    first = baseline(**_model_inputs(data))["logits"]
    second = candidate(**_model_inputs(data))["logits"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    branch = candidate.atomic_representation.residue_encoder
    second.sum().backward()
    assert branch.output.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=.01)
    optimizer.step()
    optimizer.zero_grad()
    candidate(**_model_inputs(data))["logits"].sum().backward()
    assert branch.input.weight.grad.abs().sum() > 0
    restored = copy.deepcopy(candidate)
    restored.load_state_dict(candidate.state_dict(), strict=True)
    torch.testing.assert_close(candidate(**_model_inputs(data))["logits"],
                               restored(**_model_inputs(data))["logits"], rtol=0, atol=0)


def test_ca_graph_is_independent_of_heavy_contacts_and_preserves_peptides():
    """CA contacts come from CA positions, never a truncated old heavy-contact list."""
    backbone = np.zeros((3, 4, 3))
    backbone[:, 1, 0] = [0, 9, 25]
    backbone[:, 0] = backbone[:, 1] + [0, 1, 0]
    backbone[:, 2] = backbone[:, 1] + [1, 0, 0]
    valid = np.ones((3, 4), bool)
    producer = ResidueGraph(ca_radius=12, ca_k_max=32)
    values = producer.ca_geometry(backbone, valid, np.array([[1], [2]]), np.array([1]))
    assert values["ca_edge_index"].T.tolist() == [[0, 1], [1, 2]]
    assert values["ca_edge_distance"].tolist() == [9, 16]
    assert values["ca_edge_rank"][:, 1].tolist() == [0, 0]
    valid[1, 1] = False
    assert producer.ca_geometry(backbone, valid, np.empty((2, 0), int),
                                np.empty(0, int))["ca_edge_index"].shape == (2, 0)


def test_empty_sphere_clearance_and_ray_occlusion_are_not_surface_curvature():
    """An isolated sphere has exposed outside rays and zero exterior depth, not a cavity radius."""
    atoms, radii = np.zeros((1, 3)), np.array([1.7])
    environment = SurfaceEnvironment(spacing=1.5, rays=16, hull_directions=24)
    points = np.array([[3.1, 0, 0], [-3.1, 0, 0], [0, 3.1, 0]])
    arrays = dict(atom_positions=atoms, vdw_radii=radii, surface_positions=points,
                  surface_normals=points/3.1)
    result = environment.build(arrays)
    assert not result["occlusion_far"].any()
    assert not result["empty_radius_valid"].any()
    assert np.all(result["travel_depth"] >= 0)
    domain = AccessibleSpace(np.array([[0., 0, 0], [4., 0, 0]]), np.array([1., 3.]),
                             spacing=2, hull_directions=24)
    query = np.array([[1.2, .1, .3], [8, 2, 0]])
    exact = np.min(np.linalg.norm(query[:, None]-domain.atoms, axis=2)-domain.radii, axis=1)
    np.testing.assert_allclose(domain.gaps(query), exact, atol=1e-12)
    with pytest.raises(MemoryError):
        AccessibleSpace(atoms, radii, spacing=.1, maximum_voxels=10)


def test_environment_guard_increase_restores_legacy_native_checkpoints(tmp_path, pdb_path,
                                                                      monkeypatch):
    """Increasing an allocation guard must not rebuild physically identical completed assets.

    Args:
        tmp_path: Isolated real universal archive and LF-owned checkpoint collection.
        pdb_path: Small deposited-structure fixture, never a production dataset.
        monkeypatch: Forbids environmental reconstruction after seeding a legacy checkpoint.
    """
    base = _complete_base(tmp_path, pdb_path)
    options = dict(spacing=2., rays=8, hull_directions=12)
    old = SurfaceEnvironment(**options, maximum_voxels=2000000)
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    settings = {"base": digest, "family": old.FAMILY, "schema": "1.0",
                "parameters": old.parameters}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    checkpoints = CheckpointCollection(tmp_path / "native-checkpoints")
    saved = checkpoints.file(f"structural-sidecars/{key}/{old.FAMILY}.npz",
        build=partial(old.write, base), validate=partial(old.validate, base=base))
    original = Path(saved).read_bytes()

    def forbidden(*args, **kwargs):
        """Reject an unnecessary voxel solve when an exact physical result already exists.

        Args:
            args: Unexpected producer positional arguments.
            kwargs: Unexpected producer keyword arguments.

        Raises:
            AssertionError: Always; LF must restore and scientifically validate the old file.
        """
        raise AssertionError("completed environment was rebuilt")

    monkeypatch.setattr(SurfaceEnvironment, "write", forbidden)
    service = StructuralSidecars(surface_environment=True, environment_options=options)
    work = _work(tmp_path)
    work.checkpoints = checkpoints
    output = service.generate(work, {"tiny": base}, 1)
    assert output["tiny"][old.FAMILY] == Path(saved)
    assert Path(saved).read_bytes() == original
    assert service.report["completed"] == 1
    current = SurfaceEnvironment(**options)
    assert current.parameters["maximum_voxels"] == 18000000
    assert current.validate(Path(saved), base)
    with pytest.raises(ValueError, match="configuration mismatch"):
        SurfaceEnvironment(**{**options, "spacing": 1.5}).validate(Path(saved), base)


def test_environment_allocation_ceiling_does_not_change_arrays_or_scientific_key():
    """Operational bounds affect feasibility only, never resolution, masks or physical values."""
    points = np.array([[3.1, 0., 0.], [-3.1, 0., 0.]])
    arrays = dict(atom_positions=np.zeros((1, 3)), vdw_radii=np.array([1.7]),
                  surface_positions=points, surface_normals=points / 3.1)
    small = SurfaceEnvironment(spacing=2, rays=8, hull_directions=12, maximum_voxels=2000)
    large = SurfaceEnvironment(spacing=2, rays=8, hull_directions=12, maximum_voxels=18000000)
    assert small.scientific_parameters() == large.scientific_parameters()
    left, right = small.build(arrays), large.build(arrays)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], strict=True)
    with pytest.raises(MemoryError, match="limit=1"):
        SurfaceEnvironment(spacing=2, rays=8, hull_directions=12,
                           maximum_voxels=1).build(arrays)


def test_environment_statistics_ignore_invalid_values_and_preserve_mask_control():
    """Only valid selected-training values fit statistics; masked missingness is explicit."""
    module = SurfaceEnvironmentInput(4, ["travel_depth"], mode="real")
    member = {"travel_depth": torch.tensor([1., 3., 0.]),
              "travel_depth_valid": torch.tensor([True, True, False])}
    report = module.fit([{"structural_assets": {"surface_environment": {"members": [member]}}}])
    assert report["travel_depth"] == {"count": 2, "mean": 2., "scale": 1.}
    curvature = torch.randn(3, 8)
    output = module([member], curvature)
    assert not output.any()
    output.sum().backward()
    assert module.projection.weight.grad.abs().sum() > 0


def test_residue_support_guard_and_double_zero_learning_guard():
    """Unsupported geometry and dead zero-times-zero branches fail before plausible training."""
    with pytest.raises(ValueError, match="block"):
        ResidueGraphEncoder(4, initial_scale=0)
    module = ResidueGraphEncoder(4, radius=15)
    with pytest.raises(ValueError, match="support"):
        module(torch.randn(2, 4), torch.ones(2, dtype=torch.bool), residue_member())


@pytest.mark.parametrize("kind", ["identity", "phase", "grid", "resolution", "rigid"])
def test_isolated_views_regenerate_coherent_operators_without_changing_base(
    tmp_path, pdb_path, kind,
):
    """Each admitted numerical factor gets a complete independent physical-point package."""
    base = _complete_base(tmp_path, pdb_path)
    original = base.read_bytes()
    view = SurfaceView(1, None, kind, 607)
    target = tmp_path / (kind + ".npz")
    view.write(base, target)
    assert view.validate(target, base)
    assert base.read_bytes() == original
    with np.load(base, allow_pickle=False) as stored:
        identities = {key: stored[key] for key in (
            "chain_indices", "residue_indices", "atom_names")}
    from_config = PreprocessConfig()
    sampler = view.sampler(from_config, identities)
    if kind == "phase":
        assert sampler.atom_phase_offsets is not None and sampler.phase_offset == 0
        assert not any(sampler.voxel_offset)
    if kind == "grid":
        assert sampler.atom_phase_offsets is None and any(sampler.voxel_offset)


def test_new_structural_assets_are_validated_and_selectively_resumable(tmp_path, pdb_path):
    """Exact immutable old geometry survives CA/environment enrichment and a second pass."""
    base = _complete_base(tmp_path, pdb_path)
    original = base.read_bytes()
    producer = StructuralSidecars(residue_graph=True, residue_ca_radius=12,
        surface_environment=True, environment_options={"spacing": 2., "rays": 16,
                                                      "hull_directions": 24})
    work = _work(tmp_path / "run")
    assets = producer.generate(work, {"protein": base}, 1)["protein"]
    StructuralSidecars.validate_assets(base, assets)
    assert producer.generate(work, {"protein": base}, 1)["protein"] == assets
    assert base.read_bytes() == original


def test_slae_adapter_aligns_physical_atoms_and_masks_unsupported_entities():
    """A stub at the external API boundary validates ordering, not pretrained scientific quality."""
    extractor = object.__new__(SLAEExtraction)
    extractor.names = ("N", "CA", "C", "O", "CB")
    extractor.residue_names = {"ALA"}
    extractor.device = torch.device("cpu")
    extractor.maximum_edges, extractor.radius = 100, 8.
    def data_type(**values):
        return SimpleNamespace(**values, to=lambda _: SimpleNamespace(**values))
    extractor.data_type = data_type
    extractor.model = None
    def encode(_model, data):
        assert data.residue_index.tolist() == [0, 0, 1, 1]
        return {"node_embedding": data.pos[:, :1].expand(-1, 128),
                "residue_embedding": torch.ones(2, 128)}
    extractor.encode_api = encode
    arrays = {"atom_positions": np.array([[4., 0, 0], [1., 0, 0], [2., 0, 0],
                                          [3., 0, 0], [7., 0, 0]], np.float32),
        "residue_indices": np.array([1, 0, 0, 1, 2]),
        "atom_names": np.array(["CA", "CA", "N", "N", "ZN"]),
        "residue_names": np.array(["ALA", "ALA", "ALA", "ALA", "ZN"]),
        "atomic_numbers": np.array([6, 6, 7, 7, 30])}
    result = extractor.encode(arrays)
    np.testing.assert_equal(result["embedding"][:4, 0], arrays["atom_positions"][:4, 0])
    assert result["valid"].tolist() == [True, True, True, True, False]
    assert not result["embedding"][4].any()
    unknown = extractor.encode({key: value[-1:] for key, value in arrays.items()})
    assert not unknown["valid"].any() and not unknown["residue_valid"].any()
    assert not unknown["embedding"].any()


def test_empty_cavity_path_is_not_replaced_with_exterior_zero():
    """Six overlapping expanded spheres enclose a free central cell with no exterior route."""
    atoms = np.concatenate((np.eye(3)*3., -np.eye(3)*3.))
    domain = AccessibleSpace(atoms, np.full(6, 1.1), probe=1.4, spacing=.5,
                             maximum_voxels=100000, hull_directions=24)
    paths = domain.paths()
    nearest = np.linalg.norm(domain.points, axis=1).argmin()
    assert domain.free[nearest]
    assert not np.isfinite(paths[nearest])


def test_frozen_geometry_protocols_never_read_task_targets():
    """Explicit V12 protocol choices are validated before any scientific record is evaluated."""
    assert SurfaceMapAudit(protocols=["phase", "grid"]).protocols == ("phase", "grid")
    with pytest.raises(ValueError):
        SurfaceMapAudit(protocols=["perturb_atoms_randomly"])


def test_frozen_protocols_regenerate_complete_views_and_resume(tmp_path, pdb_path):
    """A real forward audits isolated geometry without fitting or copying invalid old labels."""
    base = _complete_base(tmp_path, pdb_path)
    manifest = tmp_path / "members.csv"
    manifest.write_text("file,label,split\ntiny.npz,1,val\n")
    sample = dict(WisdomDataset(manifest, "val", include_surface_geometry=True,
                                include_atom_geometry=True)[0])
    sample["surface_target_soft"] = torch.tensor([float("nan")])
    sample["surface_target_hard"] = torch.tensor([float("nan")])
    predictor = model("baseline").eval()
    work = _work(tmp_path / "run")
    work.cache = work.checkpoints
    audit = SurfaceMapAudit(protocols=["identity", "rigid_tensor", "phase", "grid",
                                      "rigid_rebuild", "resolution_0.8", "resolution_1.25"],
                            view_seeds=[607])
    original = base.read_bytes()
    result = audit.evaluate(predictor, sample, WisdomCollator(), torch.device("cpu"), work, base)
    assert result["rigid_tensor_607_index_max_logit_error"] < 1e-5
    assert result["identity_607_reference_area_coverage"] == 1
    assert result["identity_607_constant_reference_logit_mae"] == 0
    assert result["phase_607_reference_supported_points"] > 0
    assert result["protocol"].endswith("no-local-GT")
    assert base.read_bytes() == original
    again = audit.evaluate(predictor, sample, WisdomCollator(), torch.device("cpu"), work, base)
    assert result == again


def test_point_bound_residue_candidates_cannot_leak_into_a_regenerated_view(tmp_path):
    """Parallel candidates need their own rebuilt asset, even if a view has the same row count."""
    predictor = SimpleNamespace(atomic_representation=SimpleNamespace(
        residue_encoder=SimpleNamespace(topology="parallel")))
    with pytest.raises(ValueError, match="view-bound residue"):
        SurfaceMapAudit(protocols=["phase"]).protocol_audit(
            predictor, {}, WisdomCollator(), torch.device("cpu"),
            _work(tmp_path), tmp_path / "unopened.npz")


def test_slae_asset_validator_checks_atom_slots_not_only_row_counts(tmp_path, pdb_path):
    """A finite rectangular array is insufficient without exact physical names and masks."""
    base = _complete_base(tmp_path, pdb_path)
    with np.load(base, allow_pickle=False) as stored:
        parent = stored["residue_indices"]
        names = stored["atom_names"].astype(str)
    vocabulary = AtomicProteinExtractor.ATOM_NAMES
    arrays = {"embedding": np.zeros((len(parent), 128), np.float32),
        "valid": np.ones(len(parent), bool),
        "residue_embedding": np.zeros((int(parent.max())+1, 128), np.float32),
        "residue_valid": np.ones(int(parent.max())+1, bool),
        "atom_parent": parent,
        "atom_slot": np.asarray([vocabulary.index(name) for name in names], np.int16)}
    parameters = {"checkpoint_sha256": AtomicProteinExtractor.CHECKPOINT_SHA256,
                  "revision": AtomicProteinExtractor.REVISION}
    path = tmp_path / "atom-features.npz"
    StructuralAsset.write(path, base, arrays, AtomicProteinExtractor.FAMILY, parameters)
    assert AtomicProteinExtractor.validate(path, base)
    arrays["atom_slot"][0] = (arrays["atom_slot"][0]+1) % len(vocabulary)
    StructuralAsset.write(path, base, arrays, AtomicProteinExtractor.FAMILY, parameters)
    with pytest.raises(ValueError, match="slots"):
        AtomicProteinExtractor.validate(path, base)


@pytest.mark.parametrize("context", [None, "esm2"])
@pytest.mark.parametrize("no_messages", [False, True])
def test_frozen_atomic_features_have_masks_gradients_and_checkpoint_restore(context, no_messages):
    """Real WISDOM forwards consume frozen atom channels, not a resized residue embedding."""
    options = {"sources": ["atom_name"], "atomic_external_model": "slae",
               "atomic_context_dim": 4, "atomic_message_control": no_messages}
    if context:
        options.update(external_model=context, external_mode="context", context_dim=4)
    predictor = model("baseline", atomic_encoder_type="schnet",
                      atomic_representation=options).eval()
    data = chemical_batch()
    data["structural_assets"]["external_atomic_slae"] = {"members": [
        {"embedding": torch.arange(n*128).reshape(n, 128).float(),
         "valid": torch.arange(n) != n-1} for n in (3, 4)]}
    audit = predictor.atomic_representation.fit([data])
    assert audit["external_atom"]["count"] == 5
    output = predictor(**_model_inputs(data))
    output["logits"].sum().backward()
    projection = predictor.atomic_representation.atomic_context_projection
    assert projection.weight.grad.abs().sum() > 0
    predictor.eval()
    restored = copy.deepcopy(predictor)
    restored.load_state_dict(predictor.state_dict(), strict=True)
    torch.testing.assert_close(predictor(**_model_inputs(data))["logits"],
                               restored(**_model_inputs(data))["logits"], rtol=0, atol=0)


def test_v12_to_v15_use_four_explicit_paired_model_seeds():
    """Each finite experiment repeats all alternatives on four fixed model replications."""
    root = Path(__file__).parents[1] / "experiments"
    for stage in (12, 13, 14, 15):
        for path in (root / f"v{stage}").glob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            if stage == 12:
                policy = config["with"]["seed_selection"]
                assert policy["mode"] == "explicit"
                seeds = policy["seeds"]
                assert seeds == [211, 223, 227, 229]
                assert "seeds" not in config  # No new review jobs per model seed.
            else:
                seeds = config["seeds"]
                assert "seed" not in config["with"]
                assert "seed" not in config.get("sweep", {}).get("space", {})
            assert len(seeds) == len(set(seeds)) == 4, path
