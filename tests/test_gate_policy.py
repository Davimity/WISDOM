"""Controlled V10f contracts: no production training, external data or held-out targets."""

# ruff: noqa: I001

import io
import json
import copy
import yaml
import torch
import pytest
import hashlib

from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from test_wisdom_v1 import _sample, _write_npz
from wisdom.Training import _create_model, _model_inputs, _train_wisdom
from test_representation_campaign import chemical_batch
from wisdom.models.gating.GateGroup import GateGroup
from wisdom.models.gating.GatePolicy import GatePolicy
from wisdom.models.gating.HardConcreteGate import HardConcreteGate


ROOT = Path(__file__).parents[1]
SOURCES = [name.removeprefix("atom.") for name in sorted(GatePolicy.CHEMICAL)]


def predictor(policy=None, force="learned", sources=None):
    """Construct the real SchNet/Heat6 path at small latent width and five curvature scales.

    Args:
        policy: Optional grouped constructor mapping; None is the untouched historical path.
        force: Chemical-only learned/open/closed control.
        sources: Selected chemical sources; None retains all seven.

    Returns:
        Synthetic-data-compatible V2 model with the real 41+7 semantic inventory.
    """
    model, _ = _create_model(2, {
        "hidden_dim": 8, "embedding_dim": 4, "residue_embedding_dim": 4,
        "atomic_hidden_dim": 4, "atomic_output_dim": 8, "atomic_layers": 2,
        "projection_depth": 1, "surface_layers": 2, "dropout": 0.,
        "curvature_features": 15, "architecture_implementation": "controlled",
        "gate_cost_normalization": "baseline_count", "atomic_encoder_type": "schnet",
        "surface_refiner_type": "heat", "surface_refiner_heat_length": 6.,
        "pooling_type": "max", "head_type": "single",
        "atomic_representation": {"sources": SOURCES if sources is None else sources,
                                  "force_sources": force},
        **({"gate_policy": policy} if policy is not None else {}),
    })
    return model


def data():
    """Extend only synthetic curvature scales; do not read real structures or labels.

    Returns:
        Disjoint two-protein chemical batch with five identical synthetic curvature scales.
    """
    result = chemical_batch()
    curvature = result["surface_curvatures"]
    if curvature.ndim == 2:
        result["surface_curvatures"] = curvature[:, :3].repeat(1, 5)
    else:
        result["surface_curvatures"] = curvature[:, :1].repeat(1, 5, 1)
    return result


def test_partition_cost_and_uniform_value_gradient_equivalence():
    """The reference preserves exact old arithmetic and gradients, including ancestor costs."""
    model = predictor({})
    registry = model.semantic_gates
    assert model.baseline_gate_count == 41 and len(registry.names) == 48
    assert Counter(GatePolicy.group(name) for name in registry.names) == {
        GateGroup.SURFACE: 20, GateGroup.CHEMICAL: 7, GateGroup.OTHER: 21}
    assert GatePolicy.group("surface.atom_context") is GateGroup.OTHER
    expected = model.gate_regularization() * .04
    actual = model.gate_policy.penalty(registry)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    params = tuple(registry.parameters())
    old = torch.autograd.grad(expected, params, retain_graph=True)
    new = torch.autograd.grad(actual, params)
    for left, right in zip(old, new, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    contributions = model.gate_policy.contributions(registry)
    torch.testing.assert_close(sum(contributions.values()), actual)
    # Parent cost is counted once directly; descendants additionally require that parent active.
    assert float(contributions[GateGroup.CHEMICAL].detach()) == pytest.approx(7 * .95**2 * .04 / 41)


def test_group_initialization_does_not_consume_shared_rng():
    """Changing only initial gate probabilities leaves every other weight bitwise paired."""
    torch.manual_seed(269)
    old = predictor().state_dict()
    torch.manual_seed(269)
    new = predictor({"initial_active_surface": .5, "initial_active_chemical": .5})
    for name, tensor in old.items():
        if not name.startswith("semantic_gates."):
            assert torch.equal(tensor, new.state_dict()[name]), name
    for name in new.semantic_gates.names:
        group = GatePolicy.group(name)
        probability = new.semantic_gates.probability_active(name)
        assert float(probability.detach()) == pytest.approx(.95 if group is GateGroup.OTHER else .5)
    gate = HardConcreteGate(.5)
    assert float(gate.deterministic_value().detach()) == pytest.approx(.101813, abs=1e-5)


def test_reference_construction_is_bitwise_compatible():
    """An explicitly uniform policy changes no historical state_dict keys or initial tensor."""
    torch.manual_seed(269)
    old = predictor()
    torch.manual_seed(269)
    new = predictor({})
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, tensor in old.state_dict().items():
        torch.testing.assert_close(tensor, new.state_dict()[name], rtol=0, atol=0)


def test_schedule_is_local_and_selected_warmup_restores_exactly():
    """O stays learned/penalized while S/C are forced; selected warm-up is not final state."""
    model = predictor({"lambda_surface": .16, "lambda_chemical": .16,
                       "warmup_fraction": .01, "ramp_fraction": .02})
    policy = model.gate_policy
    registry = model.semantic_gates
    assert policy.schedule(registry, 1, 500) == "all_on"
    assert policy.effective == {GateGroup.SURFACE: 0, GateGroup.CHEMICAL: 0, GateGroup.OTHER: .04}
    values = registry.sample(False)
    assert values["atom.atom_name"] == 1 and values["surface.curvature.mean.scale_0"] == 1
    assert values["surface.atom_context"] < 1
    saved = copy.deepcopy(registry.inference_state())
    assert policy.schedule(registry, 6, 500) == "ramp"
    assert not registry.schedule_open
    policy.schedule(registry, 11, 500)
    assert policy.effective[GateGroup.CHEMICAL] == pytest.approx(.08)
    assert policy.schedule(registry, 16, 500) == "learned"
    assert policy.effective[GateGroup.CHEMICAL] == .16
    registry.restore_inference_state(saved)
    assert registry.inference_state() == saved
    assert registry.deterministic_value("atom.atom_name") == 1


@pytest.mark.parametrize("force", ["open", "closed"])
def test_fixed_chemical_controls_have_no_direct_gradient(force):
    """Only added chemistry is fixed; its parent remains learnable and gets one direct cost."""
    model = predictor({}, force=force)
    policy = model.gate_policy
    registry = model.semantic_gates
    costs = policy.contributions(registry)
    assert costs[GateGroup.CHEMICAL] == 0
    policy.penalty(registry).backward()
    for name, gate in zip(registry.names, registry.gates, strict=True):
        if name in GatePolicy.CHEMICAL:
            assert gate.log_alpha.grad is None
            assert registry.deterministic_value(name) == int(force == "open")
        else:
            assert gate.log_alpha.requires_grad
    assert registry.gates[0].log_alpha.grad.abs() > 0


def test_stochastic_distribution_and_exact_closure_under_gradient():
    """A redundant synthetic source can close; real-campaign closure is not promised."""
    torch.manual_seed(269)
    gate = HardConcreteGate(.5)
    draws = torch.stack([gate().detach() for _ in range(12000)])
    assert float((draws > 0).float().mean()) == pytest.approx(.5, abs=.02)
    assert (draws == 0).any() and (draws == 1).any() and ((draws > 0) & (draws < 1)).any()
    model = predictor({"lambda_chemical": 64.})
    registry = model.semantic_gates
    parameter = registry.gates[registry.names.index("atom.atom_name")].log_alpha
    optimizer = torch.optim.Adam([parameter], lr=.1)
    # The task's useful predictor is independent of this redundant zero-valued source.
    for _ in range(150):
        optimizer.zero_grad()
        task = (torch.tensor(2.) * torch.tensor(3.) - torch.tensor(6.)).square()
        (task + model.gate_policy.penalty(registry)).backward()
        assert torch.isfinite(parameter.grad) and parameter.grad != 0
        optimizer.step()
    assert registry.deterministic_value("atom.atom_name") == 0


def test_closed_blocks_mask_all_information_including_validity_and_bias():
    """Finite feature/mask/embedding perturbations cannot cross a fixed closed source."""
    model = predictor({}, force="closed").eval()
    batch = data()
    representation = model.atomic_representation
    representation.fit([batch])
    values = model.semantic_gates.sample(False)
    original = representation(batch, values)
    mutated = copy.deepcopy(batch)
    for member in mutated["structural_assets"]["atomic_context"]["members"]:
        member["torsion"] += 4
        member["torsion_valid"].logical_not_()
        member["functional_group_bits"].fill_(2047)
        member["chemistry_known"].logical_not_()
    with torch.no_grad():
        for table in representation.tables.values():
            table.weight.fill_(1000)
    assert original.count_nonzero() == 0
    assert representation(mutated, values).count_nonzero() == 0
    first = model(**_model_inputs(batch))
    second = model(**_model_inputs(mutated))
    torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("mixed", [False, True])
def test_complete_schnet_backward_restore_and_protein_independence(mixed):
    """Real sparse backbone, weak-label loss and grouped gradients remain finite on CPU AMP."""
    model = predictor({"lambda_chemical": .16}).eval()
    batch = data()
    model.atomic_representation.fit([batch])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed):
        out = model(**_model_inputs(batch))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], batch["target"])
        loss = loss + model.gate_policy.penalty(model.semantic_gates)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    stream = io.BytesIO()
    torch.save({"state": model.state_dict(), "gates": model.semantic_gates.inference_state()},
               stream)
    stream.seek(0)
    restored = predictor({"lambda_chemical": .16}).eval()
    state = torch.load(stream, weights_only=True)
    restored.load_state_dict(state["state"], strict=True)
    restored.semantic_gates.restore_inference_state(state["gates"])
    reference = model(**_model_inputs(batch))["logits"]
    torch.testing.assert_close(
        restored(**_model_inputs(batch))["logits"], reference, rtol=0, atol=0)
    changed = copy.deepcopy(batch)
    changed["formal_charges"][3:] += 10
    torch.testing.assert_close(
        model(**_model_inputs(changed))["logits"][0], reference[0], rtol=0, atol=0)


def test_campaign_exact_count_and_frozen_ledger():
    """Exactly 20 profiles retain V10e's executed fixed settings and never request the test."""
    config = yaml.safe_load((ROOT / "experiments/v10/wisdom_v10f.yaml").read_text())
    previous = yaml.safe_load((ROOT / "experiments/v10/wisdom_v10e.yaml").read_text())
    assert config["seeds"] == [269, 271, 277, 281]
    assert len(config["sweep"]["candidates"]) == 20
    assert len({json.dumps(p, sort_keys=True) for p in config["sweep"]["candidates"]}) == 20
    assert config["sweep"]["candidates"].count(config["sweep"]["reference"]) == 1
    assert all(config["with"][key] == value for key, value in previous["with"].items())
    assert config["with"]["evaluate_test"] is False
    assert not {"search", "steps"} & config.keys()
    evidence = json.loads((ROOT / "experiments/v10/gate_study.json").read_text())
    assert evidence["required_run_count"] == 80 and not evidence["optional_run_count"]
    assert evidence["source_checkpoint_contract"]["full_gates"] == 48
    assert [row["profile"] for row in evidence["candidates"]] == config["sweep"]["candidates"]
    for letter in "abcde":
        path = ROOT / f"experiments/v10/wisdom_v10{letter}.yaml"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == evidence["frozen_yaml_sha256"][path.name]


def test_training_scalar_controls_audit_and_early_selected_override(tmp_path, monkeypatch):
    """Two synthetic epochs exercise actual Training wiring, with no test split or real data.

    Args:
        tmp_path: Disposable fixture data, checkpoints and output directory.
        monkeypatch: Pins CPU and a known decreasing G curve to select the warm-up checkpoint.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # A fixed synthetic G trajectory checks restoration even when the best epoch is all-on.
    utility = iter([.8, .7])
    monkeypatch.setattr("wisdom.Training._validation_utility", lambda metrics: next(utility, .8))
    lines = ["file,label,split"]
    for index, target in enumerate([1., 0.]):
        sample = _sample(3, 5, target)
        sample["surface_curvatures"] = sample["surface_curvatures"][:, :1].repeat(1, 5, 1)
        _write_npz(tmp_path / f"p{index}.npz", sample)
        for split in ("train", "val"):
            lines.append(f"p{index}.npz,{int(target)},{split}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n")
    metrics = []

    def directory(name, **kwargs):
        """Declare a disposable output path without any external LF registry mutation.

        Args:
            name: Artifact's logical name.
            kwargs: Public output attributes ignored by this filesystem-only test fixture.

        Returns:
            Created path inside pytest's temporary directory.
        """
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        return path

    work = SimpleNamespace(
        run_dir=tmp_path, trial=None, stop_requested=False,
        outputs=SimpleNamespace(directory=directory, artifact=lambda *args, **kwargs: None),
        metrics=SimpleNamespace(log=lambda *args, **kwargs: metrics.append((args, kwargs))),
        log=lambda message: None, progress=SimpleNamespace(update=lambda **kwargs: None),
    )
    report = _train_wisdom(
        work, manifest, model_version=2, hidden_dim=8, embedding_dim=4,
        atomic_hidden_dim=4, atomic_output_dim=8, atomic_encoder_type="schnet",
        architecture_implementation="controlled", gate_cost_normalization="baseline_count",
        gate_lambda_surface=.16, gate_lambda_chemical=.16,
        gate_group_warmup_fraction=.5, epochs=2, patience=2,
        surface_refiner_type="heat", surface_refiner_heat_length=6., batch_size=2,
        regional_positive_lambda=.3,
        surface_metrics=False, evaluate_test=False, data_workers=0, dropout=0.,
        save_initial_checkpoint=True,
    )
    audit = tmp_path / "gate-audit"
    selected = json.loads((audit / "selected.json").read_text())
    final = json.loads((audit / "final.json").read_text())
    assert report["best_epoch"] == selected["epoch"] == 1
    assert len(selected["inference_state"]["schedule_open"]) == 20
    assert not final["inference_state"]["schedule_open"]
    assert selected["groups"]["surface"]["coefficient"] == 0
    assert selected["groups"]["other"]["coefficient"] == .04
    epochs = [json.loads(line) for line in (audit / "epochs.jsonl").read_text().splitlines()]
    assert len(epochs) == 2 and all(len(row["gates"]) == 41 for row in epochs)
    assert all("gradient_rms" in row for row in epochs[1]["gates"])
    state = torch.load(tmp_path / "best-model.pt", weights_only=True)
    assert state["model_parameters"]["gate_policy"]["lambda_surface"] == .16
    assert state["inference_state"]["semantic_gate_state"] == selected["inference_state"]
    initial = torch.load(tmp_path / "initial-model.pt", weights_only=True)
    assert initial["inference_state"]["semantic_gate_state"] == selected["inference_state"]
    assert not any(kwargs.get("split") == "test" for _, kwargs in metrics)
