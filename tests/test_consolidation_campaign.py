"""Native planning and controlled scientific contracts for V16 and VN01; no Studies launched."""

from pathlib import Path

import pytest
import yaml
from lambdaforge.work import WorkConfig, WorkRunner

from wisdom.models.atomic.representation.AtomicRepresentation import AtomicRepresentation

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "v16/wisdom_v16a.yaml": 8,
    "v16/wisdom_v16b.yaml": 128,
    "v16/wisdom_v16c.yaml": 8,
    "v16/wisdom_v16d.yaml": 4,
    "vn1/wisdom_vn1.yaml": 32,
    "vn1/wisdom_vn1a.yaml": 32,
    "vn1/wisdom_vn1b.yaml": 16,
    "vn1/wisdom_vn1c.yaml": 8,
}


@pytest.mark.parametrize("relative,expected", FILES.items())
def test_campaign_native_plan_preserves_complete_seeds_and_test_seal(tmp_path, relative, expected):
    """Rebind only typed fixture input so a fresh checkout does not need Registry publications."""
    document = yaml.safe_load((ROOT / "experiments" / relative).read_text())
    assert document["run"] == "wisdom.Training.Training" and "steps" not in document
    assert len(document["seeds"]) == len(set(document["seeds"])) == 4
    gpu_count = 1 if relative in {"vn1/wisdom_vn1.yaml", "vn1/wisdom_vn1a.yaml"} else 2
    assert document["resources"] == {
        "gpu": gpu_count, "cpu": 36, "memory": "96GiB", "time": "168h"
    }
    assert document["with"]["evaluate_test"] is False
    if relative == "v16/wisdom_v16b.yaml":
        assert document["execution"] == {"runs_per_gpu": "auto", "max_parallel": "auto"}
    else:
        assert "execution" not in document
    reference = document["with"]
    assert reference["pooling_type"] == "max"
    assert reference["surface_refiner_heat_length"] == 6
    assert reference["transfer_options"] == {"message": "low_rank", "rank": 16}
    assert "residue_graph" not in reference["atomic_representation"]
    assert "environment" not in reference["surface_options"]
    fixture = tmp_path / "typed-input.txt"
    fixture.write_text("Planning fixture only; not training data or a registered DatasetVersion.")
    document["with"]["dataset"] = {"file": str(fixture)}
    path = tmp_path / Path(relative).name
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    plan = WorkRunner().plan(WorkConfig.from_yaml(path))
    assert plan.preflight["required_runs"] == expected


def test_vn1_factorial_means_exactly_the_eight_report_cells():
    """R- removes only the task prior; N2 cannot silently change model sources or loss intensity."""
    source = yaml.safe_load((ROOT / "experiments/vn1/wisdom_vn1.yaml").read_text())
    space = source["sweep"]["space"]
    assert set(space) == {"regional_positive_lambda", "negative_mining"}
    assert space["regional_positive_lambda"]["values"] == [.3, 0]
    variants = space["negative_mining"]["values"]
    assert variants[0] == {"mode": "none"}
    for variant in variants[1:]:
        assert (variant["lower"], variant["upper"], variant["regions"]) == (.25, .4, 3)
        assert variant["strong_weight"] == .1
    assert [v.get("memory", False) for v in variants] == [False, False, True, True]
    assert [bool(v.get("diversity")) for v in variants] == [False, False, False, True]
    assert "calibration_dataset" not in source["with"]


def test_vn1a_retains_the_same_no_llm_matrix_as_vn1_under_its_existing_name():
    """Retain historical Study names without pretending identical recipes contrast ESM usage."""
    baseline = yaml.safe_load((ROOT / "experiments/vn1/wisdom_vn1.yaml").read_text())
    ablation = yaml.safe_load((ROOT / "experiments/vn1/wisdom_vn1a.yaml").read_text())
    assert baseline["name"] == "wisdom-vn1"
    assert ablation["name"] == "wisdom-vn1a"
    assert baseline["with"]["atomic_representation"]["external_model"] is None
    assert baseline["with"]["atomic_representation"]["external_mode"] == "none"
    representation = ablation["with"]["atomic_representation"]
    assert representation["external_model"] is None
    assert representation["external_mode"] == "none"
    assert representation["sources"] == ["atom_name", "residue_atom"]
    assert AtomicRepresentation.required_assets(representation) == ("atomic_context",)
    module = AtomicRepresentation(**representation)
    assert module.context_projection is None and not module.contextual
    assert module.atomic_context_projection is None
    assert module.width == 16  # Two learned 8-wide identity tables, not frozen protein features.
    assert {name for name, _ in module.gate_definitions} == {
        "atom.atom_name", "atom.residue_atom"
    }
    ablation["name"] = baseline["name"]
    assert ablation == baseline


def test_vn1_renamed_pilot_and_controls_keep_their_scientific_roles():
    """The no-LLM pilot and secondary controls retain their distinct mining roles and seeds."""
    pilot = yaml.safe_load((ROOT / "experiments/vn1/wisdom_vn1b.yaml").read_text())
    controls = yaml.safe_load((ROOT / "experiments/vn1/wisdom_vn1c.yaml").read_text())
    assert pilot["name"] == "wisdom-vn1b" and pilot["with"]["epochs"] == 10
    assert controls["name"] == "wisdom-vn1c" and controls["with"]["epochs"] == 500
    assert pilot["seeds"] == controls["seeds"] == [1103, 1109, 1117, 1123]
    for document in (pilot, controls):
        assert document["with"]["atomic_representation"]["external_model"] is None
        assert document["with"]["atomic_representation"]["external_mode"] == "none"
    assert [row["mode"] for row in pilot["sweep"]["space"]["negative_mining"]["values"]] == [
        "audit", "audit", "audit", "regional"
    ]
    assert [row["mode"] for row in controls["sweep"]["space"]["negative_mining"]["values"]] == [
        "point", "distributed"
    ]


@pytest.mark.parametrize("relative,expected", {
    "vn1/wisdom_vn1.yaml": 32, "vn1/wisdom_vn1a.yaml": 32,
    "vn1/wisdom_vn1b.yaml": 16, "vn1/wisdom_vn1c.yaml": 8,
    "vf/wisdom_vf1.yaml": 32, "vf/wisdom_vf_joint.yaml": 32,
}.items())
def test_all_vn_vf_recipes_disable_frozen_descriptors_in_every_candidate(
    tmp_path, relative, expected
):
    """Plan every native sweep without checkpoint assets or constructing an external branch."""
    document = yaml.safe_load((ROOT / "experiments" / relative).read_text())
    options = document["with"]["atomic_representation"]
    assert options["external_model"] is None and options["external_mode"] == "none"
    assert options["atomic_external_model"] is None
    assert document["with"].get("external_context_dim") is None
    assert AtomicRepresentation.required_assets(options) == ("atomic_context",)
    representation = AtomicRepresentation(**options)
    assert not representation.contextual and representation.context_projection is None
    assert representation.atomic_context_projection is None and representation.width == 16
    for parameter in ("atomic_representation", "external_context_dim", "atomic_sources"):
        assert parameter not in document["sweep"]["space"]
    fixture = tmp_path / "planning-input.txt"
    fixture.write_text("Planning-only input; not a scientific dataset or Registry publication.")
    document["with"]["dataset"] = {"file": str(fixture)}
    path = tmp_path / Path(relative).name
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    plan = WorkRunner().plan(WorkConfig.from_yaml(path))
    assert plan.preflight["required_runs"] == expected


def test_v16_confirmation_seeds_are_reserved_and_hpo_uses_explicit_native_budgets():
    """Keep confirmation seeds independent and distinguish adaptive pruning from short fidelity."""
    stages = [yaml.safe_load((ROOT / f"experiments/v16/wisdom_v16{x}.yaml").read_text())
              for x in "abcd"]
    assert stages[0]["seeds"] == stages[1]["seeds"]
    assert not set(stages[2]["seeds"]) & set(stages[0]["seeds"])
    assert not set(stages[3]["seeds"]) & set(stages[2]["seeds"] + stages[0]["seeds"])
    search = stages[1]["search"]
    assert search["strategy"] == "adaptive"
    assert search["budget"] == {"candidates": 128, "runs": 300, "time": "168h"}
    assert search["replication"] == {"minimum": 1, "confirmation": [1091, 1093, 1097]}
    assert not set(search["replication"]["confirmation"]) & {
        seed for stage in stages for seed in stage["seeds"]
    }
    assert search["confirmation_top_k"] == 3
    assert search["pruning"] == {
        "enabled": True, "min_step": 12, "confirmations": 3, "probability_threshold": .02
    }
    assert "fidelity" not in search
    assert set(search["space"]) == {
        "atomic_layers", "projection_depth", "surface_layers", "dropout", "learning_rate",
        "weight_decay",
    }


def test_v16b_disables_pretrained_descriptors_without_changing_compact_atomic_width():
    """No context scalar override may reactivate or require the removed frozen ESM branch."""
    document = yaml.safe_load((ROOT / "experiments/v16/wisdom_v16b.yaml").read_text())
    reference = document["with"]
    options = reference["atomic_representation"]
    assert options == {
        "sources": ["atom_name", "residue_atom"], "identity_dim": 8, "force_sources": "learned",
        "external_model": None, "external_mode": "none", "context_dim": 16,
    }
    assert reference["external_context_dim"] is None
    assert AtomicRepresentation.required_assets(options) == ("atomic_context",)
    module = AtomicRepresentation(**options)
    assert module.context_projection is None and not module.contextual
    assert module.atomic_context_projection is None
    assert module.width == 16
    assert reference["atomic_hidden_dim"] == 64
    assert reference["atomic_output_dim"] == reference["hidden_dim"] == 128
    assert reference["surface_metrics_interval"] == 1
    assert reference["evaluate_test"] is False
    assert document["seeds"] == [1009, 1013, 1019, 1021]
