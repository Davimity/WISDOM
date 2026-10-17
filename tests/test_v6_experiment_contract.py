"""Protect V6 controls, selective extensions and the exploratory mixture contract."""

# ruff: noqa: I001

import ast
import torch
import pytest
import inspect
import hashlib
import itertools

from pathlib import Path
from yaml import safe_load
from test_wisdom_v2 import _v2
from test_wisdom_v1 import _sample
from torch.nn import functional as F
from lambdaforge.work import WorkConfig, WorkRunner
from wisdom.Training import _model_inputs, _train_wisdom
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.models.WeakSurfaceLoss import WeakSurfaceLoss


ROOT = Path(__file__).parents[1] / "experiments"
SEEDS = [4, 7, 32, 54]
GRIDS = {
    "a": ("negative_surface_lambda", [0.0, 0.03, 0.1, 0.3, 1.0], 15, 60),
    "b": ("positive_existence_lambda", [0.0, 0.03, 0.1, 0.3], 12, 48),
    "c": ("regional_positive_lambda", [0.0, 0.03, 0.1, 0.3], 12, 48),
    "d": ("regional_ranking_lambda", [0.0, 0.03, 0.1, 0.3], 12, 48),
    "e": ("cardinality_lambda", [0.0, 0.03, 0.1, 0.3], 30, 120),
    "f": ("total_variation_lambda", [0.0, 0.005, 0.01, 0.05], 24, 96),
    "g": ("dirichlet_lambda", [0.0, 0.001, 0.01, 0.05, 0.1], 30, 120),
}
FROZEN = {
    "wisdom_v6b.yaml": "c143aff18d407f77ae872eed793d1c01042b5bbfb4cffdacd9a4ce0181831453",
}
EXTENSIONS = {
    "a": {("log_sum_exp", "heat"): [1.5, 2.0, 3.0, 5.0, 10.0],
          ("attention", "heat"): [1.5, 3.0, 5.0]},
    "c": {("log_sum_exp", "heat"): [0.5, 0.75, 1.0, 2.0],
          ("max", "heat"): [0.5, 0.75, 1.0]},
    "d": {("attention", "heat"): [0.5, 0.75, 1.0, 2.0],
          ("log_sum_exp", "heat"): [0.5, 0.75, 1.0, 2.0]},
    "f": {("log_sum_exp", "none"): [0.1, 0.2, 0.5, 1.0],
          ("max", "heat"): [0.1, 0.2, 0.5]},
}
BACKBONE = (
    "dataset", "model_version", "subset", "initialization_profile", "optimization_profile",
    "hidden_dim", "embedding_dim", "residue_embedding_dim", "atomic_layers",
    "projection_depth", "surface_layers", "atom_spatial_k", "surface_atom_k",
    "diffusion_spectral_modes", "surface_atom_radius", "dropout", "learning_rate",
    "weight_decay", "gate_lambda", "architecture_spike", "head_type", "batch_size",
    "epochs", "patience", "minimum_delta", "precision", "surface_metrics",
    "surface_metrics_interval", "faithfulness_audit", "evaluate_test", "data_workers",
)

WEIGHTS = tuple(row[0] for row in GRIDS.values())


def test_unamended_v6b_remains_byte_identical():
    """Keep the unamended existence screen frozen; V6a now has authorized selective additions."""
    for name, expected in FROZEN.items():
        assert hashlib.sha256((ROOT / "v6" / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("letter", GRIDS)
def test_v6_native_complete_sweeps_preserve_v5c_and_pair_every_cell(letter):
    """Expand LF's actual conditional design and require every paired cell.

    Args:
        letter: Isolated family screen a through g.

    Checks freeze the parent and poolers, remove inactive dimensions, and keep all
    four seeds mandatory. This planning operation never fits models.
    """
    path = ROOT / "v6" / f"wisdom_v6{letter}.yaml"
    authored = safe_load(path.read_text())
    parent_document = safe_load((ROOT / "v5/wisdom_v5c.yaml").read_text())
    fixed = authored["with"]
    for name in BACKBONE:
        assert fixed[name] == parent_document["with"][name], name
    assert fixed["weak_loss_profile"] == "custom"
    assert fixed["evaluate_test"] is False
    assert fixed["visualization"]["mode"] == "none"
    assert authored["seeds"] == SEEDS
    assert "steps" not in authored and "search" not in authored
    assert authored["objective"] == parent_document["objective"]
    # GPU count is an operational request, not an invariant of the scientific loss screen.

    assert authored["resources"]["gpu"] >= 1
    assert {key: value for key, value in authored["resources"].items() if key != "gpu"} == {
        "cpu": 36, "memory": "96GiB", "time": "168h",
    }
    assert authored["execution"] == {"max_time": "168h"}
    if letter not in "fg":
        assert fixed["surface_refiner_type"] == "heat"
        assert fixed["surface_refiner_heat_length"] == 6.0
    if letter in "cd":
        assert fixed["regional_loss_length"] == 2.5
    if letter == "d":
        assert fixed["regional_ranking_margin"] == 0.5
    if letter == "e":
        assert fixed["maximum_positive_area"] is None

    config = WorkConfig.from_yaml(path)
    plan = WorkRunner().plan(config)
    active, lambdas, base_trials, base_runs = GRIDS[letter]
    extension = {(pooling, refiner, coefficient, None)
                 for (pooling, refiner), values in EXTENSIONS.get(letter, {}).items()
                 for coefficient in values}
    trial_count = base_trials + len(extension)
    run_count = base_runs + 4 * len(extension)
    assert len(plan.levels) == 1 and len(plan.levels[0]) == run_count
    assert plan.preflight["studies"][0]["candidates"] == trial_count
    assert plan.preflight["studies"][0]["shared_seeds"] == SEEDS
    for trial in range(1, trial_count + 1):
        runs = [run for run in plan.levels[0] if run["trial"] == trial]
        assert [run["seed"] for run in runs] == SEEDS
        assert all(run["search"]["strategy"] == "fixed" for run in runs)
        assert all(run["search"]["design"]["replication"] == "fixed" for run in runs)
    evidence = plan.levels[0][0]["search"]["design"]["evidence"]
    assert evidence["required_run_count"] == run_count and evidence["optional_run_count"] == 0
    assert all(row["required"] for row in evidence["requirements"])

    cells = set()
    for candidate in config.levels[0].runs[0].variants:
        merged = {**fixed, **candidate}
        assert all(merged[name] == 0 for name in WEIGHTS if name != active)
        pooling = candidate["pooling_type"]
        refiner = merged["surface_refiner_type"]
        minimum = candidate.get("minimum_positive_area")
        cells.add((pooling, refiner, merged[active], minimum))
        if letter in "fg":
            assert (candidate.get("surface_refiner_heat_length") == 6.0) == (refiner == "heat")
            assert ("surface_refiner_heat_length" in candidate) == (refiner == "heat")
        if letter == "e":
            assert ("minimum_positive_area" in candidate) == (merged[active] > 0)
        if pooling == "log_sum_exp":
            assert candidate["log_sum_exp_mode"] == "fixed"
            assert candidate["log_sum_exp_beta"] == 0.25
            assert candidate["pooling_area_mode"] == "area"
            assert not any(name.startswith("attention_") for name in candidate)
        elif pooling == "attention":
            assert candidate["attention_variant"] == "simple"
            assert candidate["attention_hidden_dim"] == 256
            assert candidate["pooling_area_mode"] == "area"
            assert not any(name.startswith("log_sum_exp_") for name in candidate)
        else:
            assert pooling == "max" and merged["pooling_area_mode"] == "point"
            assert not any(name.startswith(("attention_", "log_sum_exp_")) for name in candidate)
            assert "pooling_area_mode" not in candidate

    expected = {
        (pooling, refiner, coefficient, minimum)
        for pooling, refiner, coefficient in itertools.product(
            ("max", "log_sum_exp", "attention"),
            ("none", "heat") if letter in "fg" else ("heat",), lambdas,
        )
        for minimum in (
            (0.005, 0.01, 0.02) if letter == "e" and coefficient > 0 else (None,)
        )
    }
    assert not expected & extension
    assert len(expected) == base_trials
    assert len(cells) == trial_count and cells == expected | extension
    assert len(config.levels[0].runs[0].variants) == len(cells)

    # Removing only include must reconstruct the original, ordered baseline design exactly.

    if extension:
        base_document = {**authored, "sweep": {
            key: value for key, value in authored["sweep"].items() if key != "include"
        }}
        base = WorkConfig.from_mapping(base_document, source=path)
        baseline = base.levels[0].runs[0].variants
        assert len(baseline) == base_trials
        assert list(config.levels[0].runs[0].variants[:base_trials]) == list(baseline)


def test_v6h_sparse_mixtures_keep_matched_controls_and_no_cartesian_explosion():
    """Enumerate exploratory mixtures; preserve the V5c backbone and all core ablations."""
    text = (ROOT / "v6/wisdom_v6h.yaml").read_text()
    authored = safe_load(text)
    assert authored["seeds"] == SEEDS
    assert authored["with"]["evaluate_test"] is False
    assert all(authored["with"][name] == 0 for name in WEIGHTS)
    assert set(WEIGHTS) <= set(authored["sweep"]["space"])
    assert "candidates" in authored["sweep"] and "include" not in authored["sweep"]
    assert "Do not select families by W alone." in text
    parent = safe_load((ROOT / "v5/wisdom_v5c.yaml").read_text())["with"]
    for name in BACKBONE:
        assert authored["with"][name] == parent[name]
    assert authored["with"]["surface_refiner_type"] == "heat"
    assert authored["with"]["surface_refiner_heat_length"] == 6.0
    assert authored["with"]["regional_loss_length"] == 2.5
    assert authored["with"]["regional_ranking_margin"] == 0.5
    config = WorkConfig.from_yaml(ROOT / "v6/wisdom_v6h.yaml")
    candidates = config.levels[0].runs[0].variants
    assert len(candidates) == 54
    assert len({repr(sorted(row.items())) for row in candidates}) == 54
    core = {"negative_surface_lambda", "positive_existence_lambda", "regional_ranking_lambda"}
    expected = {frozenset(subset) for size in range(4)
                for subset in itertools.combinations(core, size)}
    anchors = {
        "max": (0.3, 0.03, 0.3, 0.1, 0.03, 0.005, 0.001),
        "log_sum_exp": (1.0, 0.03, 0.1, 0.03, 0.1, 0.005, 0.001),
        "attention": (0.03, 0.03, 0.03, 0.03, 0.03, 0.005, 0.001),
    }
    for pooling in anchors:
        rows = [row for row in candidates if row["pooling_type"] == pooling]
        assert len(rows) == 18
        active = [frozenset(name for name in WEIGHTS if row[name] > 0) for row in rows]
        assert expected <= set(active)
        assert all(frozenset([name]) in active for name in WEIGHTS)
        for smoothness in ("total_variation_lambda", "dirichlet_lambda"):
            assert frozenset(set(WEIGHTS) - (
                {"dirichlet_lambda"} if smoothness == "total_variation_lambda"
                else {"total_variation_lambda"}
            )) in active
        for row in rows:
            assert not (row["total_variation_lambda"] and row["dirichlet_lambda"])
            assert ("minimum_positive_area" in row) == (row["cardinality_lambda"] > 0)
            if row["cardinality_lambda"]:
                assert row["minimum_positive_area"] == 0.02
            for name, value in zip(WEIGHTS, anchors[pooling], strict=True):
                assert row[name] in (0.0, value)
            WeakSurfaceLoss(
                negative_weight=row[WEIGHTS[0]], positive_existence_weight=row[WEIGHTS[1]],
                regional_positive_weight=row[WEIGHTS[2]], regional_ranking_weight=row[WEIGHTS[3]],
                cardinality_weight=row[WEIGHTS[4]], total_variation_weight=row[WEIGHTS[5]],
                dirichlet_weight=row[WEIGHTS[6]],
                minimum_positive_area=row.get("minimum_positive_area", 0.0),
            )
    plan = WorkRunner().plan(config)
    assert len(plan.levels[0]) == 216
    assert plan.preflight["studies"][0]["candidates"] == 54
    assert plan.preflight["studies"][0]["shared_seeds"] == SEEDS


def test_v6_inventory_and_inference_only_review_policies():
    """Keep eight reviews with an authored saved seed and automatic diagnostic cases."""
    names = {f"wisdom_v6{letter}.yaml" for letter in "abcdefgh"}
    assert {p.name for p in (ROOT / "v6").glob("wisdom_*.yaml")} == names
    assert {p.name for p in (ROOT / "v6/visualization").glob("*.yaml")} == names
    for letter in "abcdefgh":
        path = ROOT / "v6/visualization" / f"wisdom_v6{letter}.yaml"
        authored = safe_load(path.read_text())
        settings = authored["with"]
        assert authored["run"] == "wisdom.visualization.StudyReview.StudyReview"
        assert settings["source_execution"] == f"wisdom-v6{letter}"
        assert settings["source_seed"] == 4
        assert settings["maximum_proteins"] == 8
        assert not {"seed_selection", "protein_selection", "trial_selection",
                    "maximum_surface_points", "report_maximum_mib"} & settings.keys()
        assert authored["resources"]["gpu"] == 1
        assert not any(key in authored for key in ("steps", "search", "sweep", "seeds"))
        assert "REPLACE" not in repr(authored)
        if letter in "fg":
            text = path.read_text()
            assert "surface_refiner_type: [none, heat]" in text
            assert GRIDS[letter][0] in text
        assert len(WorkRunner().plan(WorkConfig.from_yaml(path)).levels[0]) == 1


@pytest.mark.parametrize("pooling", ["max", "log_sum_exp", "attention"])
def test_v6_heat_pooling_and_weak_priors_share_operational_field(pooling):
    """Use the real model and priors: both losses and pooling consume Heat output.

    Args:
        pooling: One retained pooling family. LSE/Attention use their frozen V6 settings.

    Gradients reach the raw local head through Heat; no local target enters the loss.
    """
    torch.manual_seed(7)
    samples = [_sample(3, 4, 1.0), _sample(3, 3, 0.0)]
    batch = dict(WisdomCollator()(samples))
    model = _v2(
        pooling, surface_refiner_type="heat", surface_refiner_heat_length=6.0,
        pooling_area_mode="point" if pooling == "max" else "area",
        log_sum_exp_beta=0.25, attention_hidden_dim=256,
    ).eval()
    heat_outputs, pooled_fields = [], []
    heat_hook = model.surface_refiner.register_forward_hook(
        lambda module, args, result: heat_outputs.append(result)
    )
    pool_hook = model.pooling_head.register_forward_pre_hook(
        lambda module, args: pooled_fields.append(args[0])
    )
    output = model(**_model_inputs(batch))
    heat_hook.remove()
    pool_hook.remove()
    operational = output["surface_logits"]
    assert operational is output["refined_surface_logits"]
    assert operational is heat_outputs[0] and operational is pooled_fields[0]

    # Derive both terms from the operational field, independently of the loss implementation.

    terms = WeakSurfaceLoss(negative_weight=0.1, positive_existence_weight=0.1)(
        operational, batch["surface_area_weights"], batch["surface_batch"],
        batch["target"], batch["surface_operators"], batch["surface_ptr"],
    )
    negative = batch["surface_batch"] == 1
    areas = batch["surface_area_weights"][negative]
    expected_negative = (areas / areas.sum() * F.softplus(operational[negative])).sum()
    expected_existence = F.softplus(-operational[~negative].max())
    assert torch.allclose(terms["total"], 0.1 * (expected_negative + expected_existence))
    assert "surface_target" not in inspect.signature(WeakSurfaceLoss.forward).parameters
    output["raw_surface_logits"].retain_grad()
    terms["total"].backward()
    assert output["raw_surface_logits"].grad.abs().sum() > 0
    assert model.local_head.weight.grad.abs().sum() > 0


def test_training_passes_operational_not_raw_logits_to_weak_loss():
    """Guard the actual Training boundary without running a production fit or mocking its loop."""
    tree = ast.parse(inspect.getsource(_train_wisdom))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "weak_surface_loss"
    ]
    assert len(calls) == 1
    logits = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "logits")
    assert isinstance(logits, ast.Subscript) and isinstance(logits.value, ast.Name)
    assert logits.value.id == "output"
    assert isinstance(logits.slice, ast.Constant) and logits.slice.value == "surface_logits"
