"""Verify domain declarations against LambdaForge's real, read-only analysis API."""

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from lambdaforge.analysis import AnalysisProfile
from lambdaforge.analysis.MetricCatalog import MetricCatalog
from lambdaforge.analysis.Report import write_html
from lambdaforge.analysis.ResearchAnalysis import ResearchAnalysis
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.runtime import MetricCollection

from wisdom.analysis.WisdomAnalysisProfile import WisdomAnalysisProfile
from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite
from wisdom.evaluation.OptimizationDiagnostics import OptimizationDiagnostics
from wisdom.evaluation.SubgroupMetricSuite import SubgroupMetricSuite
from wisdom.evaluation.SurfaceFaithfulnessAudit import SurfaceFaithfulnessAudit
from wisdom.evaluation.SurfaceMetricSuite import SurfaceMetricSuite
from wisdom.evaluation.ViewConsistencyMetricSuite import ViewConsistencyMetricSuite
from wisdom.models.WisdomV2 import WisdomV2
from wisdom.Training import Training, _surface_coupling


def _catalog(*names: str) -> dict[str, Any]:
    """Resolve observed names with the same frozen declarations used by Training.

    Args:
        *names: Actual flat metric keys produced by split-prefixed logging.

    Returns:
        Framework-resolved catalog, including family membership and transitive lineage.
    """
    profile = AnalysisProfile.resolve(Training.analysis_profile)
    return MetricCatalog.resolve(profile.document, set(names))


def _candidates(count: int = 12) -> list[dict[str, Any]]:
    """Produce completed comparable evidence with deliberately inverted global/local quality.

    Args:
        count: Number of synthetic candidates; no training or target fitting is performed.

    Returns:
        Native research candidate rows, each supported by two completed screening seeds.
    """
    rows = []
    for index in range(count):
        quality = (index + 1) / (count + 1)
        values = {
            "val_wisdom_hpo_global": quality,
            "val_wisdom_hpo_surface": 1 - quality,
            "val_wisdom_hpo_coupling": 0.4 + quality / 10,
            "val_wisdom_hpo_score": 0.6 - quality / 10,
            "val_global_surface_spearman": -quality,
            "val_surface_selection_regret": quality / 5,
            "val_surface_positive_macro_auprc": 1 - quality,
            "val_protein_global_score": quality,
            "val_surface_valid_points": 100.0,
            "train_cuda_peak_gib": 2 + quality,
            "test_auprc": quality,
        }
        rows.append(
            {
                "trial": index + 1,
                "parameters": {"pooling_type": "max" if index % 2 else "mean"},
                "mean": values["val_wisdom_hpo_score"],
                "n": 2,
                "diagnostic_metrics": {name: {"mean": value} for name, value in values.items()},
                "runs": [
                    {
                        "seed": seed,
                        "state": "succeeded",
                        "final_objective": values["val_wisdom_hpo_score"],
                        "metrics": values,
                    }
                    for seed in (4, 7)
                ],
            }
        )
    return rows


def _run_profile_fixture(self: Training) -> dict[str, float]:
    """Log synthetic evidence without constructing a model, dataset or optimizer.

    Args:
        self: Native bound Training Work resolved before this callable starts.

    Returns:
        One synthetic scalar, only for an isolated framework lifecycle test.
    """
    self.metrics.log("wisdom_hpo_score", 0.5, split="val")
    return {"synthetic_score": 0.5}


def test_profile_is_native_stable_and_immutable() -> None:
    """Class defaults resolve before execution; metadata overrides change only semantics."""
    first = AnalysisProfile.resolve(Training.analysis_profile)
    again = AnalysisProfile.resolve(WisdomAnalysisProfile().build())
    override = AnalysisProfile.resolve(
        Training.analysis_profile,
        {"metrics": {"val_surface_positive_macro_auprc": {"priority": 99}}},
    )

    assert first.identity == again.identity
    assert first.identity != override.identity
    assert first.document["questions"] == override.document["questions"]
    assert (
        override.document["metrics"]["val_surface_positive_macro_auprc"]["aggregation"]
        == "selected_epoch"
    )
    with pytest.raises(TypeError):
        first.document["metrics"]["val_auprc"]["priority"] = 0


def test_actual_metric_producers_have_scientific_metadata() -> None:
    """All current evaluation outputs have explicit meaning, not a fallback name guess."""
    probability = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    local_labels = torch.tensor([1, 0] * 4)
    owners = torch.arange(4).repeat_interleave(2)
    validity = torch.ones(8, dtype=torch.bool)
    protein_labels = torch.tensor([1, 1, 0, 0])
    protein_probability = torch.tensor([0.9, 0.8, 0.2, 0.1])
    counts = torch.tensor([100, 200, 300, 400])

    binary = BinaryMetricSuite().compute(protein_probability, protein_labels)
    surface = SurfaceMetricSuite().compute(
        probability,
        local_labels,
        validity,
        owners,
        protein_labels,
        torch.ones(8),
        protein_probability,
        probability,
        torch.zeros(4),
    )
    subgroup = SubgroupMetricSuite().compute(
        protein_probability,
        protein_labels,
        counts,
        torch.full((4,), 2),
        probability,
        local_labels,
        validity,
        owners,
        ["G001"] * 4,
        ["I001"] * 4,
        ["core"] * 4,
    )
    view = ViewConsistencyMetricSuite().compute(
        probability,
        probability,
        {
            "reference_indices": torch.arange(8),
            "view_indices": torch.arange(8),
        },
    )
    coupling = _surface_coupling(
        [
            {"epoch": float(epoch), "global": epoch / 10, "surface": 1 - epoch / 10}
            for epoch in range(1, 7)
        ]
    )
    for split in ("val", "test"):
        names = {
            f"{split}_{name}" for output in (binary, surface, subgroup, view) for name in output
        }
        if split == "val":
            names.update(f"val_{name}" for name in coupling)
        catalog = _catalog(*names)["metrics"]
        for name in names:
            assert not catalog[name]["description"].startswith("Unclassified"), name
            assert catalog[name]["split"] == ("validation" if split == "val" else "test")
            assert catalog[name]["source"] in {"declared", "pattern"}


def test_families_aliases_and_lineage_are_unambiguous() -> None:
    """Numeric coordinates and identity duplicates stay distinct from independent discoveries."""
    catalog = _catalog()
    metrics = catalog["metrics"]
    assert catalog["families"]["val_surface_top_recall"]["dimensions"]["fraction"]["values"] == [
        0.05,
        0.1,
        0.25,
    ]
    assert metrics["val_faithfulness_deletion_top_10"]["dimensions"] == {
        "fraction": 0.1,
        "strategy": "top",
    }
    assert metrics["val_protein_global_score"]["derived_from"] == ["val_wisdom_hpo_global"]
    assert not metrics["val_protein_global_score"]["discovery"]
    assert metrics["val_view_map_l1"]["derived_from"] == ["val_view_map_total_variation"]
    assert "surface correlation" in metrics["val_global_surface_spearman"]["aliases"]
    assert "gpu memory" in metrics["train_cuda_peak_gib"]["aliases"]
    assert (
        metrics["train_train_seconds"]["description"]
        != metrics["train_epoch_seconds"]["description"]
    )
    assert metrics["train_cuda_peak_gib"]["aggregation"] == "latest"
    assert metrics["resource.peak_vram"]["aggregation"] == "terminal"
    assert metrics["val_surface_valid_points"]["role"] == "support"
    assert not metrics["val_surface_valid_points"]["discovery"]
    assert "val_subgroup_global_phenotype_g001_surface_auprc" not in metrics

    # Known dataset labels can be declared explicitly; unknown strata are never fabricated.

    known = AnalysisProfile.resolve(WisdomAnalysisProfile().build(["g001", "g_noise"], ["i001"]))
    family = MetricCatalog.resolve(known.document, set())["families"][
        "val_subgroup_global_phenotype"
    ]
    assert family["dimensions"]["stratum"]["values"] == ["g001", "g_noise"]


def test_audit_and_optimization_producers_are_separate_from_quality() -> None:
    """Actual intervention and instrumentation keys resolve without inventing observations."""
    model = WisdomV2(
        hidden_dim=4,
        embedding_dim=2,
        atomic_layers=1,
        projection_depth=1,
        surface_layers=1,
        dropout=0,
        curvature_features=6,
    )
    output = {
        "logits": torch.tensor([2.0]),
        "surface_logits": torch.tensor([-1.0, 0.0, 1.0, 2.0]),
        "surface_embeddings": torch.zeros(4, 4),
    }
    batch = {
        "surface_area_weights": torch.ones(4),
        "surface_batch": torch.zeros(4, dtype=torch.long),
        "surface_ptr": torch.tensor([0, 4]),
        "target": torch.ones(1),
    }
    audit, support = SurfaceFaithfulnessAudit().compute(model, output, batch)
    assert support == 1
    monitor = OptimizationDiagnostics(model)
    monitor.begin_epoch()
    monitor.observe_forward(output)
    monitor.observe_backward()
    diagnostics = monitor.metrics()
    monitor.close()
    names = {f"val_{name}" for name in audit} | {f"train_{name}" for name in diagnostics}
    catalog = _catalog(*names)["metrics"]
    for name in names:
        assert not catalog[name]["description"].startswith("Unclassified"), name
        if name.startswith("train_"):
            assert not catalog[name]["discovery"]
            assert catalog[name]["visibility"] == "hidden"


@pytest.mark.parametrize(
    "objective",
    [
        {"metric": "test_auprc", "mode": "max"},
        {"metric": "val_auprc", "mode": "max", "constraints": {"test_auprc": {"min": 0.5}}},
        {"metric": "apparently_validation", "mode": "max"},
    ],
)
def test_test_evidence_cannot_select_or_constrain_hpo(objective: dict[str, Any]) -> None:
    """Native validation rejects direct test and transitive test-derived objectives/constraints.

    Args:
        objective: Unsafe selection configuration; the apparent validation metric derives from test.
    """
    with pytest.raises(ValueError, match="Test-split"):
        AnalysisProfile.resolve(
            Training.analysis_profile,
            {
                "metrics": {
                    "apparently_validation": {"derived_from": ["test_auprc"], "split": "validation"}
                },
            },
            objective=objective,
        )


def test_native_research_detects_inversion_and_keeps_missing_optional_evidence() -> None:
    """Framework discovery, not WISDOM code, finds G/S disagreement and hides count/alias noise."""
    profile = AnalysisProfile.resolve(
        Training.analysis_profile,
        {
            "discovery": {"resamples": 32, "max_pairs": 16},
        },
    )
    research = ResearchAnalysis.compute(
        _candidates(),
        fingerprint="wisdom-semantic-integration",
        semantics=profile.document,
        parameter_names=["pooling_type"],
        status="provisional",
    )
    relation = next(
        item
        for item in research["relationships"]
        if item["x"] == "val_wisdom_hpo_global" and item["y"] == "val_wisdom_hpo_surface"
    )
    assert relation["effect"] < -0.9
    assert relation["expected"] == "positive"
    assert research["metric_profiles"]["val_surface_valid_points"]["constant"]
    assert not any("test_auprc" in (item["x"], item["y"]) for item in research["relationships"])
    assert research["families"]["val_faithfulness_deletion"]["points"][0]["support"] == 0
    assert research["metric_profiles"]["val_view_surface_consistency"]["finite_candidates"] == 0
    assert not any(
        "val_view_surface_consistency" in (item["x"], item["y"])
        for item in research["relationships"]
    )
    json.dumps(research, allow_nan=False)


def test_work_configuration_freezes_class_profile_without_constructing_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native configuration resolution retains the profile before any expensive training starts.

    Args:
        tmp_path: Isolated source location; no production data or Registry publication is used.
        monkeypatch: Temporarily expose a no-data signature; only configuration is resolved.
    """
    monkeypatch.setattr(Training, "run", lambda self: None)
    config = WorkConfig.from_mapping(
        {
            "name": "semantic-contract",
            "run": "wisdom.Training.Training",
            "objective": {"metric": "val_wisdom_hpo_score", "mode": "max"},
            "analysis": {"metrics": {"val_surface_positive_macro_auprc": {"priority": 99}}},
        },
        source=tmp_path / "contract.yaml",
    )
    declaration = config.levels[0].runs[0]
    assert declaration.analysis_semantics["metrics"]["val_wisdom_hpo_score"]["priority"] == 100
    selected = declaration.analysis_semantics["metrics"]["val_surface_positive_macro_auprc"]
    assert selected["priority"] == 99


def test_native_runner_persists_profile_and_native_html_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run only a synthetic Work boundary and render the framework's research browser.

    Args:
        tmp_path: Isolated output root; no production dataset or study is touched.
        monkeypatch: Substitute the expensive Training callable for this lifecycle test.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Training, "run", _run_profile_fixture)
    config = WorkConfig.from_mapping(
        {
            "name": "profile-fixture",
            "run": "wisdom.Training.Training",
            "objective": {"metric": "val_wisdom_hpo_score", "mode": "max"},
        },
        source=tmp_path / "fixture.yaml",
    )
    result = WorkRunner().run(config)
    assert result.status == "succeeded"
    frozen = json.loads((result.execution_dir / "analysis-semantics.json").read_text())
    assert frozen["profile-fixture"]["metrics"]["val_wisdom_hpo_score"]["priority"] == 100

    profile = AnalysisProfile.resolve(
        Training.analysis_profile,
        {
            "discovery": {"resamples": 32, "max_pairs": 16},
        },
    )
    rows = _candidates()
    research = ResearchAnalysis.compute(rows, fingerprint="wisdom-html", semantics=profile.document)
    report = write_html(
        {
            "source": {"status": "final"},
            "objective": {"metric": "val_wisdom_hpo_score", "mode": "max"},
            "search_space": {"pooling_type": {"values": ["max", "mean"]}},
            "candidates": rows,
            "research": research,
        },
        tmp_path / "research.html",
    )
    content = report.read_text()
    assert "surface correlation" in content
    assert "gpu memory" in content
    assert "val_wisdom_hpo_global" in content


def test_terminal_logging_republishes_all_defined_selected_checkpoint_metrics() -> None:
    """A plateau must not become terminal F1/loss/accuracy when another epoch selected weights."""
    module = inspect.getmodule(Training)
    assert module is not None
    tree = ast.parse(inspect.getsource(module))
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "best_validation_metrics.items()"
    )
    assert "work.metrics.log(name, value, split='val')" in ast.unparse(loop)


def test_undefined_terminal_metrics_expose_framework_limit_without_fake_values(
    tmp_path: Path,
) -> None:
    """Keep mathematical absence explicit; the framework currently cannot log a null tombstone.

    Args:
        tmp_path: Isolated metric history directory, never an existing study.
    """
    metrics = MetricCollection(tmp_path)
    metrics.log("mcc", 0.8, split="val", step=1)
    with pytest.raises(TypeError, match="finite numeric"):
        metrics.log("mcc", None, split="val")
    result = BinaryMetricSuite().compute(torch.tensor([0.9, 0.9]), torch.tensor([1, 0]))
    assert result["mcc"] is None
