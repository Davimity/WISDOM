"""Native post-hoc contracts on tiny frozen fixtures, never training production models."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import lambdaforge as lf
import numpy as np
import pytest
import torch
import yaml
from lambdaforge.products import (
    ProductBundle,
    ProductInput,
    ProductRegistry,
    SelectionPolicy,
    select_models,
)
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.ResultStore import ResultStore
from test_integration import _run

from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.Training import _checkpoint_metadata, _create_model
from wisdom.visualization.ModelSetReview import ModelSetReview
from wisdom.visualization.ReviewInference import ReviewInference
from wisdom.visualization.ReviewSelection import ReviewSelection
from wisdom.visualization.StudyReview import StudyReview


class SnapshotWork(lf.Work):
    """Publish randomly initialized test weights, without fitting even one batch."""

    def run(self, dataset: Path, epoch: int = 7):
        """Seal tiny weights and contradictory latest/snapshot metrics for native API tests.

        Args:
            dataset: Synthetic label manifest whose typed identity follows the model.
            epoch: Fixed saved-snapshot epoch for metadata binding.

        Returns:
            Published epoch; no optimizer, backward or training lifecycle is used.
        """
        model, parameters = _create_model(
            1,
            {
                "hidden_dim": 4,
                "embedding_dim": 2,
                "atomic_layers": 1,
                "surface_layers": 1,
            },
        )
        checkpoint = Path(self.outputs.directory("snapshots")) / "best-model.pt"
        torch.save(
            {
                "model_version": 1,
                "model_parameters": parameters,
                "state_dict": model.state_dict(),
                "data_parameters": {},
                "epoch": epoch,
                "inference_state": {"gate_override": "learned"},
            },
            checkpoint,
        )
        score = 0.4 + (self.seed or 0) / 1000
        self.metrics.log("wisdom_score", 1 - score)
        self.outputs.artifact(
            "best-model",
            checkpoint,
            role="checkpoint",
            metadata={
                "step": epoch,
                "metrics": {
                    "wisdom_score": score,
                    "global_score": score,
                    "surface_score": 1 - score,
                },
            },
        )
        return {"epoch": epoch}


def row(trial, seed, score, **parameters):
    """Build native-shaped evidence with deliberately wrong latest metrics."""
    return {
        "run_id": f"run-{trial}-{seed}",
        "trial_index": trial,
        "trial": {"index": trial},
        "seed": seed,
        "status": "succeeded",
        "attempt_number": 1,
        "parameters": parameters,
        "metrics": {"wisdom_score": 99},
        "artifacts": [
            {
                "name": "best-model",
                "metadata": {
                    "metrics": {
                        "wisdom_score": score,
                        "global_score": score,
                        "surface_score": 1 - score,
                    }
                },
            }
        ],
    }


@pytest.fixture
def ranked():
    """Two configurations, three seeds each, with known snapshot order."""
    return [
        row(trial, seed, score, pooling_type=pooling)
        for trial, pooling, offset in [(0, "max", 0), (1, "attention", 0.05)]
        for seed, score in [(4, 0.9 - offset), (7, 0.5 - offset), (32, 0.1 - offset)]
    ]


@pytest.mark.parametrize(
    "mode,seeds",
    [
        ("all", [4, 7, 32]),
        ("explicit", [7]),
        ("best", [4]),
        ("worst", [32]),
        ("median", [7]),
        ("representative", [4, 7, 32]),
        ("top_k", [4, 7]),
        ("bottom_k", [7, 32]),
    ],
)
def test_seed_policies_use_only_snapshot_metrics(ranked, mode, seeds):
    selected = ReviewSelection.seeds(ranked, {"mode": mode, "seeds": [7], "k": 2})
    assert {r["seed"] for r in selected} == set(seeds)
    assert len(selected) == 2 * len(seeds)
    assert ReviewSelection.seeds(ranked[:1], {"mode": "representative"}) == ranked[:1]
    assert ReviewSelection.seeds(ranked, None)[0]["seed"] == 7


@pytest.mark.parametrize(
    "policy,trials",
    [
        ({"mode": "all"}, [0, 1]),
        ({"mode": "explicit", "trial_indices": [1]}, [1]),
        ({"mode": "best_by_metric"}, [0]),
        ({"mode": "top_k_by_metric", "k": 2}, [0, 1]),
        ({"mode": "parameter_filter", "where": {"pooling_type": ["attention"]}}, [1]),
        ({"mode": "pareto", "metrics": {"global_score": "max", "surface_score": "max"}}, [0, 1]),
    ],
)
def test_trial_policies_and_aggregation(ranked, policy, trials):
    assert sorted({r["trial_index"] for r in ReviewSelection.trials(ranked, policy)}) == trials
    with pytest.raises(ValueError, match="explicit trial_selection"):
        ReviewSelection.trials(ranked, None)
    assert len(ReviewSelection.trials(ranked[:3], None)) == 3


def test_pareto_dominance_mean_and_conditional_null(ranked):
    ranked.append(row(2, 4, 0.05))
    ranked[-1]["artifacts"][0]["metadata"]["metrics"]["surface_score"] = 0.05
    chosen = ReviewSelection.trials(
        ranked,
        {
            "mode": "pareto",
            "metrics": {"global_score": "max", "surface_score": "max"},
            "aggregation": "mean",
        },
    )
    assert {r["trial_index"] for r in chosen} == {0, 1}
    absent, null = row(0, 4, 0.5), row(1, 4, 0.5, beta=None)
    assert ReviewSelection.trials(
        [absent, null],
        {
            "mode": "parameter_filter",
            "where": {"beta": None},
        },
    ) == [null]


def test_failed_pruned_partial_fidelity_and_latest_attempt_excluded(ranked):
    rows = copy.deepcopy(ranked)
    rows[0]["status"] = "failed"
    rows[1]["pruned"] = True
    rows[2]["fidelity"] = {"target": 2, "maximum": 10}
    rows[3]["termination_type"] = "performance_pruned"
    newer = {**rows[4], "attempt_number": 2, "status": "failed"}
    assert [r["run_id"] for r in ReviewSelection.eligible([*rows, newer])] == [rows[5]["run_id"]]


def test_historical_metadata_requires_explicit_policies(ranked):
    for r in ranked:
        r["artifacts"][0]["metadata"] = {}
    with pytest.raises(ValueError, match="checkpoint-bound"):
        ReviewSelection.seeds(ranked, None)
    with pytest.raises(ValueError, match="checkpoint-bound"):
        ReviewSelection.trials(ranked, {"mode": "best_by_metric"})
    assert (
        len(ReviewSelection.seeds(ReviewSelection.trials(ranked, {"mode": "all"}), {"mode": "all"}))
        == 6
    )


def test_checkpoint_metadata_exact_epoch_not_unrelated_maximum():
    curve = [
        {"epoch": 1, "global": 0.99, "surface": 0.99},
        {"epoch": 7, "global": 0.7, "surface": 0.6},
        {"epoch": 8, "global": 0.6, "surface": 0.5},
    ]
    metadata = _checkpoint_metadata(
        7, {"auprc": 0.7, "auroc": 0.7}, {"surface_positive_macro_auprc": 0.6}, curve
    )
    assert metadata["step"] == metadata["epoch"] == 7
    m = metadata["metrics"]
    assert m["global_score"] == pytest.approx(0.7)
    assert m["surface_score"] == 0.6 and m["selection_regret"] == pytest.approx(0.39)
    assert m["wisdom_score"] == pytest.approx(0.35 * 0.7 + 0.45 * 0.6 + 0.20 * m["coupling"])
    unavailable = _checkpoint_metadata(7, {"auprc": 0.7, "auroc": 0.7}, None, [])
    assert "wisdom_score" not in unavailable["metrics"]


@pytest.fixture
def native_source(tmp_path, pdb_path, monkeypatch):
    """Create registered evidence and tiny geometry, never a trained model."""
    monkeypatch.setenv("LAMBDAFORGE_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("LAMBDAFORGE_PRODUCT_ROOT", str(tmp_path / "products"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    manifest = tmp_path / "proteins.txt"
    manifest.write_text(str(pdb_path) + "\n")
    _run(tmp_path / "geometry", manifest, workers=1, chains=("A",))
    archive = tmp_path / "geometry/processed/tiny.npz"
    with np.load(archive, allow_pickle=False) as data:
        count = len(data["surface_positions"])
    rows = ["file,annotation,label,split,identifier,tier"]
    for label in (0, 1):
        hard = (np.arange(count) % 2).astype(np.uint8) * label
        sidecar = tmp_path / f"synthetic-{label}.dna.npz"
        np.savez_compressed(
            sidecar,
            surface_target_hard=hard,
            surface_target_soft=hard.astype(np.float32),
            surface_valid_mask=np.ones(count, dtype=np.bool_),
            surface_distance_to_dna=np.where(hard, 0.5, 5.0).astype(np.float32),
            surface_distance_valid=np.ones(count, dtype=np.bool_),
            surface_target_hard_sensitivity=np.column_stack((hard, hard)),
            sensitivity_gaps=np.asarray([1.0, 1.4], dtype=np.float32),
            base_npz_sha256=np.asarray(hashlib.sha256(archive.read_bytes()).hexdigest()),
            annotation_metadata_json=np.asarray('{"local_gt_available":true}'),
        )
        rows.append(f"{archive},{sidecar},{label},val,val-{label},core")
    labels = tmp_path / "synthetic.csv"
    labels.write_text("\n".join(rows) + "\n")
    config = WorkConfig.from_mapping(
        {
            "name": "snapshot-fixture",
            "run": "test_review_native.SnapshotWork",
            "with": {"dataset": {"file": str(labels)}, "epoch": 7},
            "seeds": [4, 7, 32],
            "resources": {"cpu": 1},
        },
        source=tmp_path / "fixture.yaml",
    )
    execution = WorkRunner().run(config)
    assert execution.status == "succeeded", execution.to_dict()
    return ResultStore(), execution, labels


def forbid_training(monkeypatch):
    """Fail immediately on any review optimizer or backward path."""

    def forbidden(*args, **kwargs):
        raise AssertionError("post-hoc review cannot train")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)


def review_config(tmp_path, execution, labels, **options):
    """Build a review with typed fixture data, no production selector."""
    return WorkConfig.from_mapping(
        {
            "name": "frozen-review",
            "run": "wisdom.visualization.StudyReview.StudyReview",
            "with": {
                "source_execution": execution.execution_id,
                "dataset": {"file": str(labels)},
                "maximum_surface_points": 20,
                **options,
            },
            "resources": {"cpu": 1},
        },
        source=tmp_path / "review.yaml",
    )


def output(execution, name):
    """Locate a named output through native published artifact records."""
    r = execution.runs[0]
    return r.run_dir / next(a.path for a in r.artifacts if a.name == name)


@pytest.mark.parametrize("imported,scope", [(False, "full"), (True, "explicit")])
def test_native_and_imported_review_forward_coverage(
    native_source, tmp_path, monkeypatch, imported, scope
):
    store, execution, labels = native_source
    options = {
        "evaluation_scope": scope,
        "protein_selection": {"mode": "explicit", "identifiers": ["val-1"]},
    }
    if imported:
        package = store.export(execution.execution_id, tmp_path / "exports")
        imported_store = ResultStore(tmp_path / "imported")
        imported_store.import_export(package["path"], apply=True)
        options["results_root"] = {"file": str(imported_store.root)}
    forbid_training(monkeypatch)
    seen = []
    original = WisdomDataset.__getitem__

    def observe(self, index):
        seen.append(self.records[index][3])
        return original(self, index)

    monkeypatch.setattr(WisdomDataset, "__getitem__", observe)
    review = WorkRunner().run(review_config(tmp_path, execution, labels, **options))
    assert review.status == "succeeded", review.to_dict()
    assert set(seen) == ({"val-1"} if scope == "explicit" else {"val-0", "val-1"})
    audit = json.loads(output(review, "review-audit").read_text())
    assert audit[0]["seed"] == 7 and audit[0]["checkpoint_epoch"] == 7
    assert audit[0]["coverage"] == ("complete" if scope == "full" else "requested subset only")
    assert audit[0]["evaluated_proteins"] == (2 if scope == "full" else 1)
    assert "post-hoc inference on frozen checkpoint" in output(review, "protein-report").read_text()
    assert not list(review.execution_dir.rglob("*.pt"))
    assert not list(review.execution_dir.rglob("*.npz"))


def test_native_products_select_snapshot_ranking_and_durable_modelset(
    native_source, tmp_path, monkeypatch
):
    store, execution, labels = native_source
    selection = select_models(
        store.select(execution.execution_id),
        execution.execution_dir,
        SelectionPolicy("wisdom_score", artifact="best-model"),
        name="review-models",
        contract="wisdom/review-models:v1",
    )
    assert selection.product.payload["models"][0]["seed"] == 32
    registry = ProductRegistry()
    registry.publish(selection.product, files=dict(selection.sources), apply=True)
    bundle = tmp_path / "product-export"
    ProductBundle.export(registry, "review-models", bundle, apply=True)
    receiving = ProductRegistry(tmp_path / "receiving-products")
    ProductBundle.import_bundle(receiving, bundle, apply=True)
    store.delete(execution.execution_id, apply=True)
    monkeypatch.setenv("LAMBDAFORGE_PRODUCT_ROOT", str(receiving.root))
    forbid_training(monkeypatch)
    config = WorkConfig.from_mapping(
        {
            "name": "durable-review",
            "run": "wisdom.visualization.ModelSetReview.ModelSetReview",
            "with": {
                "models": {
                    "product": {"name": "review-models", "contract": "wisdom/review-models:v1"}
                },
                "dataset": {"file": str(labels)},
                "maximum_surface_points": 20,
            },
            "resources": {"cpu": 1},
        },
        source=tmp_path / "modelset.yaml",
    )
    result = WorkRunner().run(config)
    assert result.status == "succeeded", result.to_dict()
    assert json.loads(output(result, "review-audit").read_text())[0]["seed"] == 32
    promoted = ProductInput(receiving.show("review-models"), receiving.root)
    promoted.artifact("model-1").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        promoted.artifact("model-1")
    with pytest.raises(ValueError, match="ModelSet"):
        ModelSetReview().run(
            SimpleNamespace(
                product=SimpleNamespace(
                    kind="ScientificReport", contract=SimpleNamespace(identifier="wrong")
                )
            ),
            labels,
        )


def test_dataset_mismatch_and_test_guard():
    with pytest.raises(ValueError, match="identities differ"):
        ReviewInference.matching_dataset(
            [{"name": "dataset", "content_id": "other"}], SimpleNamespace(content_id="expected")
        )
    with pytest.raises(ValueError, match="allow_test"):
        StudyReview().run("no-source", Path("no-data"), splits=["test"])
    with pytest.raises(ValueError, match="allow_test"):
        ModelSetReview().run(None, Path("no-data"), splits=["test"])


def test_missing_and_corrupt_checkpoint_native_boundary(native_source, tmp_path, monkeypatch):
    _, execution, labels = native_source
    forbid_training(monkeypatch)
    r = execution.runs[0]
    path = r.run_dir / next(a.path for a in r.artifacts if a.name == "best-model")
    original = path.read_bytes()
    path.write_bytes(b"corrupt")
    options = {"seed_selection": {"mode": "explicit", "seeds": [4]}}
    failed = WorkRunner().run(review_config(tmp_path, execution, labels, **options))
    assert failed.status == "failed"
    path.write_bytes(original)
    path.unlink()
    permissive = WorkRunner().run(
        review_config(tmp_path, execution, labels, strict_checkpoints=False, **options)
    )
    assert permissive.status == "succeeded", permissive.to_dict()
    assert permissive.runs[0].to_dict()["result"]["reviewed_models"] == 0
    strict = WorkRunner().run(review_config(tmp_path, execution, labels, **options))
    assert strict.status == "failed"


def test_authored_yaml_contracts_fixture_bound(native_source, tmp_path):
    """Validate/explain/dry-run authored YAMLs without production inputs or computation.

    Native temporary bindings establish configuration executability only. A manifest standing
    in for a frozen evidence file is never read by Selection/Preprocessing in this dry run;
    scientific file formats have their separate unit tests. Historical .lambdaforge bundles
    are execution evidence, not current configurations, and must not be rewritten or scanned.
    """
    store, execution, labels = native_source
    selection = select_models(
        store.select(execution.execution_id),
        execution.execution_dir,
        SelectionPolicy("global_score", artifact="best-model"),
        name="review-models",
        contract="wisdom/review-models:v1",
    )
    ProductRegistry().publish(selection.product, files=dict(selection.sources), apply=True)
    folder = Path(__file__).resolve().parents[1] / "experiments"
    paths = [path for path in folder.rglob("*.yaml")
             if not any(part.startswith(".") for part in path.relative_to(folder).parts)]
    assert paths
    for path in paths:
        authored = yaml.safe_load(path.read_text())
        if path.parent.name == "policies":
            SelectionPolicy.from_mapping(authored)
            continue
        def bind_inputs(value):
            """Bind external selectors without changing dependencies or scientific parameters."""
            if isinstance(value, dict):
                if set(value) == {"dataset"} or set(value) == {"file"}:
                    return {"file": str(labels)}
                return {key: bind_inputs(item) for key, item in value.items()}
            if isinstance(value, list):
                return [bind_inputs(item) for item in value]
            return value

        authored = bind_inputs(authored)
        config = WorkConfig.from_mapping(authored, source=tmp_path / path.name)
        assert not config.validation_errors(check_inputs=True), path
        assert config.explanation()
        dry = WorkRunner().run(config, dry_run=True)
        assert dry.to_dict()["plan_version"] == 1, (path, dry.to_dict())
