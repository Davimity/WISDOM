"""Native offline report integration using tiny synthetic fixtures, never production data."""

import base64
import gzip
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lambdaforge.analysis.Report import write_html
from lambdaforge.study_projection import read_html_sections
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.outputs import OutputCollection
from lambdaforge.work.ResultStore import ResultStore
from test_integration import _run

from wisdom.evaluation.SurfacePredictionReport import SurfacePredictionReport
from wisdom.preprocessing.common.structure.ProteinVisualizer import ProteinVisualizer
from wisdom.Training import Training
from wisdom.visualization.ModelValidation import ModelValidation
from wisdom.visualization.ProteinReportPage import ProteinReportPage
from wisdom.visualization.ValidationSelection import ValidationSelection


@pytest.fixture
def documents(tmp_path, pdb_path):
    """Render two tiny structural fixtures with distinct IDs and prediction channels."""
    manifest = tmp_path / "proteins.txt"
    manifest.write_text(str(pdb_path) + "\n")
    _run(tmp_path / "geometry", manifest, workers=1, chains=("A",))
    archive = tmp_path / "geometry/processed/tiny.npz"
    with np.load(archive, allow_pickle=False) as data:
        count = len(data["surface_positions"])
    probabilities = np.linspace(0, 1, count, dtype=np.float32)
    viewer = ProteinVisualizer(max_surface_points=50, max_mesh_points=50)
    result = []
    for identifier, label in [("tiny-negative", 0), ("tiny-positive", 1)]:
        page, _ = viewer.render(
            archive,
            identifier,
            protein_label=label,
            additional_channels={
                "model_prediction_probability": probabilities,
                "model_prediction_hard": (probabilities >= 0.5).astype(np.uint8),
            },
        )
        result.append(
            {"identifier": identifier, "label": label, "split": "validation", "html": page}
        )
    return result


def _payload(page):
    """Read the renderer's script-safe payload without executing any JavaScript."""
    encoded = re.search(
        r'id="wisdom-gallery-data" type="application/json">(.*?)</script>', page, re.S
    )
    return json.loads(encoded.group(1))


def test_empty_report_is_explanatory_and_contains_no_library():
    page, summary = ProteinReportPage().render([], {"seed": 4}, reason="Visualization disabled")
    data = _payload(page)
    assert data["documents"] == [] and data["plotly"] == ""
    assert data["reason"] == "Visualization disabled"
    assert summary["included_proteins"] == 0 and summary["omitted_proteins"] == []


def test_saved_report_refresh_preserves_point_arrays_and_joins_configuration(documents):
    """Refresh only renderer-owned UI; use matching audit identity, never guesses about labels."""
    source, _ = ProteinReportPage().render(
        [dict(documents[0], variant="Trial 143 / seed 4", seed=4)], {}
    )
    original = _payload(source)["documents"][0]
    audit = [{
        "trial": 143, "seed": 4, "split": "validation", "run_id": "source-run",
        "parameters": {"pooling_type": "attention", "attention_hidden_dim": 32,
                       "attention_variant": "gated"},
        "proteins": {documents[0]["identifier"]: {"protein_probability": 0.75}},
    }]
    rows = ProteinReportPage.upgrade(source, audit)
    assert rows[0]["configuration"]["attention_hidden_dim"] == 32
    assert rows[0]["details"]["protein_probability"] == 0.75
    old_html = gzip.decompress(base64.b64decode(original["content"])).decode()
    controls = r"const C=(.*?);\nconst gd="
    assert re.search(controls, old_html, re.S).group(1) == re.search(
        controls, rows[0]["html"], re.S
    ).group(1)
    assert "Resize ${side} panel" in rows[0]["html"]
    new, _ = ProteinReportPage().render(rows, {})
    caption = _payload(new)["documents"][0]["configuration_label"]
    assert "Attention" in caption and "32" in caption and "gated" in caption
    assert "143" not in caption
    with pytest.raises(ValueError, match="unsupported saved"):
        ProteinVisualizer.refresh("not an inspector")


def test_report_budget_interleaves_recorded_pooling_families(monkeypatch):
    """A large early family cannot suppress later families merely by exhausting display bytes."""
    monkeypatch.setattr(ProteinReportPage, "_encode", staticmethod(lambda text: "A" * 4000))
    rows = [
        {"identifier": f"protein-{index}", "label": index % 2, "split": "val",
         "variant": f"Trial {index}", "seed": 4, "html": "fixture",
         "configuration": {"pooling_type": family}}
        for index, family in enumerate(["attention"] * 100 + ["mean"] * 2 + ["max"] * 2)
    ]
    html, summary = ProteinReportPage(0.1).render(rows, {})
    assert summary["omitted_proteins"]
    assert {row["configuration"]["pooling_type"] for row in _payload(html)["documents"]} == {
        "attention", "mean", "max",
    }


def test_feature_profile_retains_model_channels_without_structural_traces(tmp_path, documents):
    """Selected learned matrices must become real channels, without leaking into metrics or GT."""
    archive = tmp_path / "geometry/processed/tiny.npz"
    with np.load(archive, allow_pickle=False) as arrays:
        count = len(arrays["surface_positions"])
    report = SurfacePredictionReport(
        SimpleNamespace(records=((archive, None, 0, "tiny", "core"),)),
        tmp_path / "feature-report", "val", content="features", report_only=True,
        identifiers=("tiny",), maximum_surface_points=30,
    )
    learned = torch.arange(count * 2).reshape(count, 2).float()
    report.collect(
        {"identifier": ["tiny"], "surface_ptr": torch.tensor([0, count])},
        {"surface_logits": torch.zeros(count), "logits": torch.tensor([1.0]),
         "surface_learned_embeddings": learned},
    )
    report.publish({}, 3)
    np.testing.assert_array_equal(
        report.representations["tiny"]["surface_learned_embeddings"], learned
    )
    document = report.report_documents[0]
    controls = json.loads(re.search(r"const C=(.*?);\nconst gd=", document["html"], re.S).group(1))
    assert set(controls["traces"]) == {"surface", "measurement"}
    assert "surface_learned_embeddings_1" in controls["surface"]
    assert len(controls["surface"]["surface_learned_embeddings_0"]["cloud"]) == 30
    assert document["details"]["surface_auprc"] is None
    assert document["details"]["protein_probability"] == pytest.approx(
        torch.sigmoid(torch.tensor(1.0)).item()
    )
    assert not list(report.output_root.rglob("*.npz"))


def test_prediction_profile_omits_structural_payload_and_fits_four_viewers(
    tmp_path, documents, monkeypatch,
):
    """A bounded prediction gallery must fit 4 MiB without computing hidden structural layers."""
    archive = tmp_path / "geometry/processed/tiny.npz"
    with np.load(archive, allow_pickle=False) as arrays:
        count = len(arrays["surface_positions"])
    viewer = ProteinVisualizer(max_surface_points=50)
    monkeypatch.setattr(viewer, "_figure", lambda *a: pytest.fail("Built structural scene"))
    monkeypatch.setattr(viewer, "_diagnostics", lambda *a: pytest.fail("Ran structural audit"))
    page, _ = viewer.render(
        archive, "prediction-only", content="predictions",
        additional_channels={
            "model_prediction_probability": np.linspace(0, 1, count),
            "model_prediction_hard": np.ones(count),
            "model_prediction_logit": np.linspace(-5, 5, count),
            "unused_learned_feature": np.ones(count),
        },
    )
    assert "unused_learned_feature" not in page
    assert len(page.encode()) < len(documents[0]["html"].encode())
    gallery = [dict(identifier=f"tiny-{i}", split="validation", label=i % 2, html=page)
               for i in range(4)]
    _, summary = ProteinReportPage(4).render(gallery, {})
    assert summary["included_proteins"] == 4 and not summary["omitted_proteins"]
    print(f"Compact inspector: {len(page.encode())} bytes; "
          f"structural inspector: {len(documents[0]['html'].encode())} bytes; "
          f"four-viewer report: {summary['size_bytes']} bytes")


@pytest.mark.parametrize("mode", ["report", "viewer", "full"])
def test_missing_renderer_fails_before_training_or_dataset_access(tmp_path, monkeypatch, mode):
    """A mixed installed viewer must fail before decoding data or constructing an optimizer."""
    monkeypatch.delattr(ProteinVisualizer, "render")
    with pytest.raises(RuntimeError, match=r"ProteinVisualizer\.render is missing"):
        Training().run(
            dataset=tmp_path / "must-not-be-opened.csv",
            surface_visualization=mode,
        )
    with pytest.raises(RuntimeError, match="submit a new Work"):
        SurfacePredictionReport(
            SimpleNamespace(), tmp_path / "reports", "validation", report_only=True
        )


def test_report_has_one_library_no_external_assets_and_exact_rendered_channels(documents):
    page, summary = ProteinReportPage().render(documents, {"seed": 7, "best_epoch": 8})
    data = _payload(page)
    assert len(data["documents"]) == 2 and summary["size_bytes"] < 4 * 1024 * 1024
    assert data["context"] == {"seed": 7, "best_epoch": 8}
    library = gzip.decompress(base64.b64decode(data["plotly"])).decode()
    assert "plotly.js" in library
    for entry in data["documents"]:
        source = gzip.decompress(base64.b64decode(entry["content"])).decode()
        assert "model_prediction_probability" in source
        assert "model_prediction_hard" in source
        assert 'src="' not in source and "window.wisdomDispose" in source
        assert "window.wisdomPlotReady = Plotly.newPlot(" in source


def test_small_report_budget_explains_omissions_without_failing(documents):
    page, summary = ProteinReportPage(0.1).render(documents, {})
    data = _payload(page)
    assert data["documents"] == [] and data["plotly"] == ""
    assert summary["included_proteins"] == 0 and len(summary["omitted_proteins"]) == 2
    assert "byte budget" in data["reason"]
    assert len(page.encode()) <= 0.1 * 1024 * 1024


@pytest.mark.parametrize("maximum_mib", [0.25, 4.0, 14.0])
def test_large_review_reserves_omission_metadata_and_actual_json_bytes(monkeypatch, maximum_mib):
    """Model a 206-Trial gallery and measure the complete serialized HTML budget.

    Args:
        monkeypatch: Replace compression with deterministic fixed-size payloads;
            this test checks byte accounting, not decompression or WebGL.
        maximum_mib: Small gallery budget and normal Training/review budgets.

    Unicode and less-than signs expand in script-safe JSON. Thousands of omitted
    viewer IDs also occupy bytes even though the viewers themselves are excluded.
    """
    monkeypatch.setattr(ProteinReportPage, "_encode", staticmethod(lambda text: "A" * 8192))
    documents = [
        {
            "identifier": f"protein-{index:04d}_A",
            "split": "val", "label": index % 2,
            "variant": f"Trial {index // 12 + 1} / seed 4",
            "seed": 4, "html": "renderer fixture",
            "details": {"description": "<" * 100 + "Ω" * 100},
        }
        for index in range(206 * 12)
    ]
    page, summary = ProteinReportPage(maximum_mib).render(
        documents, {"description": "Ω" * 100, "source": "historical V5a"}
    )
    data = _payload(page)
    assert summary["size_bytes"] == len(page.encode()) <= int(maximum_mib * 1024 * 1024)
    assert len(data["documents"]) + len(data["omitted"]) == len(documents)
    assert summary["omitted_proteins"] == data["omitted"]
    assert data["omitted"]
    assert data["context"]["description"] == "Ω" * 100
    assert summary["included_proteins"] > 0
    assert any(entry["label"] == 0 for entry in data["documents"])
    assert any(entry["label"] == 1 for entry in data["documents"])


def test_script_safe_context_and_native_section_metadata(tmp_path):
    page, _ = ProteinReportPage().render([], {"identifier": "</script><script>unsafe</script>"})
    assert "</script><script>unsafe" not in page
    outputs = OutputCollection(SimpleNamespace(run_dir=tmp_path))
    outputs.html_section(
        "protein-report", section="wisdom-proteins", title="WISDOM proteins"
    ).write_text(page)
    outputs.finalize()
    artifact = outputs.artifacts[0].to_dict()
    assert artifact["role"] == "html-section"
    assert artifact["metadata"]["html_section"] == {
        "name": "wisdom-proteins",
        "title": "WISDOM proteins",
    }


def test_report_only_collector_preserves_coverage_without_standalone_files(tmp_path, pdb_path):
    manifest = tmp_path / "proteins.txt"
    manifest.write_text(str(pdb_path) + "\n")
    _run(tmp_path / "geometry", manifest, workers=1, chains=("A",))
    archive = tmp_path / "geometry/processed/tiny.npz"
    with np.load(archive, allow_pickle=False) as data:
        count = len(data["surface_positions"])
    dataset = SimpleNamespace(records=((archive, None, 0, "tiny", "core"),))
    output = tmp_path / "predictions"
    report = SurfacePredictionReport(
        dataset, output, "validation", report_only=True, identifiers=("tiny",)
    )
    report.collect(
        {"identifier": ["tiny"], "surface_ptr": torch.tensor([0, count])},
        {"surface_logits": torch.linspace(-2, 2, count),
         "raw_surface_logits": torch.linspace(-3, 3, count)},
    )
    summary = report.publish({}, best_epoch=3)
    np.testing.assert_array_equal(report.logits["tiny"], torch.linspace(-2, 2, count).numpy())
    page = report.report_documents[0]["html"]
    assert '"defaultAtom":""' in page
    assert "model_prediction_logit" in page
    assert "model_prediction_raw_logit" in page
    assert "model_prediction_raw_probability" in page
    assert "model_prediction_refinement_delta" in page
    assert '"curvedness"' not in page
    assert summary["predicted_proteins"] == 1 and summary["visualized_proteins"] == 1
    assert report.report_documents[0]["identifier"] == "tiny"
    assert not list(output.rglob("*.ply")) and not list(output.rglob("*.npz"))
    assert not (output / "plotly.min.js").exists()
    assert not (output / "validation/proteins/tiny.html").exists()


@pytest.mark.parametrize("refiner", ["none", "learned_heat", "learned_anisotropic"])
def test_training_publishes_native_report_from_best_checkpoint(
    tmp_path, pdb_path, monkeypatch, refiner
):
    """Run one CPU epoch to verify real Training publication, not scientific model quality."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    manifest = tmp_path / "proteins.txt"
    manifest.write_text(str(pdb_path) + "\n")
    _run(tmp_path / "geometry", manifest, workers=1, chains=("A",))
    archive = tmp_path / "geometry/processed/tiny.npz"
    with np.load(archive, allow_pickle=False) as data:
        count = len(data["surface_positions"])
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    rows = ["file,annotation,label,split,identifier,tier"]
    for label in (0, 1):
        hard = (np.arange(count) % 2).astype(np.uint8) * label
        sidecar = tmp_path / f"synthetic-{label}.dna.npz"
        np.savez_compressed(
            sidecar,
            surface_target_hard=hard,
            surface_valid_mask=np.ones(count, dtype=np.bool_),
            surface_target_soft=hard.astype(np.float32),
            surface_distance_to_dna=np.where(hard, 0.5, 5.0).astype(np.float32),
            surface_distance_valid=np.ones(count, dtype=np.bool_),
            surface_target_hard_sensitivity=np.column_stack((hard, hard)),
            sensitivity_gaps=np.asarray([1.0, 1.4], dtype=np.float32),
            base_npz_sha256=np.asarray(digest),
            annotation_metadata_json=np.asarray('{"local_gt_available":true}'),
        )
        for split in ("train", "val"):
            rows.append(f"{archive},{sidecar},{label},{split},{split}-{label},core")
    labels = tmp_path / "synthetic.csv"
    labels.write_text("\n".join(rows) + "\n")
    config = WorkConfig.from_mapping(
        {
            "name": "synthetic-protein-report",
            "run": "wisdom.Training.Training",
            "with": {
                "dataset": {"file": str(labels)},
                "model_version": 1 if refiner == "none" else 2,
                "surface_refiner_type": refiner,
                "hidden_dim": 4,
                "embedding_dim": 2,
                "atomic_layers": 1,
                "surface_layers": 1,
                "batch_size": 2,
                "epochs": 1,
                "data_workers": 0,
                "precision": "float32",
                "evaluate_test": False,
                "visualization": {"mode": "report", "maximum_proteins": 2,
                                  "content": "predictions", "maximum_points": 50},
                "gate_warmup_fraction": 0.5,
            },
            "resources": {"cpu": 1},
        },
        source=tmp_path / "fixture.yaml",
    )
    result = WorkRunner().run(config)
    assert result.status == "succeeded", result.to_dict()
    snapshot = next(a for a in result.runs[0].artifacts if a.name == "best-model")
    saved = torch.load(result.runs[0].run_dir / snapshot.path,
                       map_location="cpu", weights_only=True)
    assert snapshot.metadata["step"] == snapshot.metadata["epoch"] == saved["epoch"] == 1
    assert snapshot.metadata["metrics"]["protein_auprc"] == saved["validation_metrics"]["auprc"]
    assert "surface_score" in snapshot.metadata["metrics"]
    assert "selection_regret" not in snapshot.metadata["metrics"]  # One epoch supplies no curve.
    reports = [
        run.run_dir / artifact.path
        for run in result.runs
        for artifact in run.artifacts
        if artifact.role == "html-section"
    ]
    assert len(reports) == 1
    payload = _payload(reports[0].read_text())
    assert payload["context"]["best_epoch"] == 1
    assert payload["context"]["test_evaluated"] is False
    assert {entry["identifier"] for entry in payload["documents"]} == {"val-0", "val-1"}
    assert not list(result.execution_dir.rglob("*.ply"))
    assert not list(result.execution_dir.rglob("*.predictions.npz"))
    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["seed"] == 0

    # NONE wins over the legacy default embed flag. No renderer should run, no empty tab should
    # be written, and numerical surface metrics must remain available independently.

    disabled_mapping = config.to_dict()
    disabled_mapping["name"] = "synthetic-no-report"
    disabled_mapping["with"]["visualization"] = {"mode": "none"}
    with monkeypatch.context() as no_viewers:
        no_viewers.setattr(ProteinVisualizer, "render",
                          lambda *a, **k: pytest.fail("NONE invoked the viewer"))
        disabled = WorkRunner().run(
            WorkConfig.from_mapping(disabled_mapping, source=tmp_path / "disabled.yaml")
        )
    assert disabled.status == "succeeded", disabled.to_dict()
    assert not any(a.role == "html-section" for run in disabled.runs for a in run.artifacts)

    # Export the real native execution, then reuse its exact checkpoint. Prohibit optimizer
    # construction so this end-to-end test cannot accidentally validate by training again.

    exported = ResultStore(result.execution_dir.parent.parent).export(
        result.execution_id, tmp_path / "reviews"
    )
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Unexpected retraining"))
    review_mapping = {
        "name": "synthetic-frozen-review",
        "run": "wisdom.visualization.ModelValidation.ModelValidation",
        "with": {
            "source_study": {"file": exported["path"]},
            "dataset": {"file": str(labels)},
            "selected_trials": {"max": 0},
            "selection": "all",
        },
        "resources": {"cpu": 1},
    }
    review = WorkRunner().run(
        WorkConfig.from_mapping(review_mapping, source=tmp_path / "review.yaml")
    )
    assert review.status == "succeeded", review.to_dict()
    evidence = review.runs[0].run_dir
    artifacts = {asset.name: evidence / asset.path for asset in review.runs[0].artifacts}
    audit = json.loads(artifacts["validation-audit"].read_text())
    assert len(audit) == 1 and len(audit[0]["proteins"]) == 2
    assert audit[0]["gate_override"] == "all_on"
    assert audit[0]["proteins"]["val-0"]["surface_auprc"] is None
    assert audit[0]["proteins"]["val-1"]["surface_auprc"] is not None
    gallery = _payload(artifacts["protein-report"].read_text())
    assert {entry["variant"] for entry in gallery["documents"]} == {"max"}
    assert not list(evidence.rglob("*.npz"))
    assert not list(evidence.rglob("*.pt"))

    # An empty reviewer ledger publishes inventory without guessing a model or doing inference.

    review_mapping["name"] = "synthetic-empty-review"
    review_mapping["with"]["selected_trials"] = {}
    empty = WorkRunner().run(
        WorkConfig.from_mapping(review_mapping, source=tmp_path / "empty.yaml")
    )
    assert empty.status == "succeeded", empty.to_dict()
    empty_audit = next(a for a in empty.runs[0].artifacts if a.name == "validation-audit")
    assert json.loads((empty.runs[0].run_dir / empty_audit.path).read_text()) == []

    # A deleted exported checkpoint must be audited/skipped in the explicit permissive mode,
    # never reconstructed by a hidden trainer. The strict mode must identify its original Run.

    export_root = Path(exported["path"])
    export_manifest = json.loads((export_root / "manifest.json").read_text())
    checkpoint = next(
        x["path"] for x in export_manifest["inventory"]
        if x["path"].endswith("best-model.pt")
    )
    (export_root / checkpoint).unlink()
    review_mapping["name"] = "synthetic-missing-review"
    review_mapping["with"]["selected_trials"] = {"max": 0}
    review_mapping["with"]["strict_checkpoints"] = False
    missing = WorkRunner().run(
        WorkConfig.from_mapping(review_mapping, source=tmp_path / "missing.yaml")
    )
    assert missing.status == "succeeded", missing.to_dict()
    missing_outputs = missing.to_dict()["runs"][0]["result"]
    assert missing_outputs["reviewed_models"] == 0 and missing_outputs["retrained_models"] == 0
    assert len(missing_outputs["missing"]) == 1
    review_mapping["name"] = "synthetic-strict-review"
    review_mapping["with"]["strict_checkpoints"] = True
    strict = WorkRunner().run(
        WorkConfig.from_mapping(review_mapping, source=tmp_path / "strict.yaml")
    )
    assert strict.status == "failed"


def test_review_selection_keeps_extremes_overlap_and_unavailable_targets():
    """Contrasting pictures are descriptive samples; unavailable GT is not a worst-map zero."""
    rows = {
        "bad": dict(
            label=1,
            protein_probability=0.1,
            binary_error=True,
            surface_auprc=0.2,
            normalized_surface_auprc=0.0,
            positive_mass=0.2,
        ),
        "good": dict(
            label=1,
            protein_probability=0.9,
            binary_error=False,
            surface_auprc=0.95,
            normalized_surface_auprc=0.9,
            positive_mass=0.3,
        ),
        "unknown": dict(
            label=1,
            protein_probability=0.7,
            binary_error=False,
            surface_auprc=None,
            normalized_surface_auprc=None,
            positive_mass=0.4,
        ),
        "negative": dict(
            label=0,
            protein_probability=0.8,
            binary_error=True,
            surface_auprc=None,
            normalized_surface_auprc=None,
            positive_mass=0.6,
        ),
    }
    chosen, reasons = ModelValidation._select(rows, ValidationSelection.DIAGNOSTIC, 1, ())
    assert "worst_surface" in reasons["bad"] and "best_surface" in reasons["good"]
    assert "negative_false_mass" in reasons["negative"] and "unknown" not in chosen
    chosen, _ = ModelValidation._select(rows, ValidationSelection.ALL, 1, ())
    assert chosen == sorted(rows)
    chosen, _ = ModelValidation._select(rows, ValidationSelection.EXPLICIT, 1, ("good", "missing"))
    assert chosen == ["good"]


def test_review_rejects_modified_checkpoint_bytes(tmp_path):
    """Portable checkpoint verification must reject corruption, not merely trust filenames."""
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    inventory = {"model.pt": {"size_bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}}
    assert ModelValidation._verified(tmp_path, "model.pt", inventory) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        ModelValidation._verified(tmp_path, "model.pt", inventory)


@pytest.mark.parametrize("content", ["full", "predictions", "features"])
def test_native_report_keeps_seeds_and_inspector_navigation_in_isolated_frame(
    tmp_path, documents, content,
):
    """Exercise the real LF report CSP, lazy viewer, return path and per-seed selection."""
    playwright = pytest.importorskip("playwright.sync_api")
    if content != "full":
        archive = tmp_path / "geometry/processed/tiny.npz"
        with np.load(archive, allow_pickle=False) as arrays:
            count = len(arrays["surface_positions"])
        for doc in documents:
            doc["html"], _ = ProteinVisualizer(max_surface_points=50).render(
                archive, doc["identifier"], content=content,
                additional_channels={
                    "model_prediction_probability": np.linspace(0, 1, count),
                    "model_prediction_hard": np.ones(count),
                    "model_prediction_logit": np.linspace(-5, 5, count),
                    "surface_learned_embeddings_0": np.linspace(-1, 1, count),
                },
            )
    runs = []
    for seed in (4, 7):
        root = tmp_path / f"runs/run-{seed}/attempts/attempt-0001"
        root.mkdir(parents=True)
        outputs = OutputCollection(SimpleNamespace(run_dir=root))
        labelled = [
            dict(
                doc, variant=("max" if i == 0 else "topk"), seed=seed,
                details={"protein_probability": 0.1 if i == 0 else 0.9,
                         "surface_auprc": None if i == 0 else 0.8,
                         "binary_error": False,
                         "selection_reasons": ["correct_negative" if i == 0
                                               else "best_localization"]},
            )
            for i, doc in enumerate(documents)
        ]
        page, _ = ProteinReportPage().render(labelled, {"seed": seed})
        outputs.html_section(
            "protein-report", section="wisdom-proteins", title="WISDOM proteins"
        ).write_text(page)
        outputs.finalize()
        runs.append(
            {
                "run_dir": str(root),
                "trial": {"index": 1},
                "seed": seed,
                "artifacts": [artifact.to_dict() for artifact in outputs.artifacts],
            }
        )
    sections = read_html_sections(runs, tmp_path)
    assert (
        len(sections) == 2 and "seed 4" in sections[0]["label"] and "seed 7" in sections[1]["label"]
    )
    report = write_html(
        {
            "source": {"status": "final"},
            "objective": {"metric": "val_auprc", "mode": "max"},
            "search_space": {},
            "candidates": [],
        },
        tmp_path / "integrated.html",
        sections=sections,
    )
    with playwright.sync_playwright() as runtime:
        executable = Path(runtime.chromium.executable_path)
        if not executable.exists():
            pytest.skip("Optional Chromium unavailable")
        browser = runtime.chromium.launch(
            args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        errors, external = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                external.append(request.url)
                if request.url.startswith(("http:", "https:"))
                else None
            ),
        )
        page.goto(report.as_uri())
        page.get_by_role("button", name="WISDOM proteins", exact=True).click()
        frame = page.frame_locator('iframe[title="WISDOM proteins"]')
        assert frame.get_by_role("heading", name="WISDOM proteins").is_visible()
        assert frame.get_by_text("Correct negative", exact=True).first.is_visible()
        frame.get_by_label("Metric threshold").fill("0.2")
        frame.get_by_label("Protein label", exact=True).select_option("0")
        assert frame.get_by_role("button", name="Compare configurations").count() == 0
        frame.get_by_label("Protein label", exact=True).select_option("1")
        assert frame.get_by_role("button", name="Compare configurations").count() == 1
        frame.get_by_label("Metric threshold").fill("")
        frame.get_by_label("Protein label", exact=True).select_option("")
        frame.get_by_label("Model or pooling").select_option("max")
        assert frame.get_by_role("button", name="Compare configurations").count() == 1
        frame.get_by_label("Model or pooling").select_option("")
        frame.get_by_label("Source seed").select_option("4")
        assert not frame.locator("pre").count()
        frame.get_by_role("button", name="Compare configurations").first.click()
        frame.get_by_role("button", name="Open protein").first.click()
        frame.get_by_role("button", name="← Protein list").wait_for(timeout=30000)
        assert frame.locator("#selection-reasons").is_visible()
        assert frame.locator("#selection-reasons").inner_text().startswith("Correct negative")
        assert frame.locator("#wisdom-plot").evaluate("gd=>gd.data.length") == (
            11 if content == "full" else 2
        )
        frame.locator("#surface-channel").select_option("model_prediction_hard")
        assert frame.locator("#prediction-threshold-field").is_visible()
        frame.locator("#prediction-threshold").fill("0.8")
        if content == "full":
            frame.get_by_role("button", name="Solid mesh", exact=True).click()
        else:
            assert not frame.get_by_role("button", name="Solid mesh", exact=True).is_visible()
            frame.locator("#surface-channel").select_option("model_prediction_logit")
            frame.locator("#surface-palette").select_option("Cividis")
            frame.get_by_role("button", name="Surface", exact=True).click()
        frame.locator("#viewer-state").get_by_text("Ready", exact=True).wait_for()
        frame.get_by_role("button", name="Surface", exact=True).click()
        frame.locator("#viewer-state").get_by_text("Ready", exact=True).wait_for()
        frame.locator("#surface-palette").select_option("Turbo")
        frame.locator("#viewer-state").get_by_text("Ready", exact=True).wait_for()
        assert frame.locator("#wisdom-plot").evaluate(
            "gd=>gd._fullData[0].marker.colorscale[0][1]"
        ) == "#30123b"
        if content == "features":
            assert frame.locator(
                '#surface-channel option[value="surface_learned_embeddings_0"]'
            ).count() == 1
        frame.locator("#surface-range-mode").select_option("unit")
        assert frame.locator("#surface-min").input_value() == "0.00000"
        frame.locator("#compare-toggle").click()
        frame.locator("#viewer-state").get_by_text("Ready", exact=True).wait_for()
        assert frame.locator("#wisdom-plot").evaluate(
            "gd=>gd._fullLayout.scene2.domain.x"
        ) == [0.52, 1]
        frame.locator("#compare-right").select_option("model_prediction_hard")
        frame.locator("#prediction-threshold").fill("0.9")
        frame.locator("#compare-right-range").select_option("custom")
        frame.locator("#compare-right-min").fill("-2")
        frame.locator("#compare-right-min").dispatch_event("change")
        frame.locator("#compare-right-max").fill("2")
        frame.locator("#compare-right-max").dispatch_event("change")
        frame.locator("#viewer-state").get_by_text("Ready", exact=True).wait_for()
        assert frame.locator("#wisdom-plot").evaluate(
            "gd=>gd.data[gd.data.length-1].marker.cmax"
        ) == 2
        frame.locator("#wisdom-plot").evaluate(
            "gd=>Plotly.relayout(gd,{'scene.camera.eye':{x:2,y:3,z:4}})"
        )
        assert frame.locator("#wisdom-plot").evaluate(
            "gd=>gd._fullLayout.scene2.camera.eye.x===2"
        )
        frame.locator("#compare-toggle").click()
        frame.locator("#viewer-state").get_by_text("Ready", exact=True).wait_for()
        frame.locator('[data-panel="left"][data-action="close"]').click()
        frame.locator('[data-panel="left"][data-action="open"]').click()
        grip = frame.get_by_role("separator", name="Resize left panel")
        grip.focus()
        page.keyboard.press("ArrowRight")
        assert frame.locator("#controls-panel").bounding_box()["width"] == 320
        frame.locator("#measure-toggle").click()
        frame.locator("#wisdom-plot").evaluate(
            """gd=>{
                gd.emit('plotly_click',{});gd.emit('plotly_click',{points:[]});
                gd.emit('plotly_click',{points:[{curveNumber:0,pointNumber:0,
                    customdata:0,x:1,y:2,z:3}]});
                gd.emit('plotly_click',{points:[{curveNumber:0,pointNumber:1,
                    customdata:1,x:4,y:6,z:3}]});
            }"""
        )
        assert "5.0000 Å" in frame.locator("#measure-result").inner_text()
        page.screenshot(path=str(tmp_path / "protein-report.png"), full_page=True)
        frame.get_by_role("button", name="← Protein list").click()
        frame.get_by_role("button", name="Compare configurations").last.click()
        frame.get_by_role("button", name="Open protein").last.click()
        frame.get_by_role("button", name="← Protein list").wait_for(timeout=30000)
        assert frame.locator(".viewer-head strong").inner_text() == "tiny-positive"
        frame.get_by_role("button", name="← Protein list").click()
        frame.get_by_role("heading", name="WISDOM proteins").wait_for()
        page.click("#project-section-0-document-picker")
        page.locator('#study-dropdown-options input[data-choice="1"]').check()
        frame.get_by_label("Source seed").select_option("7")
        frame.get_by_role("button", name="Compare configurations").first.click()
        frame.get_by_role("button", name="Open protein").first.click()
        frame.get_by_role("button", name="← Protein list").wait_for(timeout=30000)
        assert not errors and not external
        browser.close()
