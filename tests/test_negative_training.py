"""Real tiny NPZ/Training execution for VN01 timing, native continuation and G selection."""

import json

import torch
import yaml
from lambdaforge.work import WorkConfig, WorkRunner
from test_structural_enrichment import _complete_base


def test_negative_mining_native_training_and_recovery(tmp_path, pdb_path, monkeypatch, request):
    """Recover a committed epoch's optimizer/masks without reviewing it twice."""
    # Four labeled records deliberately reuse one tiny structural fixture. This is a software
    # execution fixture, NOT a leakage-safe scientific benchmark or a measured model comparison.

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    base = _complete_base(tmp_path / "base", pdb_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file,label,split\n" + "".join(
        f"{base},{label},{split}\n" for split in ("train", "val") for label in (0, 1)))
    config = tmp_path / "tiny-vn.yaml"
    config.write_text(yaml.safe_dump({"run": "wisdom.Training.Training", "with": {
        "dataset": {"file": str(manifest)}, "model_version": 2,
        "architecture_spike": "baseline", "architecture_implementation": "controlled",
        "head_implementation": "controlled", "initialization_profile": "baseline",
        "optimization_profile": "baseline", "weak_loss_profile": "custom",
        "hidden_dim": 8, "embedding_dim": 4, "residue_embedding_dim": 4,
        "atomic_layers": 1, "surface_layers": 1, "projection_depth": 1,
        "diffusion_spectral_modes": 8, "batch_size": 2, "epochs": 3, "patience": 3,
        "precision": "float32", "data_workers": 0, "surface_metrics": False,
        "visualization": {"mode": "none"}, "evaluate_test": False,
        "negative_mining": {"mode": "regional", "lower": .01, "upper": .02,
                            "strong_weight": .1, "regions": 2},
        "negative_calibration": {"formal": False},
    }}, sort_keys=False))
    training_module = __import__("wisdom.Training", fromlist=["_negative_snapshot"])
    snapshot = training_module._negative_snapshot
    calls = []

    def interrupted(*args):
        """Fail during the second review; the first optimizer epoch must already be committed."""
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic interruption after committed epoch one")
        return snapshot(*args)

    monkeypatch.setattr(training_module, "_negative_snapshot", interrupted)
    runner = WorkRunner(execution_root=tmp_path / "runs")
    first = runner.run(WorkConfig.from_yaml(config))
    assert first.status == "failed"
    checkpoints = list((tmp_path / "runs").rglob("continuation.pt"))
    assert len(checkpoints) == 1
    saved = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    assert saved["epoch"] == saved["mining"]["review_epoch"] == 1
    monkeypatch.setattr(training_module, "_negative_snapshot", snapshot)
    second = runner.run(WorkConfig.from_yaml(config))
    assert second.status == "succeeded"
    saved = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    assert saved["epoch"] == 3
    assert saved["mining"]["review_epoch"] == 2  # No unused review after the final epoch.
    reports = list((tmp_path / "runs").rglob("evaluation.json"))
    report = json.loads(reports[-1].read_text())
    assert report["test_evaluated"] is False
    assert report["negative_admission"]["status"] == "unavailable_no_independent_negative_reserve"
    assert report["best_epoch"] in (1, 2, 3)
    assert [r["epoch"] for r in report["negative_mining_reviews"]] == [1, 2]
    assert saved["best"]["epoch"] == report["best_epoch"]
    assert report["negative_mining_first_gradient_epoch"] == 2
    assert report["negative_mining_selected_before_auxiliary"] == (report["best_epoch"] < 2)
    assert not list((tmp_path / "runs").rglob("*.html"))

    uninterrupted = WorkRunner(execution_root=tmp_path / "uninterrupted")
    assert uninterrupted.run(WorkConfig.from_yaml(config)).status == "succeeded"
    complete = torch.load(next((tmp_path / "uninterrupted").rglob("continuation.pt")),
                          map_location="cpu", weights_only=True)
    assert complete["mining"] == saved["mining"]
    for key, value in complete["model"].items():
        # Sparse FP32 reductions may differ in their last bits between CPU executions;
        # masks/RNG state are exact, and parameter comparison has an explicit FP32 tolerance.

        torch.testing.assert_close(value, saved["model"][key], rtol=1e-6, atol=1e-7)
