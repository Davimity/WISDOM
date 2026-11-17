"""VN01 CPU contracts: current-logit loss, temporal tracking, real FGW and safe calibration."""

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lambdaforge.work.checkpoints import CheckpointCollection
from scipy.sparse import csr_matrix

from wisdom.evaluation.NegativeAdmission import NegativeAdmission
from wisdom.models.negative.NegativeMining import NegativeMining
from wisdom.models.negative.RegionDiversity import RegionDiversity
from wisdom.models.negative.SurfaceRegions import SurfaceRegions
from wisdom.models.optimization.MiningContinuation import MiningContinuation


def geometry(count=7, edges=None):
    """Make a sparse line with equal proxy weights; no protein or local-target labels."""
    result = SurfaceRegions.__new__(SurfaceRegions)
    result.points = np.column_stack((np.arange(count), np.zeros((count, 2))))
    result.weights = np.full(count, 1 / count)
    edges = np.array(edges if edges is not None else [[i, i+1] for i in range(count-1)])
    edges = edges.reshape(-1, 2)
    src, dst = edges.T
    result.graph = csr_matrix((np.ones(2 * len(edges)),
        (np.r_[src, dst], np.r_[dst, src])), shape=(count, count))
    result.identity = "immutable-line-" + str(count)
    result.features = np.column_stack((np.arange(count), np.zeros(count)))
    result.feature_valid = np.ones_like(result.features, dtype=bool)
    return result


def test_components_global_budget_ties_and_point_ownership():
    """Disconnected high islands stay separate; global K ignores loader/protein order."""
    surfaces = {"B": geometry(), "A": geometry()}
    probabilities = {"B": np.array([.7, .3, .1, .8, .1, .1, .1]),
                     "A": np.array([.7, .3, .1, .6, .1, .1, .1])}
    miner = NegativeMining({"regions": 2}, surfaces, .5)
    report = miner.review(probabilities, 1)
    assert [(r["protein"], r["support"]) for r in miner.active] == [
        ("B", [3]), ("A", [0, 1])]
    assert report["components"] == 4
    assert report["selected"] == report["selected_proteins"] == 2
    reordered = NegativeMining({"regions": 2}, dict(reversed(list(surfaces.items()))), .5)
    reordered.review(dict(reversed(list(probabilities.items()))), 1)
    assert miner.state_dict() == reordered.state_dict()


def test_current_logit_loss_exact_area_batch_reduction_and_gradient():
    """No per-region normalization, stale score targets, positive labels or background gradient."""
    miner = NegativeMining({"regions": 1, "strong_weight": 2}, {"N": geometry(4)}, .5)
    miner.review({"N": np.array([.8, .3, .1, .1])}, 1)
    threshold = np.log(.25 / .75)
    logits = torch.tensor([threshold+1, threshold+2, 10., 10., 10., 10., 10., 10.],
                          dtype=torch.float64, requires_grad=True)
    loss = miner.loss(logits, torch.ones(8), torch.tensor([0, 4, 8]), ["N", "P"],
                      torch.tensor([0., 1.]))
    assert float(loss.detach()) == pytest.approx(2 * (1 + 4) / 4)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([1., 2., 0., 0., 0., 0., 0., 0.],
                                                        dtype=torch.float64))
    corrected = torch.full((8,), -3., requires_grad=True)
    value = miner.loss(corrected, torch.ones(8), torch.tensor([0, 4, 8]), ["N", "P"],
                       torch.tensor([0., 1.]))
    value.backward()
    assert float(value.detach()) == 0 and corrected.grad.count_nonzero() == 0


def test_memory_contracts_splits_merges_current_support_and_expiry():
    """Remembered descendants use only current supports and inherit the last priority epoch."""
    miner = NegativeMining({"regions": 1, "memory": True, "memory_epochs": 2},
                           {"N": geometry(5)}, 1.)
    miner.review({"N": np.array([.8, .5, .5, .5, .5])}, 1)
    first_id = miner.active[0]["id"]
    split = miner.review({"N": np.array([.8, .3, .1, .3, .35])}, 2)
    assert split["events_split"] == 1
    assert all(r["parents"] == [first_id] for r in miner.active)
    assert miner.active[1]["last_priority"] == 1
    assert miner.masks["N"][0][2] == miner.masks["N"][1][2] == 0
    merged = miner.review({"N": np.array([.8, .3, .3, .3, .35])}, 3)
    assert merged["events_merge"] == 1
    assert len(miner.active) == 1
    miner.review({"N": np.array([.3] * 5)}, 4)
    assert miner.active[0]["strength"] == .25
    miner.review({"N": np.array([.3] * 5)}, 5)
    expired = miner.review({"N": np.array([.3] * 5)}, 6)
    assert expired["events_expired"] == 1 and miner.active == []


def test_correction_lost_tracking_budget_and_json_restore():
    """A vanished support is not reused; restoration rejects a different discretization."""
    options = {"regions": 1, "memory": True}
    miner = NegativeMining(options, {"N": geometry(5)}, 1.)
    miner.review({"N": np.array([.8, .3, .3, .1, .1])}, 1)
    restored = NegativeMining(options, {"N": geometry(5)}, 1.)
    restored.load_state_dict(json.loads(json.dumps(miner.state_dict())))
    probability = {"N": np.array([.1, .3, .3, .35, .3])}
    assert restored.review(probability, 2) == miner.review(probability, 2)
    assert miner.last_audit["events_tracking_lost"] == 1
    with pytest.raises(ValueError, match="different geometry"):
        NegativeMining(options, {"N": geometry(6)}, 1.).load_state_dict(restored.state_dict())


def test_displaced_nearby_peak_is_not_reported_as_confirmed_correction():
    """No old-mask penalty survives, but a neighboring unresolved peak prevents a false claim."""
    miner = NegativeMining({"regions": 1, "memory": True}, {"N": geometry(5)}, 1.)
    miner.review({"N": np.array([.8, .1, .1, .1, .1])}, 1)
    report = miner.review({"N": np.array([.1, .3, .1, .1, .1])}, 2)
    assert report["events_tracking_lost"] == 1
    assert report["events_corrected"] == 0
    assert miner.active == []  # Below the high activation cut; no invented correspondence.


def test_area_matched_point_control_and_distributed_loss():
    """Point control matches area; distributed control uses whole-protein softplus."""
    probabilities = {"N": np.array([.8, .26, .1, .7, .1])}
    regional = NegativeMining({"regions": 1}, {"N": geometry(5)}, 1.)
    point = NegativeMining({"regions": 1, "mode": "point"}, {"N": geometry(5)}, 1.)
    regional.review(probabilities, 1)
    point.review(probabilities, 1)
    assert regional.masks["N"][0].sum() == point.masks["N"][0].sum() == 2
    assert point.masks["N"][0].tolist() == [1, 0, 0, 1, 0]
    distributed = NegativeMining({"mode": "distributed", "strong_weight": 1}, {}, .5)
    logits = torch.zeros(10, requires_grad=True)
    loss = distributed.loss(logits, torch.ones(10), torch.tensor([0, 5, 10]), ["N", "P"],
                            torch.tensor([0., 1.]))
    assert float(loss.detach()) == pytest.approx(np.log(2))


def test_no_negative_support_or_empty_batch_is_not_silent_corruption():
    """Absent train negatives is invalid; positive-only batches produce connected zero."""
    with pytest.raises(ValueError, match="training negatives"):
        NegativeMining({}, {}, 0.)
    miner = NegativeMining({}, {"N": geometry(3)}, .5)
    logits = torch.ones(3, requires_grad=True)
    loss = miner.loss(logits, torch.ones(3), torch.tensor([0, 3]), ["P"], torch.ones(1))
    loss.backward()
    assert loss == 0 and logits.grad.count_nonzero() == 0


def test_fgw_real_solver_conserved_masses_and_noise_selection():
    """N2 calls native POT-backed FGW and exposes all-noise without treating it as one family."""
    pytest.importorskip("ot")
    surfaces = {"A": geometry(7), "B": geometry(7)}
    selector = RegionDiversity(surfaces, {"representatives": 3})
    regions = [{"protein": name, "support": list(range(7)), "score": .9-i*.1}
               for i, name in enumerate(surfaces)]
    measure = selector.measure(regions[0], surfaces["A"])
    assert len(measure.weights) == 3 and measure.weights.sum() == pytest.approx(1)
    assert measure.structure.max() == pytest.approx(.6)
    labels, audit = selector.cluster(regions, surfaces)
    assert not audit["fallback"] and audit["comparisons"] == 1
    assert labels == [-1, -1]
    assert selector.order(regions, labels) == regions
    four = [{"id": i} for i in range(4)]
    assert [r["id"] for r in selector.order(four, [0, 0, 1, -1])] == [0, 2, 3, 1]
    selector.budget = 1
    labels, audit = selector.cluster(regions, surfaces)
    assert audit["fallback"] and "maximum_bytes" in audit["reason"]


def test_fgw_native_clustering_and_valid_only_statistics():
    """The density backend accepts actual FGW distances and missing values do not fit scales."""
    pytest.importorskip("ot")
    surfaces = {str(i): geometry(3) for i in range(5)}
    surfaces["0"].features[0, 0] = 1e9
    surfaces["0"].feature_valid[0, 0] = False
    selector = RegionDiversity(surfaces, {"min_cluster_size": 2, "min_samples": 2})
    assert selector.center[0] < 2
    regions = [{"protein": key, "support": [0, 1, 2], "score": .9}
               for key in surfaces]
    labels, audit = selector.cluster(regions, surfaces)
    assert len(labels) == 5 and not audit["fallback"]
    assert audit["comparisons"] == 10
    assert audit["clustering"]
    json.dumps(audit)


def test_native_continuation_replaces_epoch_and_restores_rng_optimizer_and_memory(tmp_path):
    """A single native file replaces prior epochs; no manual cache or checkpoint tree exists."""
    work = SimpleNamespace(checkpoints=CheckpointCollection(tmp_path / "checkpoints"))
    service = MiningContinuation(work)
    generator = torch.Generator().manual_seed(77)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    miner = NegativeMining({}, {"N": geometry(3)}, 1.)
    miner.review({"N": np.array([.9, .1, .1])}, 1)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "mining": miner.state_dict()}
    service.save(1, state, generator)
    expected = (random.random(), np.random.random(), torch.rand(3),
                torch.rand(3, generator=generator))
    loaded = service.restore(generator)
    actual = (random.random(), np.random.random(), torch.rand(3),
              torch.rand(3, generator=generator))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
    torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
    assert loaded["mining"] == miner.state_dict()
    restored = copy.deepcopy(model)
    restored.load_state_dict(loaded["model"])
    torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)
    service.save(2, state, generator)
    assert service.restore(generator)["epoch"] == 2
    assert len(list((tmp_path / "checkpoints").rglob("*.pt"))) == 1


def test_calibration_ties_resolution_threshold_and_overlap():
    """Conservative ranks use proteins, and an overlapping group is never called a reserve."""
    calibration = NegativeAdmission([0., 0., 1.])
    assert calibration.pvalues([0., 1., 2.]).tolist() == [1., .5, .25]
    assert not calibration.summary(.05)["admission_resolvable"]
    assert calibration.summary(.5)["scope"] == "exploratory_no_guarantee"
    assert NegativeAdmission.development_threshold([0., 1., 2., 3.], [0, 0, 1, 1]) == 2.
    with pytest.raises(ValueError, match="overlaps"):
        NegativeAdmission.verify_reserve([{"identifier": "A", "label": 0, "leakage_group": "G"}],
                                         [{"identifier": "B", "leakage_group": "G"}])


def test_admission_preserves_scores_and_reports_missing_local_denominators():
    """F0/F1/F2 only change decisions; empty predictions never fabricate precision or Dice."""
    logits = torch.tensor([-2., -2., 2.])
    original = logits.clone()
    snapshot = {"A": {"logits": logits, "label": 0, "protein_logit": 2.,
        "local_target": torch.zeros(3), "local_valid": torch.ones(3, dtype=torch.bool)}}
    calibration = NegativeAdmission([10.] * 19)
    report = calibration.evaluate(snapshot, {"A": geometry(3)}, [.05])
    assert report["procedures"]["F0"]["false_admission_rate"] == 1
    rank = report["procedures"]["F2_0.05"]
    assert rank["false_admission_rate"] == 0 and rank["records"][0]["precision"] is None
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
