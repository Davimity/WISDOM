"""Exact LF topology preparation and reuse outside the neural global-attention hot path."""

import csv
import types

import numpy as np
import pytest
import torch
from lambdaforge.nn import GraphStructuralEncoding
from lambdaforge.work.cache import WorkCache
from test_v9_atomic_encoders import batch, model
from test_wisdom_v1 import _sample, _write_npz

from wisdom.data.AtomicStructuralInputs import AtomicStructuralInputs
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.Training import _device_batch, _model_inputs


def test_managed_codes_restore_without_bfs_and_invalidate_changed_topology(tmp_path, monkeypatch):
    """A Run reuses fixed code files, but never aliases another K/topology or a corrupt file.

    Args:
        tmp_path: Isolated source and actual native managed-cache roots.
        monkeypatch: Pytest fixture counting real LF path construction calls.
    """
    source = tmp_path / "base.npz"
    _write_npz(source, _sample(5, 4, 1.0))
    logs = []
    work = types.SimpleNamespace(cache=WorkCache(tmp_path / "cache"), log=logs.append)
    native = GraphStructuralEncoding.from_edges
    calls = []

    def tracked(*args, **kwargs):
        """Count actual backend calls rather than substituting a graph algorithm.

        Args:
            args: Positional arguments forwarded to the public LF constructor.
            kwargs: Keyword arguments forwarded to the public LF constructor.

        Returns:
            Unmodified LF structural encoding.
        """
        calls.append(args[1])
        return native(*args, **kwargs)

    monkeypatch.setattr(GraphStructuralEncoding, "from_edges", tracked)
    service = AtomicStructuralInputs(2)
    # The implementation digest is stable across equivalent service instances; test patching
    # changes no graph equations and is part of this fixture's exact implementation identity.

    rng = torch.random.get_rng_state().clone()
    paths = service.prepare(work, [source, source])
    assert calls == [5, 5] and torch.equal(rng, torch.random.get_rng_state())
    assert AtomicStructuralInputs(2).prepare(work, [source]) == paths
    assert calls == [5, 5]
    cached = paths[source]
    assert service.valid(cached, nodes=5)
    with np.load(cached, allow_pickle=False) as values:
        for relation in service.RELATIONS:
            assert values[f"{relation}_codes"].dtype == np.uint8
    assert AtomicStructuralInputs(1).prepare(work, [source])[source] != cached
    assert len(calls) == 4
    cached.write_bytes(b"incomplete cache file")
    service.prepare(work, [source])
    assert len(calls) == 6 and service.valid(cached, nodes=5)
    _write_npz(source, _sample(4, 4, 0.0))
    assert service.prepare(work, [source])[source] != cached
    assert len(calls) == 8


def test_preparation_guards_all_members_before_building(tmp_path, monkeypatch):
    """A later oversized molecule fails before the first expensive graph construction.

    Args:
        tmp_path: Isolated small/large source archives and managed cache.
        monkeypatch: Pytest fixture forbidding any quadratic path allocation.
    """
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _write_npz(first, _sample(3, 4, 0.0))
    _write_npz(second, _sample(5, 4, 1.0))
    service = AtomicStructuralInputs(2, maximum_bytes=128)
    work = types.SimpleNamespace(cache=WorkCache(tmp_path / "cache"), log=lambda _: None)

    def forbidden(*args, **kwargs):
        """Fail if any backend allocation precedes the cheap all-member capacity scan.

        Args:
            args: Unused backend positional arguments.
            kwargs: Unused backend keyword arguments.

        Raises:
            AssertionError: Always; expected rejection is before path construction.
        """
        raise AssertionError("unexpected quadratic graph work")

    monkeypatch.setattr(GraphStructuralEncoding, "from_edges", forbidden)
    with pytest.raises(MemoryError, match=r"atoms=5.*200.*128"):
        service.prepare(work, [first, second])


def test_dataset_collator_device_path_reuses_the_exact_local_relations(tmp_path, monkeypatch):
    """Prepare, shuffle and transfer unequal members without mixing or rebuilding their codes.

    Args:
        tmp_path: Isolated schema-3 fixtures and native file cache.
        monkeypatch: Pytest fixture forbidding backend work during repeated model forwards.
    """
    sources = [tmp_path / "a.npz", tmp_path / "b.npz"]
    for source, count in zip(sources, [3, 5], strict=True):
        _write_npz(source, _sample(count, 4, 1.0))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "label", "split"])
        for source in sources:
            writer.writerow([source.name, 1, "train"])
    dataset = WisdomDataset(manifest, "train")
    service = AtomicStructuralInputs(2)
    work = types.SimpleNamespace(cache=WorkCache(tmp_path / "cache"), log=lambda _: None)
    dataset.atomic_structure_paths = service.prepare(work, sources)
    samples = [dataset[1], dataset[0]]
    collated = WisdomCollator(atom_spatial_k=2)(samples)
    assert collated["atomic_operators"][0]["spatial_codes"].shape == (5, 5)
    assert collated["atomic_operators"][0]["spatial_codes"].dtype == torch.uint8
    moved = _device_batch(collated, torch.device("cpu"))
    assert moved["atomic_operators"][0]["spatial_codes"].dtype == torch.long
    assert "atomic_operators" in _model_inputs(moved)
    with pytest.raises(ValueError, match="another spatial K"):
        WisdomCollator(atom_spatial_k=1)(samples)

    def forbidden(*args, **kwargs):
        """Prove the complete training input path no longer calls the public BFS builder.

        Args:
            args: Unused backend positional arguments.
            kwargs: Unused backend keyword arguments.

        Raises:
            AssertionError: Always, including evaluation and repeated layers.
        """
        raise AssertionError("path construction in neural execution")

    monkeypatch.setattr(GraphStructuralEncoding, "from_edges", forbidden)
    predictor = model("structural_global").eval()
    expected = predictor(**_model_inputs(moved))["logits"]
    actual = predictor(**_model_inputs(moved))["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires prepared atomic_operators"):
        predictor(**{k: v for k, v in _model_inputs(moved).items() if k != "atomic_operators"})


@pytest.mark.parametrize("training", [False, True])
def test_prepared_global_matches_original_forward_and_parameter_gradients(training, monkeypatch):
    """Moving fixed graph work preserves outputs, RNG consumption and every parameter gradient.

    Args:
        training: Enable original gate sampling and dropout, or deterministic evaluation.
        monkeypatch: Pytest fixture providing the exact pre-optimization public-LF execution.
    """
    predictor = model("structural_global").train(training)
    data = batch()
    encoder = predictor.atomic_encoder

    def reference(self, operator, hidden, edges, atom_batch, atomic_operators, relation):
        """Recreate the former forward-local BFS with unchanged weights and relation topology.

        Args:
            self: Original encoder and its per-protein resource bound.
            operator: Public LF structural attention block.
            hidden: Atomic states [N,D].
            edges: Complete directed relation [2,E].
            atom_batch: Explicit molecule ownership [N].
            atomic_operators: Unused precomputed packs; the reference recomputes their contents.
            relation: Unused name; the directed input already selects the exact relation.

        Returns:
            Original-order [N,D] result from the unmodified LF attention operator.
        """
        output = torch.zeros_like(hidden)
        for owner in torch.unique(atom_batch, sorted=True):
            ids = torch.where(atom_batch == owner)[0]
            inverse = torch.full((len(hidden),), -1, dtype=torch.long)
            inverse[ids] = torch.arange(len(ids))
            local = inverse[edges[:, atom_batch[edges[0]] == owner]]
            encoding = GraphStructuralEncoding.from_edges(
                local, len(ids), max_pairwise_bytes=self.maximum_pairwise_bytes,
            )
            output[ids] = operator(
                hidden[ids][None], encoding.spatial_indices[None],
                encoding.in_degree[None], encoding.out_degree[None],
            )[0]
        return output

    torch.manual_seed(641)
    with monkeypatch.context() as context:
        context.setattr(encoder, "_global", types.MethodType(reference, encoder))
        expected = predictor(**_model_inputs(data))
        expected["logits"].sum().backward()
        rng = torch.random.get_rng_state().clone()
    gradients = {name: p.grad.clone() for name, p in predictor.named_parameters()
                 if p.grad is not None}
    keys = tuple(predictor.state_dict())
    predictor.zero_grad(set_to_none=True)
    torch.manual_seed(641)
    actual = predictor(**_model_inputs(data))
    actual["logits"].sum().backward()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert keys == tuple(predictor.state_dict())
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for name, parameter in predictor.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
