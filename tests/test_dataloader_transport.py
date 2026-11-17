"""Exact protein batching under low descriptor limits, including spawned decoder workers."""

# ruff: noqa: I001

import ast
import torch
import pytest
import resource

from pathlib import Path
from torch import Tensor
from collections.abc import Mapping
from test_wisdom_v1 import _sample
from torch.utils.data import Dataset, DataLoader, get_worker_info
from wisdom.data.WisdomCollator import WisdomCollator


class TensorRichProteins(Dataset):
    """Generate small independent tensors, not production data or a GPU workload."""

    def __len__(self):
        """Return two sixteen-protein batches, enough to exercise concurrent prefetched packets."""
        return 32

    def __getitem__(self, index):
        """Produce a deterministic sample with many independent optional-asset storages.

        Args:
            index: Synthetic member index determining labels and exact tensor values.

        Returns:
            Ordinary WisdomCollator input with 24 protein-local sidecar fields.

        Raises:
            AssertionError: A subprocess did not receive the native transport initialization
                or the deliberately constrained descriptor limit used by this test.
        """
        if get_worker_info() is not None:
            assert torch.multiprocessing.get_sharing_strategy() == "file_system"
            assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 256
        sample = _sample(3, 5, float(index % 2))
        for field in range(24):
            sample[f"atomic_context__field_{field}"] = torch.tensor([index, field])
        return sample


def _initialize_limited_worker(worker_id):
    """Constrain only the disposable test worker before running WISDOM's actual callback.

    Args:
        worker_id: Native DataLoader index forwarded unchanged to the production callback.
    """
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    WisdomCollator.initialize_worker(worker_id)


def _assert_exact(actual, expected):
    """Compare every tensor, dtype, shape and nested ownership field without tolerances.

    Args:
        actual: Batch received through real multiprocessing queues.
        expected: Equivalent single-process collator output.
    """
    if isinstance(expected, Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_exact(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_exact(left, right)
    else:
        assert actual == expected


def _storage_addresses(value):
    """Count physical independent storages, not the number of tensor references.

    Args:
        value: Nested CPU batch returned by the real protein collator.

    Returns:
        Set of storage addresses that would each need a cached file descriptor under the
        default strategy; aliases and slices are counted only once.
    """
    if isinstance(value, Tensor):
        return {value.untyped_storage().data_ptr()} if value.numel() else set()
    if isinstance(value, Mapping):
        return set().union(*(_storage_addresses(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_storage_addresses(item) for item in value))
    return set()


def test_worker_initialization_preserves_seed_and_os_limits():
    """Selecting native transport changes neither native RNG state nor OS limits/parent policy."""
    previous = torch.multiprocessing.get_sharing_strategy()
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    rng = torch.random.get_rng_state().clone()
    try:
        WisdomCollator.initialize_worker(0)
        assert torch.multiprocessing.get_sharing_strategy() == "file_system"
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert resource.getrlimit(resource.RLIMIT_NOFILE) == limits
    finally:
        torch.multiprocessing.set_sharing_strategy(previous)


@pytest.mark.parametrize("context", torch.multiprocessing.get_all_start_methods())
def test_tensor_rich_batches_survive_low_nofile_and_persistent_epochs(context):
    """Send more storages than the worker's descriptor limit without altering a single value.

    Args:
        context: Real spawn, fork or forkserver context supported by the host.
    """
    previous = torch.multiprocessing.get_sharing_strategy()
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    data = TensorRichProteins()
    collator = WisdomCollator()
    expected = [collator([data[index] for index in range(start, start + 16)])
                for start in (0, 16)]
    assert len(_storage_addresses(expected[0])) > 256

    # A single ordinary batch exceeds the constrained worker's descriptor budget. Persistent
    # epochs also verify that transport reuse does not accumulate handles or change ordering.

    loader = DataLoader(data, batch_size=16, num_workers=2, collate_fn=collator,
        worker_init_fn=_initialize_limited_worker, multiprocessing_context=context,
        persistent_workers=True, prefetch_factor=1, timeout=30)
    for _ in range(2):
        received = list(loader)
        _assert_exact(received, expected)
    del loader
    assert torch.multiprocessing.get_sharing_strategy() == previous
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == limits


@pytest.mark.parametrize("module", ["Training.py", "visualization/ReviewInference.py",
                                   "visualization/ModelValidation.py"])
def test_all_parallel_protein_loaders_initialize_transport(module):
    """Keep training, validation and saved-model inference on the same native transport hook.

    Args:
        module: Production module containing actual DataLoader construction, never a test stub.
    """
    source = Path(__file__).parents[1] / "src" / "wisdom" / module
    tree = ast.parse(source.read_text())
    hook = "WisdomCollator.initialize_worker"
    assert any(isinstance(node, ast.keyword) and node.arg == "worker_init_fn"
               and ast.unparse(node.value) == hook for node in ast.walk(tree))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or ast.unparse(node.func) != "DataLoader":
            continue
        keywords = {argument.arg: argument.value for argument in node.keywords}
        if isinstance(keywords.get("num_workers"), ast.Constant):
            assert keywords["num_workers"].value == 0
            continue
        if None in keywords:
            assert ast.unparse(keywords[None]) == "loader_options"
        else:
            assert ast.unparse(keywords["worker_init_fn"]) == hook
