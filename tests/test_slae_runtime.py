"""Exercise the native isolated SLAE boundary without installing external model packages."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from lambdaforge.controlplane.ToolRuntimeSpecification import ToolRuntimeSpecification

from wisdom.preprocessing.common.structure.assets.external.AtomicProteinExtractor import (
    AtomicProteinExtractor,
)
from wisdom.preprocessing.common.structure.assets.external.SLAEExtraction import SLAEExtraction
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive


@pytest.fixture
def slae_loading_boundary(tmp_path, monkeypatch):
    """Provide real PyTorch weights behind the optional upstream package boundary.

    Args:
        tmp_path: Disposable trusted checkpoint; no production model is downloaded or replaced.
        monkeypatch: Bind the fake upstream API and expected digest for this checkpoint only.

    Returns:
        Checkpoint path, independent encoder state and upstream API with a real Linear module.
    """
    checkpoint = tmp_path / "weights.ckpt"
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
             "bias": torch.tensor([0.25, -0.5])}
    encoder = {**state, "allegro.env_sum_normalizations": torch.tensor([0.1172, 0.1172]),
               "edge_eng_sum.per_edge_scales": torch.empty(0)}
    torch.save({"encoder": encoder, "decoder": {"unused": torch.ones(1)}}, checkpoint)
    monkeypatch.setattr(SLAEExtraction, "CHECKPOINT_SHA256", ProteinArchive.sha256(checkpoint))
    api = SimpleNamespace(ProteinEncoder=Mock(side_effect=lambda **_: torch.nn.Linear(3, 2)),
                          default_encoder_config=Mock(return_value={"out_dim": 2}),
                          load_autoencoder=Mock(side_effect=AssertionError("decoder is unused")),
                          encode=Mock())
    modules = {
        "SLAE": api,
        "SLAE.util.constants": SimpleNamespace(PROTEIN_ATOMS=SLAEExtraction.ATOM_NAMES,
                                               STANDARD_RESIDUE_ATOMS={"ALA": ()}),
        "torch_geometric.data": SimpleNamespace(Data=Mock()),
    }
    # Only replace this module's importlib reference, not Python's global import machinery.

    monkeypatch.setattr(
        "wisdom.preprocessing.common.structure.assets.external.SLAEExtraction.importlib",
        SimpleNamespace(import_module=modules.__getitem__),
    )
    return checkpoint, state, api


def test_encoder_load_ignores_only_retired_buffers_and_preserves_weights(slae_loading_boundary):
    """Use real strict state loading and compare numerical outputs after the narrow migration.

    Args:
        slae_loading_boundary: Byte-verified fixture checkpoint and fake public constructor.
    """
    checkpoint, expected, api = slae_loading_boundary
    before = checkpoint.read_bytes()
    extractor = SLAEExtraction(checkpoint, "cpu")
    for name, value in extractor.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    with torch.inference_mode():
        torch.testing.assert_close(extractor.model(x), x @ expected["weight"].T + expected["bias"])
    api.ProteinEncoder.assert_called_once_with(out_dim=2)
    api.load_autoencoder.assert_not_called()
    assert not extractor.model.training
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("defect", ["missing", "unexpected", "wrong_shape"])
def test_encoder_load_still_rejects_incompatible_weights(
    slae_loading_boundary, monkeypatch, defect,
):
    """A legacy-buffer allowance must never hide a damaged or mismatched trained encoder.

    Args:
        slae_loading_boundary: Isolated source checkpoint and mocked upstream constructor.
        monkeypatch: Update fixture trust only, so real strict weight validation is exercised.
        defect: Missing weight, unknown parameter or incompatible parameter dimensions.
    """
    checkpoint, _, _ = slae_loading_boundary
    stored = torch.load(checkpoint, weights_only=True)
    if defect == "missing":
        del stored["encoder"]["weight"]
    elif defect == "unexpected":
        stored["encoder"]["new_trainable_weight"] = torch.ones(1)
    else:
        stored["encoder"]["weight"] = torch.zeros(2, 4)
    torch.save(stored, checkpoint)
    monkeypatch.setattr(SLAEExtraction, "CHECKPOINT_SHA256", ProteinArchive.sha256(checkpoint))
    with pytest.raises(RuntimeError, match="state_dict"):
        SLAEExtraction(checkpoint, "cpu")


def test_encoder_load_checks_bytes_before_deserialization(tmp_path, monkeypatch):
    """Untrusted pickle files must fail their digest guard before torch.load can execute.

    Args:
        tmp_path: Disposable invalid file, never a researcher checkpoint.
        monkeypatch: Observe the deserialization boundary without decoding untrusted bytes.
    """
    checkpoint = tmp_path / "invalid.ckpt"
    checkpoint.write_bytes(b"unreviewed checkpoint")
    load = Mock(side_effect=AssertionError("must not deserialize untrusted bytes"))
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="reviewed"):
        SLAEExtraction(checkpoint, "cpu")
    load.assert_not_called()


def test_declared_slae_runtime_is_separate_and_stages_only_standalone_code():
    """Parse the actual public LF schema and reject accidental consumer/framework imports."""
    root = Path(__file__).resolve().parents[1]
    specs = {spec.name: spec for spec in ToolRuntimeSpecification.discover(root)}
    spec = specs["slae"]
    assert spec.policy["torch_cuda"] == "12.4"
    assert "torch==2.6.0+cu124" in spec.policy["requirements"]
    assert "transformers==4.48.1" in spec.policy["requirements"]
    assert list(spec.entrypoints) == ["extract"]
    tree = ast.parse(spec.entrypoints["extract"].read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(name.split(".")[0] in {"wisdom", "lambdaforge"} for name in names)


def test_isolated_extraction_validates_output_and_keeps_base_unchanged(tmp_path, monkeypatch):
    """Exchange explicit files through native tools and validate physical alignment on return.

    Args:
        tmp_path: Disposable checkpoint, universal atomic fields and native scratch.
        monkeypatch: Replace only the external tool/checkpoint trust boundaries, not validation.
    """
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"trusted-test-weights")
    monkeypatch.setattr(AtomicProteinExtractor, "CHECKPOINT_SHA256",
                        ProteinArchive.sha256(checkpoint))
    base = tmp_path / "parent.npz"
    arrays = {
        "atom_positions": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32),
        "residue_indices": np.array([0, 0, 1], np.int32),
        "atom_names": np.array(["N", "CA", "ZN"]),
        "residue_names": np.array(["ALA", "ALA", "ZN"]),
        "atomic_numbers": np.array([7, 6, 30]),
        "chain_indices": np.array([0, 0, 1]),
    }
    np.savez(base, **arrays)
    original = base.read_bytes()
    features = {
        "embedding": np.concatenate((np.ones((2, 128)), np.zeros((1, 128)))).astype(np.float32),
        "valid": np.array([True, True, False]),
        "residue_embedding": np.concatenate((np.ones((1, 128)), np.zeros((1, 128)))).astype(
            np.float32),
        "residue_valid": np.array([True, False]),
        "atom_parent": np.array([0, 0, 1], np.int32),
        "atom_slot": np.array([0, 1, -1], np.int16),
    }
    tool = Mock(binding={"policy": {"torch_cuda": "12.4"}})
    tool.entrypoint.return_value = tmp_path / "staged-extract.py"
    work = SimpleNamespace(tools=Mock(), temp_dir=tmp_path, log=Mock())
    work.tools.runtime.return_value = tool

    def run(command, **options):
        """Stand in for the external process while leaving all host-side validation real."""
        assert options["cwd"]
        if "--output" in command:
            np.savez(command[command.index("--output") + 1], **features)

    tool.run.side_effect = run
    extractor = AtomicProteinExtractor(checkpoint, "cuda", work=work)
    work.tools.runtime.assert_called_once_with("slae")
    tool.preflight.assert_called_once_with()
    assert "--probe" in tool.run.call_args_list[0].args[0]
    output = tmp_path / "asset.npz"
    extractor.write(base, output)
    assert extractor.validate(output, base)
    assert base.read_bytes() == original
    assert "device" not in extractor.parameters
    with np.load(output, allow_pickle=False) as stored:
        header = json.loads(str(stored["metadata_json"]))
        assert header["atom_coverage"]["atom_name"]["ZN"]["supported"] == 0

    # A plausible but misaligned external result must never become a publishable sidecar.

    features["atom_slot"][1] = 0
    with pytest.raises(ValueError, match="slots"):
        extractor.write(base, tmp_path / "invalid.npz")


def test_wrong_checkpoint_fails_before_requesting_a_runtime(tmp_path):
    """Do not touch a prepared environment for an unreviewed or damaged weight file."""
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"not-the-reviewed-checkpoint")
    work = SimpleNamespace(tools=Mock())
    with pytest.raises(ValueError, match="reviewed"):
        AtomicProteinExtractor(checkpoint, "cuda", work=work)
    work.tools.runtime.assert_not_called()


def test_standalone_graph_budget_prevents_pair_table_allocation():
    """Count directed neighbors before allocating edges or invoking the upstream encoder."""
    extractor = object.__new__(SLAEExtraction)
    extractor.names = ("N", "CA")
    extractor.residue_names = {"ALA"}
    extractor.maximum_edges, extractor.radius = 1, 8.0
    extractor.encode_api = Mock()
    arrays = {
        "atom_positions": np.array([[0, 0, 0], [1, 0, 0]], np.float32),
        "residue_indices": np.array([0, 0]),
        "atom_names": np.array(["N", "CA"]),
        "residue_names": np.array(["ALA", "ALA"]),
        "atomic_numbers": np.array([7, 6]),
    }
    with pytest.raises(MemoryError, match="budget"):
        extractor.encode(arrays)
    extractor.encode_api.assert_not_called()
