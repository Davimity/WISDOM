"""V10h initialization contracts against native LF outputs and inactive residue routes."""

# ruff: noqa: I001

import json
import torch
import pytest

from pathlib import Path
from types import SimpleNamespace
from test_v8_architectures import model
from test_wisdom_v1 import _sample, _write_npz
from wisdom.data.WisdomDataset import WisdomDataset
from lambdaforge.work.outputs import OutputCollection
from test_representation_campaign import chemical_batch
from wisdom.Training import _train_wisdom, _model_inputs, _move_to_device
from wisdom.models.atomic.representation.AtomicRepresentation import AtomicRepresentation


@pytest.mark.parametrize("extractor", ["esm2", "esmc", "saprot"])
@pytest.mark.parametrize("mode", ["static", "both"])
def test_static_initialization_declares_one_shared_audit(tmp_path, monkeypatch, extractor, mode):
    """Normalization and pretrained-table reports share one native output declaration.

    Args:
        tmp_path: Disposable NPZ fixtures and native attempt-owned artifact storage.
        monkeypatch: Injects frozen synthetic external assets and stops before any training.
        extractor: Frozen representation family; ESM-C uses width 960, the others 480.
        mode: Static table initialization alone or combined with per-residue context.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_get = WisdomDataset.__getitem__
    original_fit = AtomicRepresentation.fit
    width = 960 if extractor == "esmc" else 480
    weights = torch.randn(20, width, generator=torch.Generator().manual_seed(211))
    alphabet = json.dumps(list("ARNDCQEGHILKMFPSTWYV"))

    def with_static_table(dataset, index):
        """Supply the exact static asset fields at the validated-reader boundary.

        Args:
            dataset: Real synthetic-data WisdomDataset, never a production placement.
            index: Member index forwarded to the unmodified base reader.

        Returns:
            Normal model sample plus the already-extracted synthetic table and alphabet.
        """
        sample = original_get(dataset, index)
        sample[f"static_embedding_table_{extractor}__weights"] = weights
        sample[f"static_embedding_table_{extractor}__alphabet"] = alphabet
        return sample

    def fit_synthetic(representation, batches):
        """Exercise real fitting equations on explicit frozen toy context without a downloader.

        Args:
            representation: Real initialized V10 atomic input module.
            batches: Actual selected-training traversal, consumed to test the reader boundary.

        Returns:
            Real normalization report from the deterministic, two-protein chemical fixture.
        """
        assert len(tuple(batches)) == 2
        data = chemical_batch()
        members = data["structural_assets"]["external_embeddings_esm2"]["members"]
        if width == 960:
            members = [{**member, "embedding": member["embedding"].repeat(1, 2)}
                       for member in members]
        data["structural_assets"][f"external_embeddings_{extractor}"] = {"members": members}
        return original_fit(representation, [data])

    def stop_before_device(module, *args, **kwargs):
        """End this initialization-only test before optimization or GPU allocation.

        Args:
            module: Completed initialized model.
            args: Unused device-transfer positional arguments.
            kwargs: Unused device-transfer keyword arguments.

        Raises:
            RuntimeError: Deliberate sentinel marking successful pre-training initialization.
        """
        raise RuntimeError("synthetic initialization complete; no training requested")

    monkeypatch.setattr(WisdomDataset, "__getitem__", with_static_table)
    monkeypatch.setattr(AtomicRepresentation, "required_assets", staticmethod(lambda _: ()))
    monkeypatch.setattr(AtomicRepresentation, "fit", fit_synthetic)
    monkeypatch.setattr(torch.nn.Module, "to", stop_before_device)
    rows = ["file,label,split"]
    for index, label in enumerate([1., 0.]):
        sample = _sample(3, 5, label)
        sample["surface_curvatures"] = sample["surface_curvatures"][:, :1].repeat(1, 5, 1)
        _write_npz(tmp_path / f"protein-{index}.npz", sample)
        for split in ("train", "val"):
            rows.append(f"protein-{index}.npz,{int(label)},{split}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n")
    native = OutputCollection(SimpleNamespace(run_dir=tmp_path))
    declared = {}

    def directory(name, **kwargs):
        """Record native paths without weakening LF's duplicate-name rejection.

        Args:
            name: Exact logical output name supplied by actual Training code.
            kwargs: Output properties forwarded unchanged to native LF.

        Returns:
            Unmodified managed output, or propagates native registration errors.
        """
        output = native.directory(name, **kwargs)
        declared[name] = Path(output)
        return output

    work = SimpleNamespace(trial=None, run_dir=tmp_path, log=lambda _: None,
        outputs=SimpleNamespace(directory=directory),
        metrics=SimpleNamespace(log=lambda *args, **kwargs: None))
    with pytest.raises(RuntimeError, match="synthetic initialization complete"):
        _train_wisdom(work, manifest, model_version=2, hidden_dim=8, embedding_dim=4,
            atomic_hidden_dim=4, atomic_output_dim=8, atomic_encoder_type="schnet",
            architecture_implementation="controlled", gate_cost_normalization="baseline_count",
            atomic_representation={"sources": ["atom_name", "residue_atom"],
                "external_model": extractor, "external_mode": mode},
            gate_policy={}, evaluate_test=False, surface_metrics=False, data_workers=0)
    audit = declared["atomic-input-audit"]
    assert {p.name for p in audit.iterdir()} == {"normalization.json", "static-table.json"}
    static = json.loads((audit / "static-table.json").read_text())
    assert 0 < static["rank"] <= 19 and static["trainable"]
    normalization = json.loads((audit / "normalization.json").read_text())
    assert ("external_context" in normalization) == (mode == "both")


@pytest.mark.parametrize("extractor", ["esm2", "esmc", "saprot"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def test_replace_does_not_execute_zero_width_residue_embedding(monkeypatch, extractor, device):
    """A removed residue source must not invoke gather/backward kernels on an empty table.

    Args:
        monkeypatch: Replaces only the zero-width module's forward with a strict failure spy.
        extractor: Frozen representation family; no pretrained weights or real data are loaded.
        device: CPU always, plus CUDA when available, including an actual AdamW update.
    """
    options = {"sources": ["atom_name", "residue_atom"], "external_model": extractor,
               "external_mode": "replace"}
    predictor = model("baseline", atomic_encoder_type="schnet", atomic_representation=options)
    data = chemical_batch()
    members = data["structural_assets"]["external_embeddings_esm2"]["members"]
    if extractor == "esmc":
        members = [{**member, "embedding": member["embedding"].repeat(1, 2)}
                   for member in members]
    data["structural_assets"][f"external_embeddings_{extractor}"] = {"members": members}
    predictor.atomic_representation.fit([data])
    predictor.to(device)
    data = _move_to_device(data, torch.device(device))

    def forbidden(*args, **kwargs):
        """Reject lookup of an explicitly absent residue feature route.

        Args:
            args: Unexpected embedding inputs.
            kwargs: Unexpected embedding options.

        Raises:
            AssertionError: Always; correct replacement bypasses this module entirely.
        """
        raise AssertionError("zero-width residue lookup must not execute")

    embedding = predictor.residue_type_embedding
    assert embedding.weight.shape == (21, 0)
    monkeypatch.setattr(embedding, "forward", forbidden)
    use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bfloat16):
        output = predictor(**_model_inputs(data))
    output["logits"].sum().backward()
    assert embedding.weight.grad is None
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.001)
    optimizer.step()
    assert embedding.weight not in optimizer.state
    restored = model("baseline", atomic_encoder_type="schnet", atomic_representation=options)
    restored.to(device)
    restored.load_state_dict(predictor.state_dict(), strict=True)
    predictor.eval()
    restored.eval()
    tolerance = 1e-5 if device == "cuda" else 0
    torch.testing.assert_close(predictor(**_model_inputs(data))["logits"],
                               restored(**_model_inputs(data))["logits"],
                               rtol=tolerance, atol=tolerance)
