"""Distinct all-model assets share geometry and retain native per-protein resume boundaries."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_structural_enrichment import _chemistry, _save, _work

from wisdom.preprocessing.common.structure.assets.context.SequenceContext import SequenceContext
from wisdom.preprocessing.common.structure.assets.external.ExternalProteinRepresentation import (
    ExternalProteinRepresentation,
)
from wisdom.preprocessing.common.structure.assets.external.ProteinModelAssets import (
    ProteinModelAssets,
)
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars


def _models(root):
    specs = {}
    for kind in ("esm2", "esmc", "saprot"):
        directory = root / kind
        directory.mkdir(parents=True)
        path = directory / "config.json"
        path.write_text("{}")
        specs[kind] = {
            "checkpoint": str(directory),
            "identifier": "test/" + kind,
            "revision": "frozen-test",
            "files": {"config.json": {"bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}},
        }
    (root / "checkpoints.json").write_text(json.dumps({"models": specs}))
    return root


def test_all_model_inventory_and_byte_integrity_are_required_before_extraction(tmp_path):
    root = _models(tmp_path)
    producer = ProteinModelAssets(root)
    assert len(producer.requested_families) == 6
    (root / "esmc/config.json").write_text('{"modified":true}')
    with pytest.raises(ValueError, match="integrity mismatch"):
        ProteinModelAssets(root)


@pytest.mark.parametrize("kinds,static_tables", [
    (None, True), (["esm2"], False), (["esmc"], True), (["saprot"], False),
])
def test_model_qualified_assets_do_not_overwrite_and_resume_without_inference(
    tmp_path, monkeypatch, kinds, static_tables
):
    root = _models(tmp_path / "models")
    protein, arrays = _chemistry()
    base, sequence, tokens = [tmp_path / name for name in ("base.npz", "seq.npz", "tokens.npz")]
    _save(base, arrays)
    SequenceContext().write(base, sequence, protein)
    count = len(protein.chains[0].residues)
    StructuralAsset.write(
        tokens,
        base,
        {"token": np.full(count, "a", dtype="U1"), "valid": np.ones(count, np.bool_)},
        "structure_tokens",
        {"method": "official-structureto3didescriptor"},
    )
    calls = []

    def extractor(self, kind):
        def encode(words):
            calls.append(kind)
            return np.full((len(words), 4), {"esm2": 1, "esmc": 2, "saprot": 3}[kind], np.float32)

        return SimpleNamespace(
            kind=kind,
            dimension=4,
            encode=encode,
            static_table=lambda: (np.ones((20, 4), np.float32), list("ACDEFGHIKLMNPQRSTVWY")),
            metadata={
                "extractor": kind,
                "dimension": 4,
                "checkpoint_sha256": "a" * 64,
                "frozen": True,
            },
        )

    monkeypatch.setattr(ProteinModelAssets, "extractor", extractor)
    producer = ProteinModelAssets(root, kinds=kinds, static_tables=static_tables)
    work = _work(tmp_path / "run")
    assets = {"member": {"sequence_context": sequence}}
    if producer.requires_structure_tokens:
        assets["member"]["structure_tokens"] = tokens
    result = producer.generate(work, {"member": base}, assets)
    assert set(result["member"]) == producer.requested_families
    assert set(calls) == set(producer.models)
    StructuralSidecars.validate_assets(base, result["member"])
    for kind, value in (("esm2", 1), ("esmc", 2), ("saprot", 3)):
        if kind not in producer.models:
            assert f"external_embeddings_{kind}" not in result["member"]
            continue
        path = result["member"]["external_embeddings_" + kind]
        with np.load(path, allow_pickle=False) as stored:
            assert np.all(stored["embedding"][stored["valid"]] == value)
    calls.clear()
    assert producer.generate(work, {"member": base}, assets) == result
    assert not calls
    report = json.loads((work.run_dir / "protein-model-report/report.json").read_text())
    assert report["complete"] and not report["failures"]


def test_unselected_checkpoints_are_not_required_or_loaded(tmp_path, monkeypatch):
    """A selected ESM-2 bundle does not inspect missing or changed ESM-C/SaProt weights."""
    root = _models(tmp_path)
    (root / "esmc/config.json").unlink()
    (root / "saprot/config.json").write_text("unselected fixture changed")
    producer = ProteinModelAssets(root, kinds=["esm2"], static_tables=False)
    assert set(producer.models) == {"esm2"}
    assert producer.requested_families == {"external_embeddings_esm2"}
    assert not producer.requires_structure_tokens
    static_table = Mock(side_effect=AssertionError("unrequested static table"))
    load = Mock(return_value=SimpleNamespace(static_table=static_table))
    monkeypatch.setattr(producer, "extractor", load)
    producer.preflight()
    load.assert_called_once_with("esm2")
    static_table.assert_not_called()
    with pytest.raises(ValueError, match="inventory changed"):
        ProteinModelAssets(root)


@pytest.mark.parametrize("kinds", [[], ["esm2", "esm2"], ["not-a-model"]])
def test_invalid_model_selection_cannot_publish_an_empty_or_ambiguous_bundle(tmp_path, kinds):
    """Reject empty, duplicate or unknown selections before extraction can start."""
    root = _models(tmp_path)
    with pytest.raises(ValueError):
        ProteinModelAssets(root, kinds=kinds)


def test_masked_saprot_does_not_invent_missing_structural_tokens(tmp_path):
    protein, arrays = _chemistry()
    base, sequence, tokens, output = [
        tmp_path / name for name in ("base.npz", "seq.npz", "tokens.npz", "embedding.npz")
    ]
    _save(base, arrays)
    SequenceContext().write(base, sequence, protein)
    count = len(protein.chains[0].residues)
    StructuralAsset.write(
        tokens,
        base,
        {"token": np.full(count, "#", dtype="U1"), "valid": np.zeros(count, np.bool_)},
        "structure_tokens",
        {},
    )
    extractor = SimpleNamespace(
        kind="saprot",
        dimension=4,
        metadata={
            "extractor": "saprot",
            "dimension": 4,
            "checkpoint_sha256": "a" * 64,
            "frozen": True,
        },
    )
    ExternalProteinRepresentation(extractor).write(base, output, sequence, tokens)
    with np.load(output, allow_pickle=False) as stored:
        assert not stored["valid"].any() and not stored["embedding"].any()
