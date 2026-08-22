from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.preprocessing import PreprocessingRecord
from lambdaforge.tasks import TaskContext
from torch import Tensor

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.models.WisdomV1 import WisdomV1
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessPipeline import PreprocessPipeline


def _sample(atom_count: int, surface_count: int, target: float) -> dict[str, Tensor]:
    atom_edges = torch.stack(
        (torch.arange(atom_count - 1), torch.arange(1, atom_count))
    ).long()
    surface_edges = torch.stack(
        (torch.arange(surface_count - 1), torch.arange(1, surface_count))
    ).long()
    surface_ids = torch.arange(surface_count).repeat_interleave(2)
    atom_ids = torch.stack(
        (
            torch.arange(surface_count) % atom_count,
            (torch.arange(surface_count) + 1) % atom_count,
        ),
        dim=1,
    ).flatten()
    curvatures = torch.zeros(surface_count, 2, 3)
    curvatures[:, 0, 0] = torch.linspace(0.1, 0.2, surface_count)
    curvatures[:, 0, 1] = curvatures[:, 0, 0].square()
    curvatures[:, 0, 2] = curvatures[:, 0, 0]
    curvatures[:, 1] = curvatures[:, 0] * 0.5
    return {
        "atomic_numbers": torch.arange(6, 6 + atom_count).long(),
        "residue_type_ids": (torch.arange(atom_count) % 20 + 1).long(),
        "atom_edge_index": atom_edges,
        "atom_edge_types": torch.arange(atom_count - 1).remainder(3).long(),
        "surface_curvatures": curvatures,
        "surface_edge_index": surface_edges,
        "surface_atom_edge_index": torch.stack((surface_ids, atom_ids)),
        "surface_area_weights": torch.arange(1, surface_count + 1).float(),
        "target": torch.tensor(target),
    }


def _model() -> WisdomV1:
    return WisdomV1(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=2,
        projection_depth=1,
        surface_layers=2,
        dropout=0.0,
        curvature_features=6,
    )


def _write_npz(path: Path, sample: dict[str, Tensor]) -> None:
    np.savez_compressed(
        path,
        atomic_numbers=sample["atomic_numbers"].numpy().astype(np.uint8),
        residue_type_ids=sample["residue_type_ids"].numpy().astype(np.uint8),
        atom_edge_index=sample["atom_edge_index"].numpy().astype(np.int32),
        atom_edge_relation_mask=(sample["atom_edge_types"].numpy() + 1).astype(np.uint8),
        surface_curvatures=sample["surface_curvatures"].numpy().astype(np.float32),
        surface_edge_index=sample["surface_edge_index"].numpy().astype(np.int32),
        surface_atom_edge_index=sample["surface_atom_edge_index"].numpy().astype(np.int32),
        surface_area_weights=sample["surface_area_weights"].numpy().astype(np.float32),
    )


def test_dataset_loads_explicit_split_dtypes_and_relation_mapping(tmp_path: Path) -> None:
    first = _sample(3, 2, 1.0)
    second = _sample(2, 3, 0.0)
    _write_npz(tmp_path / "first.npz", first)
    _write_npz(tmp_path / "second.npz", second)
    with (tmp_path / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("file", "label", "split"))
        writer.writerow(("first.npz", 1, "train"))
        writer.writerow(("second.npz", 0, "val"))

    dataset = WisdomDataset(tmp_path / "manifest.csv", "train")
    loaded = dataset[0]
    assert len(dataset) == 1
    assert loaded["atomic_numbers"].dtype == torch.int64
    assert loaded["surface_curvatures"].dtype == torch.float32
    assert loaded["target"].shape == () and loaded["target"].item() == 1.0
    assert torch.equal(loaded["atom_edge_types"], first["atom_edge_types"])


def test_dataset_reads_managed_index_partitions_targets_and_dilutions(tmp_path: Path) -> None:
    first  = _sample(3, 2, 1.0)
    second = _sample(2, 3, 0.0)
    _write_npz(tmp_path / "first.npz", first)
    _write_npz(tmp_path / "second.npz", second)
    DatasetIndex.write(
        tmp_path / "index.jsonl",
        (
            DatasetMember(
                member_id="FIRST_A",
                partitions={"split": "train", "tier": "core"},
                targets={"dna_binding": 1},
                metadata={"dilutions": ["10pct", "25pct"]},
                assets={"universal_npz": DatasetAsset(path="first.npz")},
            ),
            DatasetMember(
                member_id="SECOND_A",
                partitions={"split": "train", "tier": "challenge"},
                targets={"dna_binding": 0},
                metadata={"dilutions": ["25pct"]},
                assets={"universal_npz": DatasetAsset(path="second.npz")},
            ),
        ),
    )

    full  = WisdomDataset(tmp_path, "train")
    small = WisdomDataset(tmp_path, "train", subset="10pct")

    assert len(full) == 2
    assert len(small) == 1
    assert small[0]["identifier"] == "FIRST_A"
    assert small[0]["tier"] == "core"
    assert small[0]["target"].item() == 1.0


def test_dataset_rejects_corrupt_relation_semantics(tmp_path: Path) -> None:
    sample = _sample(3, 2, 1.0)
    sample["atom_edge_types"][0] = 3
    _write_npz(tmp_path / "corrupt.npz", sample)
    (tmp_path / "manifest.csv").write_text(
        "file,label,split\ncorrupt.npz,1,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relation masks"):
        WisdomDataset(tmp_path / "manifest.csv", "train")[0]


def test_collator_offsets_every_graph_domain_and_builds_batch_vectors() -> None:
    first = _sample(3, 2, 1.0)
    second = _sample(2, 3, 0.0)
    batch = WisdomCollator()((first, second))

    assert torch.equal(
        batch["atom_edge_index"],
        torch.tensor([[0, 1, 1, 2, 3, 4], [1, 2, 0, 1, 4, 3]]),
    )
    assert torch.equal(batch["atom_edge_types"], torch.tensor([0, 1, 0, 1, 0, 0]))
    assert torch.equal(
        batch["surface_edge_index"],
        torch.tensor([[0, 1, 2, 3, 3, 4], [1, 0, 3, 4, 2, 3]]),
    )
    expected_bipartite = torch.cat(
        (
            first["surface_atom_edge_index"],
            second["surface_atom_edge_index"] + torch.tensor([[2], [3]]),
        ),
        dim=1,
    )
    assert torch.equal(batch["surface_atom_edge_index"], expected_bipartite)
    assert torch.equal(batch["atom_batch"], torch.tensor([0, 0, 0, 1, 1]))
    assert torch.equal(batch["surface_batch"], torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(batch["target"], torch.tensor([1.0, 0.0]))


def test_model_forward_pooling_backward_and_cpu_contract() -> None:
    model = _model()
    batch = WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0)))
    outputs = model(
        **{
            name: batch[name]
            for name in (
                "atomic_numbers",
                "residue_type_ids",
                "atom_edge_index",
                "atom_edge_types",
                "surface_curvatures",
                "surface_edge_index",
                "surface_atom_edge_index",
                "surface_area_weights",
                "surface_batch",
            )
        }
    )
    assert outputs["logits"].shape == (2,)
    assert outputs["surface_logits"].shape == (5,)
    assert torch.isfinite(outputs["logits"]).all()

    expected = torch.stack(
        (
            outputs["surface_logits"][:2].max(),
            outputs["surface_logits"][2:].max(),
        )
    )
    assert torch.allclose(outputs["logits"], expected)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], batch["target"])
    loss.backward()
    modules = (
        model.atomic_number_embedding,
        model.atomic_encoder,
        model.surface_encoder,
        model.local_head,
    )
    for module in modules:
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert gradients and all(
            value is not None and torch.isfinite(value).all() for value in gradients
        )


def test_surface_permutation_preserves_protein_logits() -> None:
    model = _model().eval()
    batch = dict(WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0))))
    input_names = (
        "atomic_numbers",
        "residue_type_ids",
        "atom_edge_index",
        "atom_edge_types",
        "surface_curvatures",
        "surface_edge_index",
        "surface_atom_edge_index",
        "surface_area_weights",
        "surface_batch",
    )
    original = model(**{name: batch[name] for name in input_names})["logits"]

    permutation = torch.tensor([1, 0, 4, 2, 3])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    batch["surface_curvatures"] = batch["surface_curvatures"][permutation]
    batch["surface_area_weights"] = batch["surface_area_weights"][permutation]
    batch["surface_batch"] = batch["surface_batch"][permutation]
    batch["surface_edge_index"] = inverse[batch["surface_edge_index"]]
    batch["surface_atom_edge_index"][0] = inverse[batch["surface_atom_edge_index"][0]]

    permuted = model(**{name: batch[name] for name in input_names})["logits"]
    assert torch.allclose(original, permuted, rtol=1.0e-5, atol=1.0e-6)


def test_real_preprocessing_output_reaches_a_finite_protein_logit(
    tmp_path: Path, pdb_path: Path
) -> None:
    config      = PreprocessConfig(chains=("A",), surface_resolution=1.2)
    identifiers = tmp_path / "proteins.txt"
    identifiers.write_text(f"{pdb_path}\n", encoding="utf-8")
    context = TaskContext(
        name="model-preprocess-fixture",
        run_dir=tmp_path,
        source_dir=tmp_path,
        attempt_id="model-preprocess-attempt",
        config_fingerprint="model-preprocess-fingerprint",
        resume=False,
        inputs=(
            {
                "name": "protein_identifiers",
                "path": str(identifiers),
                "resolved_path": str(identifiers),
                "sha256": "fixture-identifiers",
                "size_bytes": identifiers.stat().st_size,
            },
            {
                "name": "local_structures",
                "path": str(pdb_path.parent),
                "resolved_path": str(pdb_path.parent),
                "sha256": "fixture-structures",
                "size_bytes": 0,
            },
        ),
        outputs={"downloads": "raw"},
    )
    transformed = PreprocessPipeline(config, download=False).transform(
        PreprocessingRecord(
            key=str(pdb_path),
            value=str(pdb_path),
            metadata={"output_name": "tiny.npz"},
        ),
        context,
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    np.savez_compressed(processed / "tiny.npz", **transformed.value["arrays"])

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file,label,split\nprocessed/tiny.npz,1,train\n", encoding="utf-8")
    batch = WisdomCollator()((WisdomDataset(manifest, "train")[0],))
    input_names = (
        "atomic_numbers",
        "residue_type_ids",
        "atom_edge_index",
        "atom_edge_types",
        "surface_curvatures",
        "surface_edge_index",
        "surface_atom_edge_index",
        "surface_area_weights",
        "surface_batch",
    )

    outputs = _model()(**{name: batch[name] for name in input_names})

    assert outputs["logits"].shape == (1,)
    assert outputs["surface_logits"].shape == (len(batch["surface_batch"]),)
    assert torch.isfinite(outputs["logits"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this environment")
def test_cuda_forward_backward_when_available() -> None:
    model = _model().cuda()
    batch = {
        name: value.cuda()
        for name, value in WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0))).items()
    }
    outputs = model(
        **{
            name: batch[name]
            for name in (
                "atomic_numbers",
                "residue_type_ids",
                "atom_edge_index",
                "atom_edge_types",
                "surface_curvatures",
                "surface_edge_index",
                "surface_atom_edge_index",
                "surface_area_weights",
                "surface_batch",
            )
        }
    )
    torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["logits"], batch["target"]
    ).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
