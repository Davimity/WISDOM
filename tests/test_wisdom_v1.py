from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from torch import Tensor
from torch.utils.data import DataLoader

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.WisdomV2 import WisdomV2
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.ProteinPreprocessor import ProteinPreprocessor
from wisdom.Training import _create_model, _evaluate

MODEL_INPUT_NAMES = (
    "atomic_numbers",
    "residue_type_ids",
    "atom_edge_index",
    "atom_edge_types",
    "surface_curvatures",
    "surface_atom_neighbors",
    "surface_atom_distances",
    "surface_atom_normal_offsets",
    "surface_atom_tangential_distances",
    "surface_atom_mask",
    "surface_area_weights",
    "surface_batch",
    "surface_operators",
    "surface_ptr",
)


def _sample(atom_count: int, surface_count: int, target: float) -> dict[str, Tensor]:
    """Create one tiny schema-3 model sample with identity diffusion operators.

    Args:
        atom_count: Number of synthetic atoms; must be at least two.
        surface_count: Number of synthetic surface points; must be at least two.
        target: Protein-level binary target.

    Returns:
        Tensor mapping accepted directly by :class:`WisdomCollator`.
    """
    atom_edge_index = torch.stack(
        (torch.arange(atom_count - 1), torch.arange(1, atom_count))
    ).long()
    atom_edge_is_covalent = torch.arange(atom_count - 1).remainder(2) == 0
    atom_edge_spatial_rank = torch.arange(1, atom_count).remainder(2).add(1).long()

    table_width = 16
    neighbors   = torch.full((surface_count, table_width), -1, dtype=torch.long)
    distances   = torch.zeros(surface_count, table_width)
    normal      = torch.zeros(surface_count, table_width)
    tangential  = torch.zeros(surface_count, table_width)
    mask        = torch.zeros(surface_count, table_width, dtype=torch.bool)

    valid_width = min(atom_count, table_width)
    for point in range(surface_count):
        atom_ids        = torch.roll(torch.arange(atom_count), shifts=-point)[:valid_width]
        local_distances = torch.arange(1, valid_width + 1).float()

        neighbors[point, :valid_width]  = atom_ids
        distances[point, :valid_width]  = local_distances
        normal[point, :valid_width]     = 0.5 * local_distances
        tangential[point, :valid_width] = torch.sqrt(
            local_distances.square() - normal[point, :valid_width].square()
        )
        mask[point, :valid_width] = True

    curvatures = torch.zeros(surface_count, 2, 3)
    curvatures[:, 0, 0] = torch.linspace(0.1, 0.2, surface_count)
    curvatures[:, 0, 1] = curvatures[:, 0, 0].square()
    curvatures[:, 0, 2] = curvatures[:, 0, 0]
    curvatures[:, 1]    = curvatures[:, 0] * 0.5

    # Zero eigenvalues and an orthonormal identity basis make diffusion an exact identity in this
    # model-contract fixture. Dedicated tests exercise non-trivial geometric operators.

    gradient_index = torch.stack((torch.arange(surface_count), torch.arange(surface_count)))
    surface_neighbors = torch.full((surface_count, 4), -1, dtype=torch.long)
    surface_neighbor_distances = torch.zeros(surface_count, 4)
    surface_neighbor_mask = torch.zeros(surface_count, 4, dtype=torch.bool)
    for point in range(surface_count):
        surface_neighbors[point, 0]          = (point + 1) % surface_count
        surface_neighbor_distances[point, 0] = 1.0
        surface_neighbor_mask[point, 0]      = True

    return {
        "atomic_numbers": torch.arange(6, 6 + atom_count).long(),
        "residue_type_ids": (torch.arange(atom_count) % 20 + 1).long(),
        "atom_edge_index": atom_edge_index,
        "atom_edge_is_covalent": atom_edge_is_covalent,
        "atom_edge_spatial_rank": atom_edge_spatial_rank,
        "surface_curvatures": curvatures,
        "surface_atom_neighbors": neighbors,
        "surface_atom_distances": distances,
        "surface_atom_normal_offsets": normal,
        "surface_atom_tangential_distances": tangential,
        "surface_atom_mask": mask,
        "surface_area_weights": torch.arange(1, surface_count + 1).float(),
        "diffusion_mass": torch.ones(surface_count),
        "diffusion_eigenvalues": torch.zeros(surface_count),
        "diffusion_eigenvectors": torch.eye(surface_count),
        "diffusion_gradient_index": gradient_index.long(),
        "diffusion_gradient_x": torch.zeros(surface_count),
        "diffusion_gradient_y": torch.zeros(surface_count),
        "surface_positions": torch.column_stack(
            (torch.arange(surface_count).float(), torch.zeros(surface_count, 2))
        ),
        "surface_normals": torch.tensor([[0.0, 0.0, 1.0]]).repeat(surface_count, 1),
        "surface_neighbors": surface_neighbors,
        "surface_neighbor_distances": surface_neighbor_distances,
        "surface_neighbor_mask": surface_neighbor_mask,
        "target": torch.tensor(target),
    }


def _model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Select only the common model inputs from one collated mapping.

    Args:
        batch: Collated WISDOM tensor/operator mapping.

    Returns:
        Keyword mapping accepted by WISDOM v1 and v2.
    """
    return {name: batch[name] for name in MODEL_INPUT_NAMES}


def _model() -> WisdomV1:
    """Return the small deterministic v1 model shared by CPU smoke tests."""
    return WisdomV1(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=2,
        projection_depth=1,
        surface_layers=2,
        dropout=0.0,
        curvature_features=6,
        atom_spatial_k=2,
        surface_atom_k=16,
        diffusion_spectral_modes=8,
    )


def _write_npz(path: Path, sample: dict[str, Tensor]) -> None:
    """Write the model-facing portion of a pickle-free schema-3 archive.

    Args:
        path: Destination NPZ path, which may have no extension.
        sample: Synthetic sample returned by :func:`_sample`.
    """
    names = (
        "atomic_numbers",
        "residue_type_ids",
        "atom_edge_index",
        "atom_edge_is_covalent",
        "atom_edge_spatial_rank",
        "surface_curvatures",
        "surface_area_weights",
        "surface_atom_neighbors",
        "surface_atom_distances",
        "surface_atom_normal_offsets",
        "surface_atom_tangential_distances",
        "surface_atom_mask",
        "diffusion_mass",
        "diffusion_eigenvalues",
        "diffusion_eigenvectors",
        "diffusion_gradient_index",
        "diffusion_gradient_x",
        "diffusion_gradient_y",
        "surface_positions",
        "surface_normals",
        "surface_neighbors",
        "surface_neighbor_distances",
        "surface_neighbor_mask",
    )
    arrays = {name: sample[name].numpy() for name in names}
    arrays["metadata_json"] = np.asarray(
        json.dumps({"preprocessing_schema_version": "3.0"})
    )

    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_training_resolves_model_generations_by_convention() -> None:
    """Training discovers compatible generations without version-specific branches."""
    parameters = {
        "hidden_dim":                    16,
        "embedding_dim":                 4,
        "use_residue_type":              True,
        "atomic_layers":                 1,
        "projection_depth":              1,
        "surface_layers":                1,
        "dropout":                       0.0,
        "pooling_type":                  "mean",
        "topk_fraction":                 0.1,
        "attention_hidden_dim":          8,
        "regional_diffusion_scale":      2.5,
        "log_sum_exp_beta":              3.0,
        "curvature_features":            15,
        "atom_spatial_k":                8,
        "surface_atom_k":                8,
        "diffusion_spectral_modes":      64,
        "surface_atom_radius":           6.0,
        "surface_chunk_size":            128,
        "atomic_message_chunk_size":     256,
    }

    v1, v1_parameters = _create_model(1, parameters)
    v2, v2_parameters = _create_model(2, parameters)

    assert isinstance(v1, WisdomV1)
    assert not isinstance(v1, WisdomV2)
    assert v1.ARCHITECTURE_NAME == "bounded-atomic-diffusionnet"
    assert v1.STRUCTURAL_SCHEMA_VERSION == WisdomDataset.STRUCTURAL_SCHEMA_VERSION
    assert isinstance(v1.surface_encoder, DiffusionSurfaceEncoder)
    assert "pooling_type" not in v1_parameters
    assert v1.curvature_features == 15
    assert isinstance(v2, WisdomV2)
    assert v2_parameters["pooling_type"] == "mean"
    assert v2.curvature_features == 15

    with pytest.raises(ValueError, match="unsupported WISDOM model version"):
        _create_model(999, parameters)


def test_dataset_accepts_extensionless_managed_npz_assets(tmp_path: Path) -> None:
    """Managed assets use logical names and media types rather than filename extensions."""
    base = tmp_path / "assets" / "10AC_A" / "universal_npz"
    base.parent.mkdir(parents=True)
    _write_npz(base, _sample(3, 2, 0.0))

    DatasetIndex.write(
        tmp_path / "index.jsonl",
        (
            DatasetMember(
                member_id="10AC_A",
                partitions={"split": "train", "tier": "core"},
                targets={"dna_binding": 0},
                assets={
                    "universal_npz": DatasetAsset(
                        path="assets/10AC_A/universal_npz",
                        kind="file",
                        media_type="application/x-npz",
                    )
                },
            ),
        ),
    )

    dataset = WisdomDataset(tmp_path, "train")
    sample  = dataset[0]

    assert len(dataset) == 1
    assert sample["identifier"] == "10AC_A"
    assert sample["target"].item() == 0.0


def test_dataset_loads_schema3_inputs_and_optional_geometry(tmp_path: Path) -> None:
    """The hot path excludes coordinates while V3 can request explicit geometry."""
    first  = _sample(3, 2, 1.0)
    second = _sample(2, 3, 0.0)
    _write_npz(tmp_path / "first.npz", first)
    _write_npz(tmp_path / "second.npz", second)
    with (tmp_path / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("file", "label", "split"))
        writer.writerow(("first.npz", 1, "train"))
        writer.writerow(("second.npz", 0, "val"))

    loaded = WisdomDataset(tmp_path / "manifest.csv", "train")[0]

    assert loaded["atomic_numbers"].dtype == torch.int64
    assert loaded["surface_curvatures"].dtype == torch.float32
    assert loaded["target"].shape == () and loaded["target"].item() == 1.0
    assert torch.equal(loaded["atom_edge_spatial_rank"], first["atom_edge_spatial_rank"])
    assert "surface_positions" not in loaded

    geometry = WisdomDataset(
        tmp_path / "manifest.csv",
        "train",
        include_surface_geometry=True,
    )[0]

    assert geometry["surface_positions"].shape == (2, 3)
    assert geometry["surface_normals"].shape == (2, 3)


def test_dataset_rejects_legacy_schema_clearly(tmp_path: Path) -> None:
    """The materially incompatible edge-list representation never loads silently."""
    np.savez_compressed(tmp_path / "legacy.npz", atomic_numbers=np.asarray([6]))
    (tmp_path / "manifest.csv").write_text(
        "file,label,split\nlegacy.npz,1,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported pre-schema-3"):
        WisdomDataset(tmp_path / "manifest.csv", "train")[0]


def test_dataset_loads_only_required_surface_metric_targets(tmp_path: Path) -> None:
    """Epoch diagnostics avoid loading unrelated coordinates and soft sidecar arrays."""
    base = tmp_path / "protein.npz"
    _write_npz(base, _sample(3, 3, 1.0))

    sidecar = tmp_path / "protein.dna.npz"
    digest  = hashlib.sha256(base.read_bytes()).hexdigest()
    hard    = np.asarray([1, 0, 0], dtype=np.uint8)
    valid   = np.asarray([True, False, True], dtype=np.bool_)

    np.savez_compressed(
        sidecar,
        surface_target_hard=hard,
        surface_valid_mask=valid,
        surface_target_soft=hard.astype(np.float32),
        surface_distance_to_dna=np.asarray([0.5, 2.0, 4.0], dtype=np.float32),
        surface_distance_valid=np.ones(3, dtype=np.bool_),
        surface_target_hard_sensitivity=np.column_stack((hard, hard)),
        sensitivity_gaps=np.asarray([1.0, 1.4], dtype=np.float32),
        base_npz_sha256=np.asarray(digest),
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "file,annotation,label,split,identifier,tier\n"
        "protein.npz,protein.dna.npz,1,val,PROTEIN_A,core\n",
        encoding="utf-8",
    )

    sample = WisdomDataset(manifest, "val", include_surface_targets=True)[0]
    batch  = WisdomCollator()((sample,))

    assert torch.equal(sample["surface_target_hard"], torch.from_numpy(hard.astype(np.int64)))
    assert torch.equal(batch["surface_valid_mask"], torch.from_numpy(valid))
    assert "surface_target_soft" not in sample
    assert "surface_positions" not in sample


def test_evaluation_reports_surface_metrics_across_multiple_batches() -> None:
    """Validation joins point owners across batches without affecting protein metrics."""
    positive = _sample(3, 2, 1.0)
    positive["surface_target_hard"] = torch.tensor([1, 0])
    positive["surface_valid_mask"]  = torch.tensor([True, True])

    negative = _sample(2, 3, 0.0)
    negative["surface_target_hard"] = torch.tensor([0, 0, 0])
    negative["surface_valid_mask"]  = torch.tensor([True, True, True])

    for sample in (positive, negative):
        for name in (
            "surface_positions",
            "surface_normals",
            "surface_neighbors",
            "surface_neighbor_distances",
            "surface_neighbor_mask",
        ):
            del sample[name]

    loader = DataLoader((positive, negative), batch_size=1, collate_fn=WisdomCollator())
    protein_metrics, surface_metrics = _evaluate(
        _model(),
        loader,
        torch.device("cpu"),
        None,
    )

    assert protein_metrics["auprc"] is not None
    assert "mcc" in protein_metrics
    assert protein_metrics["loss"] is not None and protein_metrics["loss"] > 0.0
    assert surface_metrics["surface_valid_points"] == 5.0
    assert surface_metrics["surface_positive_proteins"] == 1.0
    assert surface_metrics["surface_positive_macro_auprc"] is not None


def test_dataset_reads_managed_partitions_targets_and_dilutions(tmp_path: Path) -> None:
    """Managed dataset membership remains explicit after the representation migration."""
    _write_npz(tmp_path / "first.npz", _sample(3, 2, 1.0))
    _write_npz(tmp_path / "second.npz", _sample(2, 3, 0.0))
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
            DatasetMember(
                member_id="VALIDATION_A",
                partitions={"split": "validation", "tier": "core"},
                targets={"dna_binding": 1},
                metadata={"dilutions": []},
                assets={"universal_npz": DatasetAsset(path="first.npz")},
            ),
            DatasetMember(
                member_id="TEST_A",
                partitions={"split": "test", "tier": "core"},
                targets={"dna_binding": 0},
                metadata={"dilutions": []},
                assets={"universal_npz": DatasetAsset(path="second.npz")},
            ),
        ),
    )

    full       = WisdomDataset(tmp_path, "train")
    small      = WisdomDataset(tmp_path, "train", subset="10pct")
    validation = WisdomDataset(tmp_path, "val", subset="10pct")
    test       = WisdomDataset(tmp_path, "test", subset="10pct")

    assert len(full) == 2
    assert len(small) == 1
    assert len(validation) == 1
    assert len(test) == 1
    assert small[0]["identifier"] == "FIRST_A"
    assert small[0]["tier"] == "core"
    assert validation[0]["identifier"] == "VALIDATION_A"
    assert test[0]["identifier"] == "TEST_A"


def test_collator_activates_nested_topology_and_offsets_atom_tables() -> None:
    """Disjoint batching derives relations and cannot create cross-protein references."""
    first  = _sample(3, 2, 1.0)
    second = _sample(2, 3, 0.0)

    # Rank two is outside the selected K=1 topology. The first edge remains because it is
    # covalent; the second non-covalent edge must disappear rather than reaching the model.

    first["atom_edge_is_covalent"] = torch.tensor([True, False])
    first["atom_edge_spatial_rank"] = torch.tensor([2, 2])
    batch  = WisdomCollator(atom_spatial_k=1)((first, second))

    assert batch["atom_edge_index"].shape == (2, 4)
    assert torch.equal(batch["atom_edge_types"], torch.ones(4, dtype=torch.long))
    assert set(batch["atom_edge_types"].tolist()) <= {0, 1, 2}
    assert torch.equal(batch["atom_batch"], torch.tensor([0, 0, 0, 1, 1]))
    assert torch.equal(batch["surface_batch"], torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(batch["surface_ptr"], torch.tensor([0, 2, 5]))

    first_atom_references = batch["surface_atom_neighbors"][:2][batch["surface_atom_mask"][:2]]
    second_atom_references = batch["surface_atom_neighbors"][2:][batch["surface_atom_mask"][2:]]
    assert torch.all(first_atom_references < 3)
    assert torch.all(second_atom_references >= 3)
    assert len(batch["surface_operators"]) == 2


def test_model_forward_max_pooling_backward_and_no_surface_graph() -> None:
    """V1 executes the complete bounded path without any surface edge-list input."""
    model = _model()
    batch = dict(WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0))))

    assert "surface_edge_index" not in batch
    assert "surface_atom_edge_index" not in batch

    outputs = model(**_model_inputs(batch))
    assert outputs["logits"].shape == (2,)
    assert outputs["surface_logits"].shape == (5,)
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.allclose(
        outputs["logits"],
        torch.stack((outputs["surface_logits"][:2].max(), outputs["surface_logits"][2:].max())),
    )

    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], batch["target"])
    loss.backward()

    modules = (
        model.atomic_number_embedding,
        model.atomic_encoder,
        model.surface_atom_transfer,
        model.surface_encoder,
        model.local_head,
    )
    for module in modules:
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert gradients and all(
            value is not None and torch.isfinite(value).all() for value in gradients
        )


def test_batched_logits_match_individual_proteins() -> None:
    """Operator packs and compact references preserve protein independence."""
    model  = _model().eval()
    first  = _sample(3, 2, 1.0)
    second = _sample(2, 3, 0.0)

    together = WisdomCollator()((first, second))
    separate = (WisdomCollator()((first,)), WisdomCollator()((second,)))

    with torch.no_grad():
        combined_logits = model(**_model_inputs(dict(together)))["logits"]
        separate_logits = torch.cat(
            [model(**_model_inputs(dict(batch)))["logits"] for batch in separate]
        )

    assert torch.allclose(combined_logits, separate_logits, rtol=1.0e-5, atol=1.0e-6)


def test_real_preprocessing_output_reaches_a_finite_protein_logit(
    tmp_path: Path,
    pdb_path: Path,
) -> None:
    """A real fixture passes directly from schema-3 preprocessing into v1."""
    config = PreprocessConfig(
        chains=("A",),
        surface_resolution=1.2,
        diffusion_spectral_modes_max=32,
    )
    identifiers = tmp_path / "proteins.txt"
    identifiers.write_text(f"{pdb_path}\n", encoding="utf-8")
    transformed = ProteinPreprocessor(config).transform(
        {"key": str(pdb_path), "identifier": str(pdb_path), "output_name": "tiny.npz"},
        identifiers,
        pdb_path.parent,
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    np.savez_compressed(
        processed / "tiny.npz",
        **transformed["arrays"],
        metadata_json=np.asarray(json.dumps(transformed["metadata"])),
    )

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file,label,split\nprocessed/tiny.npz,1,train\n", encoding="utf-8")
    batch = dict(
        WisdomCollator(diffusion_spectral_modes=32)(
            (WisdomDataset(manifest, "train")[0],)
        )
    )
    curvature_features = int(batch["surface_curvatures"].shape[1] * 3)
    model = WisdomV1(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=1,
        surface_layers=1,
        dropout=0.0,
        curvature_features=curvature_features,
        diffusion_spectral_modes=32,
    )

    outputs = model(**_model_inputs(batch))

    assert outputs["logits"].shape == (1,)
    assert outputs["surface_logits"].shape == (len(batch["surface_batch"]),)
    assert torch.isfinite(outputs["logits"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this environment")
def test_cuda_forward_backward_when_available() -> None:
    """The bounded operator containers support CUDA autocast and autograd."""
    model      = _model().cuda()
    host_batch = WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0)))

    def move(value: Any) -> Any:
        """Move one test-only nested model value to CUDA."""
        if isinstance(value, Tensor):
            return value.cuda()
        if isinstance(value, list):
            return [{key: tensor.cuda() for key, tensor in item.items()} for item in value]
        return value

    batch = {name: move(value) for name, value in host_batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**_model_inputs(batch))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["logits"], batch["target"]
        )

    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
