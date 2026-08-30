from __future__ import annotations

# Project style orders complete import lines by length rather than lexicographically.
# ruff: noqa: I001

from lambdaforge.data import DatasetMember

from wisdom.visualization.Visualization import Visualization
from wisdom.preprocessing.structure.ProteinVisualizer import ProteinVisualizer


def test_visualization_sampling_cycles_through_split_and_label_strata() -> None:
    members = tuple(
        DatasetMember(
            member_id = f"{split}-{label}-{index}",
            partitions = {"split": split},
            targets = {"dna_binding": label},
        )
        for split in ("train", "validation", "test")
        for label in (0, 1)
        for index in range(3)
    )

    selected = Visualization._select_members(
        members,
        identifiers      = (),
        splits           = ("train", "validation", "test"),
        labels           = (0, 1),
        maximum_proteins = 12,
    )

    first_round = {
        (member.partitions["split"], member.targets["dna_binding"])
        for member in selected[:6]
    }
    assert first_round == {
        (split, label)
        for split in ("train", "validation", "test")
        for label in (0, 1)
    }
    assert len(selected) == 12


def test_visualization_explicit_identifiers_preserve_order_and_ignore_sample_cap() -> None:
    members = tuple(
        DatasetMember(member_id=f"protein-{index}")
        for index in range(4)
    )

    selected = Visualization._select_members(
        members,
        identifiers      = ("protein-3", "protein-1"),
        splits           = (),
        labels           = (),
        maximum_proteins = 1,
    )

    assert [member.member_id for member in selected] == ["protein-3", "protein-1"]


def test_visualization_zero_limit_retains_every_eligible_member() -> None:
    members = tuple(
        DatasetMember(
            member_id = f"protein-{index}",
            partitions = {"split": "test"},
            targets = {"dna_binding": index % 2},
        )
        for index in range(7)
    )

    selected = Visualization._select_members(
        members,
        identifiers      = (),
        splits           = ("test",),
        labels           = (0, 1),
        maximum_proteins = 0,
    )

    assert len(selected) == len(members)


def test_visualization_index_exposes_search_and_scientific_filters() -> None:
    reports = (
        {
            "identifier": "1ABC_A",
            "split": "test",
            "label": 1,
            "html": "proteins/1ABC_A.html",
            "ply": "proteins/1ABC_A.ply",
            "diagnostics": {"status": "PASS"},
        },
    )

    html = Visualization._index_html(reports)

    assert 'id="search"' in html
    assert 'id="split"' in html
    assert 'id="label"' in html
    assert 'id="status"' in html
    assert "1ABC_A" in html
    assert "Waals envelopes" in html


def test_protein_page_exposes_opaque_surface_and_uniform_mesh_controls() -> None:
    """The generated viewer must let researchers control depth and mesh interpretation."""
    page = ProteinVisualizer._html(
        identifier      = "1ABC_A",
        status          = "PASS",
        diagnostic_rows = "",
        inventory_rows  = "",
        provenance      = "{}",
        plot            = '<div id="wisdom-plot"></div>',
        controls        = '{"meshVertexCount":0}',
    )

    assert 'id="surface-size"' in page
    assert 'id="surface-opacity"' in page
    assert 'id="mesh-colour-mode"' in page
    assert '<option value="uniform" selected>' in page
    assert 'id="mesh-opacity"' in page
