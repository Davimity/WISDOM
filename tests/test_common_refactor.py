"""Offline contracts for common preprocessing, weighted fields and site-level Zn evidence."""

import ast
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
from test_surface_features_and_zinc import _assembly, _base

from wisdom.features.SurfaceFeatureProjector import SurfaceFeatureProjector
from wisdom.features.SurfaceFeatureSchema import SurfaceFeatureSchema
from wisdom.features.SurfaceFeatureSidecar import SurfaceFeatureSidecar
from wisdom.interpretability.SparseConceptDiscovery import SparseConceptDiscovery
from wisdom.preprocessing.common.functional_metadata import annotate_functions
from wisdom.preprocessing.common.population import select_population
from wisdom.preprocessing.common.splits import assign_splits
from wisdom.preprocessing.common.structure.AtomicStructureBuilder import AtomicStructureBuilder
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.StructureSource import StructureSource
from wisdom.preprocessing.common.structure.SurfaceBuilder import SurfaceBuilder
from wisdom.preprocessing.dna.DNAValidation import DNAValidation
from wisdom.preprocessing.zinc import phenotypes
from wisdom.preprocessing.zinc.evidence import annotate_sites
from wisdom.preprocessing.zinc.structures import _surface_points
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination


def test_common_infrastructure_has_no_task_imports():
    """Common algorithms must not acquire DNA/Zn evidence through a hidden import."""
    root = Path(__file__).parents[1] / "src/wisdom/preprocessing/common"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            assert not any(module.startswith(("wisdom.preprocessing.dna",
                                              "wisdom.preprocessing.zinc")) for module in modules)


def test_historical_local_phenotype_CSV_can_be_plotted_without_rewriting_it(tmp_path):
    """Legacy phenotype tables remain useful report inputs, not new registry mutations."""
    descriptors, clusters = tmp_path / "descriptors", tmp_path / "clusters"
    descriptors.mkdir()
    clusters.mkdir()
    source = descriptors / "positive-interface-features.csv"
    labels = clusters / "positive-interface-phenotypes.csv"
    source.write_text("identifier,x,y\nA,1,2\nB,3,4\n")
    labels.write_text("identifier,interface_phenotype\nA,L1\nB,L2\n")
    original = labels.read_bytes()
    output = tmp_path / "plot.png"
    DNAValidation._plot_phenotype_pca(tmp_path, output)
    assert output.stat().st_size > 0
    assert labels.read_bytes() == original


@pytest.mark.parametrize("weighting,expected", [
    ("pooled_points", 10 / 3), ("pooled_area", 30 / 7), ("equal_protein", 5),
])
def test_weighted_statistics_match_manual_population_moments(tmp_path, monkeypatch,
                                                            weighting, expected):
    """Point, area and protein measures give different, explicitly reproducible means."""
    values = {"first": np.array([[0.], [0.]], dtype=np.float32),
              "second": np.array([[10.]], dtype=np.float32)}
    records = []
    for name, areas in (("first", [1., 3.]), ("second", [3.])):
        base, sidecar = tmp_path / f"{name}.npz", tmp_path / name
        np.savez(base, surface_area_weights=np.array(areas, dtype=np.float32))
        sidecar.write_text(name)
        records.append((name, base, sidecar))
    monkeypatch.setattr(SurfaceFeatureSidecar, "read",
                        staticmethod(lambda path, *_: values[path.name]))
    stats = SurfaceFeatureSidecar.fit_statistics(records, ("hydropathy",), weighting)
    assert stats["mean"] == pytest.approx([expected])
    weights = {"pooled_points": [1., 1., 1.], "pooled_area": [1., 3., 3.],
               "equal_protein": [.5, .5, 1.]}[weighting]
    variance = np.average((np.array([0., 0., 10.]) - expected) ** 2, weights=weights)
    assert stats["std"] == pytest.approx([np.sqrt(variance)])
    assert stats["sources"][1]["point_count"] == 1
    assert stats["sources"][1]["feature_sha256"] == hashlib.sha256(b"second").hexdigest()


def test_normalization_audits_exact_views_and_undefined_correlations(tmp_path):
    """Fits bind exact train membership; unsupported columns do not get invented correlations."""
    base, sidecar = tmp_path / "base.npz", tmp_path / "fields.npz"
    _base(base)
    names = ("hydropathy", "formal_charge_density")
    SurfaceFeatureSidecar.write(base, sidecar, names)
    full = SurfaceFeatureSidecar.fit_statistics([("A", base, sidecar)], names)
    reduced = SurfaceFeatureSidecar.fit_statistics([("A", base, sidecar)], names,
                                                   population="replicate-00/train-25")
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    stats = {"schema_version": "2.0", "populations": {
        "full": full, "replicate-00/train-25": reduced}}
    populations = {view: {"A": digest} for view in stats["populations"]}
    assert SurfaceFeatureSidecar.validate_statistics(stats, names, populations)
    assert full["audit"]["correlation"][0][1] is None
    assert "hydropathy" in full["audit"]["unsupported_channels"]
    with pytest.raises(ValueError, match="non-train or changed"):
        SurfaceFeatureSidecar.validate_statistics(stats, names,
            {**populations, "replicate-00/train-25": {"VALIDATION": digest}})
    with pytest.raises(ValueError, match="misaligned"):
        SurfaceFeatureSidecar.normalize(np.zeros((1, 2)), names, {**full, "mean": [0.]})


def test_corrected_strict_donors_exclude_backbone_and_broad_element_duplicates(tmp_path):
    """Sidechain donor eligibility must not make every nearby N/O atom a Zn donor."""
    base = tmp_path / "base.npz"
    _base(base)
    with np.load(base, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["atom_names"] = np.array(["O", "OXT", "N"])
    arrays["atomic_numbers"] = np.array([8, 8, 7])
    names = ("O_density", "N_density", "zn_lewis_strict",
             "zn_donor_O_density", "zn_donor_N_density")
    projected = SurfaceFeatureProjector().project(arrays, names)
    assert projected[:, :2].sum() > 0
    assert np.all(projected[:, 2:] == 0)


def test_residue_averages_ignore_non_CA_votes(tmp_path):
    """Changing a non-CA atom's residue cannot change residue-average hydropathy/polarity."""
    base = tmp_path / "base.npz"
    _base(base)
    with np.load(base, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["atom_names"] = np.array(["CA", "CA", "N"])
    projector = SurfaceFeatureProjector()
    before = projector.project(arrays, ("hydropathy", "polarity"))
    arrays["residue_names"][2] = "TRP"
    after = projector.project(arrays, ("hydropathy", "polarity"))
    assert np.array_equal(before, after)


def test_historical_feature_vocabulary_is_not_silently_renamed():
    """Old stored names keep old semantics; new authoring requires corrected names."""
    SurfaceFeatureSchema.validate_stored(("zn_N_density",), "1.0")
    with pytest.raises(ValueError, match="unsupported"):
        SurfaceFeatureSchema.validate_stored(("zn_N_density",), "2.0")
    with pytest.raises(ValueError, match="unknown"):
        SurfaceFeatureSchema.resolve(("zn_N_density",))


@pytest.mark.parametrize("mode", ["explicit", "hybrid"])
def test_sparse_concepts_reject_nonlearned_evidence_before_model_reconstruction(tmp_path, mode):
    """X/Z cannot masquerade as the learned H coordinates used by sparse concepts."""
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_version": 1,
                "model_parameters": {"surface_representation_mode": mode}}, checkpoint)
    with pytest.raises(ValueError, match="learned-only surface H"):
        SparseConceptDiscovery()._load_predictor(checkpoint, torch.device("cpu"))


def test_site_evidence_preserves_residue_multiplicity_and_order_independent_legacy():
    """Current coordination adds counts/angles; legacy donor order is not scientific evidence."""
    sites = ZincCoordination().analyse(_assembly(), "AQ", 1)
    accepted = next(site for site in sites if site["accepted"])
    assert accepted["selected_residue_counts"] == {"ASP": 1, "HIS": 1}
    assert accepted["donor_angles_degrees"] == pytest.approx([180.])
    assert accepted["coordination_distance_mean"] == pytest.approx(2.)
    prior = [{key: value for key, value in site.items() if key not in {
        "coordination_schema", "selected_residue_counts", "donor_angles_degrees"}}
        for site in sites]
    for site in prior:
        site["partners"] = [dict(partner) for partner in reversed(site["partners"])]
    assert ZincCoordination.matches(sites, prior)
    prior[0]["selected_donor_count"] += 1
    assert not ZincCoordination.matches(sites, prior)


def test_early_surface_support_reuses_exact_boundary_and_sparse_signed_gaps():
    """Cheap support sampling uses build's points; unequal donor radii still give exact gaps."""
    positions = np.array([[0., 0., 0.], [2., 0., 0.]])
    radii = np.array([1.7, 1.5])
    builder = SurfaceBuilder(resolution=1.2)
    points, _ = builder.sample(positions, radii)
    built, _ = builder.build(positions, radii)
    assert np.array_equal(points.astype(np.float32), built["surface_positions"])
    gaps = ZincCoordination.surface_gaps(points, positions, radii)
    brute = np.min(np.linalg.norm(points[:, None] - positions, axis=2) - radii, axis=1)
    assert np.allclose(gaps, brute, atol=1e-6)


def test_global_phenotypes_include_both_classes_and_local_fits_use_each_site(monkeypatch):
    """Separate fits cannot drop positive morphology or average several sites into one."""
    site = {"accepted": True, "selected_donor_count": 4, "selected_residues": [1, 2, 3, 4],
            "selected_element_counts": {"N": 2, "S": 2}, "selected_residue_counts": {
                "CYS": 2, "HIS": 2}, "interchain": False,
            "coordination_distance_mean": 2., "coordination_distance_std": .1}
    rows = [{"identifier": "POS_A", "label": 1, "quality_eligible": True,
             "sites": [{**site, "site_id": "one"}, {**site, "site_id": "two"}]},
            {"identifier": "NEG_A", "label": 0, "quality_eligible": True, "sites": []}]
    calls = []

    def fitted(records, *args):
        """Record the two input populations without requiring an unstable tiny clustering fit."""
        calls.append(records)
        return {row["identifier"]: f"cluster-{i}" for i, row in enumerate(records)}, {}

    monkeypatch.setattr(phenotypes, "stable_phenotypes", fitted)
    assigned, _ = phenotypes.assign_phenotypes(rows, (2, 3), 2, .7, 1)
    assert {row["label"] for row in calls[0]} == {0, 1}
    assert len(calls[1]) == 2
    assert calls[1][0]["CYS_count"] == 2
    assert assigned[0]["local_phenotype"] == "cluster-0|cluster-1"


def test_buried_members_force_their_whole_leakage_group_to_train():
    """A global-only positive may never put its homolog in a held-out split."""
    rows = [{"identifier": str(i), "label": i % 2, "leakage_group": f"G{i // 2}",
             "global_phenotype": "global", "local_phenotype": "local", "origin": "fixture"}
            for i in range(40)]
    selected, _ = assign_splits(rows, .7, .15, .15, 2026, training_only_ids=("1",))
    assert {row["split"] for row in selected if row["leakage_group"] == "G0"} == {"train"}


def test_keep_all_positive_policy_reports_real_ratio_without_hiding_the_tail():
    """All-positive mode retains diversity instead of falsely claiming class balance."""
    rows = [{"identifier": str(i), "label": int(i >= 2), "quality_eligible": True,
             "leakage_group": f"G{i}", "global_phenotype": "global",
             "local_phenotype": "local", "origin": "reviewed"} for i in range(7)]
    selected, report = select_population(rows, 1., True, False, 2026, keep_all_positives=True)
    assert len(selected) == 7
    assert report["positive_negative_ratio"] == 2.5
    quota, _ = select_population(rows, 1., True, False, 2026)
    assert len(quota) == 4


def test_early_exposure_uses_deposition_sampling_before_assembly_rotation(cif_path):
    """Voxel sampling must not change because an assembly rotates the deposited molecule."""
    digest = hashlib.sha256(cif_path.read_bytes()).hexdigest()
    row = {"identifier": "TEST_A", "pdb_id": "TEST", "protein_chain": "A"}
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([10., 20., 30.])
    sampled = _surface_points(cif_path, row, digest, 1., 1.4, rotation, translation)
    source = StructureSource("TEST_A", "TEST", ("A",), cif_path, digest, "mmcif", False)
    protein, provenance = ProteinReader(PreprocessConfig()).read(source)
    atoms = [atom for chain in protein.chains for residue in chain.residues
             for atom in residue.atoms]
    # The universal atom builder stores float32 coordinates and radii before surface sampling.
    from_atoms = np.array([atom.position for atom in atoms], dtype=np.float32)
    arrays = AtomicStructureBuilder(radius=6.).build(protein)
    assert np.array_equal(from_atoms, arrays["atom_positions"])
    points, _ = SurfaceBuilder(1., 1.4).sample(arrays["atom_positions"], arrays["vdw_radii"])
    expected = (points.astype(np.float32).astype(np.float64) + provenance.coordinate_origin)
    assert np.array_equal(sampled, expected @ rotation.T + translation)


def test_external_site_annotations_bind_coordinates_without_changing_acceptance(tmp_path):
    """A reviewed database link may enrich a site, never manufacture coordination evidence."""
    rows = [{"identifier": "A", "structure_sha256": "digest", "label": 1,
             "sites": [{"site_id": "ZN0001", "accepted": True}]}]
    entry = {"identifier": "A", "site_id": "ZN0001", "structure_sha256": "digest",
             "source": "reviewed-MetalPDB-export", "version": "snapshot-1", "external_id": "site"}
    path = tmp_path / "sites.jsonl"
    path.write_text(json.dumps(entry) + "\n")
    annotated = annotate_sites(rows, path)[0]
    assert annotated["sites"][0]["accepted"] is True
    assert annotated["sites"][0]["external_metadata"]["external_id"] == "site"
    path.write_text(json.dumps({**entry, "structure_sha256": "different"}) + "\n")
    with pytest.raises(ValueError, match="exact coordinates"):
        annotate_sites(rows, path)


def test_coordination_distinguishes_Cys2His2_from_Cys1His3():
    """Counting distinct residue types alone must not collapse different coordination motifs."""
    assembly = _assembly()
    chain = assembly.model.find_chain("AQ")
    chain[0].name, chain[0][0].name, chain[0][0].element = "CYS", "SG", gemmi.Element("S")
    chain[1].name, chain[1][0].name, chain[1][0].element = "HIS", "ND1", gemmi.Element("N")
    for index in range(2):
        residue = chain[index].clone()
        residue.seqid = gemmi.SeqId(index + 3, " ")
        residue[0].pos = gemmi.Position(0., 2. if index == 0 else -2., 0.)
        chain.add_residue(residue)
    first = ZincCoordination().analyse(assembly, "AQ", 1)[0]
    assert first["selected_residue_counts"] == {"CYS": 2, "HIS": 2}
    chain[2].name, chain[2][0].name, chain[2][0].element = "HIS", "ND1", gemmi.Element("N")
    second = ZincCoordination().analyse(assembly, "AQ", 1)[0]
    assert second["selected_residue_counts"] == {"CYS": 1, "HIS": 3}
    assert first["selected_residue_names"] == second["selected_residue_names"]


def test_frozen_functional_metadata_binds_sequence_without_changing_labels(tmp_path):
    """Family provenance may guide an audit, never override binding evidence or group identity."""
    row = {"identifier": "A", "sequence_sha256": "abc", "label": 0, "leakage_group": "G"}
    entry = {"identifier": "A", "sequence_sha256": "abc", "source": "reviewed-source",
             "version": "frozen-1", "family": "family", "label": 1}
    path = tmp_path / "functional.jsonl"
    path.write_text(json.dumps(entry) + "\n")
    annotated = annotate_functions([row], path)[0]
    assert annotated["label"] == 0 and annotated["leakage_group"] == "G"
    assert annotated["family"] == "family"
    assert annotated["functional_metadata_provenance"]["input_sha256"] == (
        hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_text(json.dumps({**entry, "sequence_sha256": "other"}) + "\n")
    with pytest.raises(ValueError, match="exact sequence"):
        annotate_functions([row], path)
