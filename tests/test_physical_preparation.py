"""Bounded chemical-completion contracts; synthetic fixtures do not validate molecular physics."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gemmi
import numpy as np
import pytest
import yaml
from lambdaforge.work.cache import WorkCache
from lambdaforge.work.checkpoints import CheckpointCollection

from wisdom.preprocessing.common.StructuralEnrichment import StructuralEnrichment
from wisdom.preprocessing.common.structure.assets.physics.MolecularPreparation import (
    MolecularPreparation,
)
from wisdom.preprocessing.common.structure.assets.physics.PhysicalFields import PhysicalFields
from wisdom.preprocessing.common.structure.assets.physics.PhysicalReferenceUnavailable import (
    PhysicalReferenceUnavailable,
)
from wisdom.preprocessing.common.structure.assets.physics.PhysicsConfig import PhysicsConfig
from wisdom.preprocessing.common.structure.assets.physics.PreparationAdmission import (
    PreparationAdmission,
)
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset


def _chain(remove_beta=True):
    """Return twenty complete synthetic ALA backbones with one optional observed CB absent."""
    names, positions, residues, numbers = [], [], [], []
    for residue in range(20):
        for name, offset, element in (
            ("N", (0, 0, 0), 7), ("CA", (1.2, 1, 0), 6), ("C", (2.5, 0, 0), 6),
            ("O", (2.5, -1.2, 0), 8), ("CB", (1.2, 1, 1.5), 6),
        ):
            if remove_beta and residue == 9 and name == "CB":
                continue
            names.append(name)
            positions.append(np.asarray(offset) + np.array([residue * 3.8, 0, 0]))
            residues.append(residue)
            numbers.append(element)
    arrays = {"atom_names": np.asarray(names), "atom_positions": np.asarray(positions),
        "atomic_numbers": np.asarray(numbers), "residue_indices": np.asarray(residues),
        "chain_indices": np.zeros(len(names), dtype=int)}
    keys = [("A", residue + 1, name) for residue, name in zip(residues, names, strict=True)]
    prepared = arrays["atom_positions"].copy()
    elements = arrays["atomic_numbers"].copy()
    if remove_beta:
        keys.append(("A", 10, "CB"))
        prepared = np.vstack((prepared, [9 * 3.8 + 1.2, 1, 1.5]))
        elements = np.r_[elements, 6]
    lookup = {key: i for i, key in enumerate(keys)}
    bonds = [(lookup["A", i, a], lookup["A", i, b]) for i in range(1, 21)
             for a, b in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"))]
    bonds += [(lookup["A", i, "C"], lookup["A", i + 1, "N"]) for i in range(1, 20)]
    return arrays, keys, prepared, elements, bonds


def _sequence():
    """Return exact deposited-token evidence in the synthetic chain's residue ordinal space."""
    return {"complete_valid": np.ones(20, dtype=bool), "complete_index": np.arange(20),
            "complete_ptr": np.array([0, 20]), "residue_chain": np.zeros(20, dtype=int)}


@pytest.mark.parametrize("policy", ["strict", "bounded"])
@pytest.mark.parametrize("distance", [.34, 3.0])
def test_incompatible_observed_terminal_bond_is_unavailable_not_a_lost_atom(policy, distance):
    """An already observed OXT cannot be normalized away or treated as a small repair."""
    arrays, _, _, _, _ = _chain(remove_beta=False)
    carbon = arrays["atom_positions"][2]
    arrays = {key: np.concatenate((value, [extra])) for key, value, extra in (
        ("atom_positions", arrays["atom_positions"], carbon + np.array([0, 0, distance])),
        ("atom_names", arrays["atom_names"], "OXT"),
        ("atomic_numbers", arrays["atomic_numbers"], 8),
        ("residue_indices", arrays["residue_indices"], 0),
        ("chain_indices", arrays["chain_indices"], 0),
    )}
    before = {key: value.copy() for key, value in arrays.items()}
    admission = PreparationAdmission(PhysicsConfig(heavy_atom_policy=policy))
    with pytest.raises(PhysicalReferenceUnavailable, match="observed C--OXT") as caught:
        admission.validate_observed_termini(arrays)
    assert caught.value.audit["category"] == "incompatible_observed_terminus"
    assert caught.value.audit["terminal_atom"] == ["A", 1, "OXT"]
    assert caught.value.audit["carbon_oxygen_distance_angstrom"] == pytest.approx(distance)
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, before[key])


def test_internal_oxt_requires_chemical_consistency_not_just_a_short_bond():
    """A real terminal segment is allowed, but OXT and a continuing peptide cannot coexist."""
    arrays, _, _, _, _ = _chain(remove_beta=False)
    carbon = arrays["atom_positions"][2]
    arrays = {**arrays,
        "atom_positions": np.vstack((arrays["atom_positions"], carbon + np.array([0, 0, 1.25]))),
        "atom_names": np.concatenate((arrays["atom_names"], ["OXT"])),
        "atomic_numbers": np.r_[arrays["atomic_numbers"], 8],
        "residue_indices": np.r_[arrays["residue_indices"], 0],
        "chain_indices": np.r_[arrays["chain_indices"], 0]}
    admission = PreparationAdmission(PhysicsConfig())
    with pytest.raises(PhysicalReferenceUnavailable, match="continuing peptide") as caught:
        admission.validate_observed_termini(arrays)
    assert caught.value.audit["following_residue"] == ["A", 2]
    assert caught.value.audit["peptide_distance_angstrom"] == pytest.approx(1.3)
    arrays["atom_positions"][5:] += [10, 0, 0]
    arrays["atom_positions"][-1] -= [10, 0, 0]
    admission.validate_observed_termini(arrays)  # Genuinely disconnected following segment.
    arrays["residue_indices"][-1] = 19
    arrays["atom_positions"][-1] = arrays["atom_positions"][97] + [0, 0, 1.25]
    admission.validate_observed_termini(arrays)  # Normal final observed OXT.


def test_oxt_without_its_carbon_is_known_unsupported_chemistry():
    """Missing original C cannot be filled in merely to make an OXT-bearing source acceptable."""
    arrays = {"atom_positions": np.zeros((1, 3)), "atom_names": np.array(["OXT"]),
              "residue_indices": np.zeros(1, int), "chain_indices": np.zeros(1, int)}
    with pytest.raises(PhysicalReferenceUnavailable, match="lacks its carbonyl C"):
        PreparationAdmission(PhysicsConfig()).validate_observed_termini(arrays)


def test_bad_observed_terminus_is_recorded_before_protonation_and_generates_no_targets(tmp_path):
    """A source conflict cannot run PDB2PQR/APBS or block production of a qualified empty bank."""
    arrays = {"atom_positions": np.array([[0., 0, 0], [.34, 0, 0]]),
        "atom_names": np.array(["C", "OXT"]), "atomic_numbers": np.array([6, 8]),
        "residue_names": np.array(["THR", "THR"]),
        "residue_indices": np.zeros(2, int), "chain_indices": np.zeros(2, int)}
    base = tmp_path / "base.npz"
    np.savez(base, **arrays, metadata_json=np.asarray("{}"))
    before = base.read_bytes()
    preparation = MolecularPreparation.__new__(MolecularPreparation)
    preparation.config = PhysicsConfig()
    preparation.provenance = {"unit": True}
    preparation.work = SimpleNamespace(tools=Mock())
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = preparation.config
    producer.reference_parameters = producer.config.parameters()
    producer.reference_parameters.pop("bank_size")
    producer.preparation = preparation
    producer.work = SimpleNamespace(log=Mock(), temp_dir=tmp_path,
        checkpoints=CheckpointCollection(tmp_path / "checkpoints"),
        cache=WorkCache(tmp_path / "cache"),
        map=lambda items, function, **kwargs: [function(item) for item in items])
    paths = producer.generate({"problem": base}, workers=1)["problem"]
    reference, queries = paths[producer.REFERENCE], paths[producer.QUERIES]
    assert producer.validate_reference(reference, base, producer.reference_parameters)
    assert producer.validate_queries(queries, base, producer.config.parameters())
    bank, header = StructuralAsset.read(queries, base, producer.QUERIES)
    assert header["availability"] == "unavailable"
    assert header["preparation_audit"]["category"] == "incompatible_observed_terminus"
    assert bank["targets"].shape == (0, 3)
    assert base.read_bytes() == before
    preparation.work.tools.run.assert_not_called()
    assert producer.report["failed"] == 0 and producer.report["unavailable"] == 1
    reference_bytes, query_bytes = reference.read_bytes(), queries.read_bytes()
    preparation.prepare = Mock(side_effect=AssertionError("valid checkpoint must be restored"))
    assert producer.generate({"problem": base}, workers=1)["problem"] == paths
    assert reference.read_bytes() == reference_bytes and queries.read_bytes() == query_bytes
    preparation.prepare.assert_not_called()


def test_missing_prepared_atoms_without_a_known_source_conflict_still_fail():
    """Unexplained atom losses remain hard failures instead of being hidden by a blanket catch."""
    arrays, keys, prepared, elements, _ = _chain(remove_beta=False)
    with pytest.raises(ValueError, match="removed observed heavy atoms"):
        PreparationAdmission(PhysicsConfig()).assess(arrays, keys[1:], prepared[1:], elements[1:])


def test_small_sidechain_completion_preserves_every_observed_atom():
    """A missing CB is admitted without rewriting geometry, while strict admission rejects it."""
    arrays, keys, prepared, elements, bonds = _chain()
    producer = PreparationAdmission(PhysicsConfig(heavy_atom_policy="bounded"))
    audit = producer.assess(arrays, keys, prepared, elements)
    assert audit["reconstructed_heavy_count"] == 1
    assert audit["reconstructed_fraction"] == pytest.approx(1 / 99)
    assert audit["affected_residue_fraction"] == .05
    assert audit["reconstructed_heavy"] == [("A", 10, "CB")]
    assert audit["heavy_mapping"] == list(range(99))
    assert producer.geometry(prepared, elements, np.full(100, .34), bonds,
                             audit["rebuilt_indices"])["passed"]
    with pytest.raises(PhysicalReferenceUnavailable, match="strict policy"):
        PreparationAdmission(PhysicsConfig()).assess(arrays, keys, prepared, elements)


@pytest.mark.parametrize("option,value", [
    ("repair_maximum_atoms", 0), ("repair_maximum_fraction", .001),
    ("repair_maximum_per_residue", 0), ("repair_maximum_residue_fraction", .01),
])
def test_each_completion_budget_is_independently_enforced(option, value):
    """A generous absolute count cannot override the relative or per-residue safeguards."""
    arrays, keys, prepared, elements, _ = _chain()
    config = replace(PhysicsConfig(heavy_atom_policy="bounded"), **{option: value})
    with pytest.raises(PhysicalReferenceUnavailable, match="count or fraction"):
        PreparationAdmission(config).assess(arrays, keys, prepared, elements)


@pytest.mark.parametrize("evidence", ["confirmed", "missing", "unresolved_tail"])
def test_modeled_oxt_requires_a_deposited_not_merely_observed_chain_end(evidence):
    """A short observed fragment must not silently acquire a new charged C terminus."""
    arrays, keys, prepared, elements, _ = _chain(remove_beta=False)
    keys.append(("A", 20, "OXT"))
    prepared = np.vstack((prepared, [19 * 3.8 + 2.5, -.6, 1.3]))
    elements = np.r_[elements, 8]
    sequence = _sequence() if evidence != "missing" else None
    if evidence == "unresolved_tail":
        sequence["complete_ptr"][-1] = 21
    admission = PreparationAdmission(PhysicsConfig(heavy_atom_policy="bounded"))
    if evidence == "confirmed":
        audit = admission.assess(arrays, keys, prepared, elements, sequence)
        assert audit["verified_modeled_c_termini"] == [["A", 20]]
    else:
        with pytest.raises(PhysicalReferenceUnavailable, match="OXT"):
            admission.assess(arrays, keys, prepared, elements, sequence)


def test_backbone_reconstruction_gap_and_observed_motion_are_not_minor_repairs():
    """Count limits alone cannot admit an incomplete backbone, disconnected peptide or movement."""
    arrays, keys, prepared, elements, _ = _chain()
    admission = PreparationAdmission(PhysicsConfig(heavy_atom_policy="bounded"))
    moved = prepared.copy()
    moved[0, 0] += .1
    with pytest.raises(ValueError, match="coordinates"):
        admission.assess(arrays, keys, moved, elements)
    without_n = {key: value[1:] for key, value in arrays.items()}
    with pytest.raises(PhysicalReferenceUnavailable, match="backbone rebuilding"):
        admission.assess(without_n, keys, prepared, elements)
    arrays["atom_positions"][2] += 10
    prepared[2] += 10
    with pytest.raises(PhysicalReferenceUnavailable, match="gap"):
        admission.assess(arrays, keys, prepared, elements)


def test_count_admission_does_not_override_repaired_geometry():
    """Added heavy atoms need plausible bonds and no severe nonbonded heavy overlap."""
    arrays, keys, prepared, elements, bonds = _chain()
    admission = PreparationAdmission(PhysicsConfig(heavy_atom_policy="bounded"))
    audit = admission.assess(arrays, keys, prepared, elements)
    bad = prepared.copy()
    bad[-1] += 10
    with pytest.raises(PhysicalReferenceUnavailable, match="bond is implausible"):
        admission.geometry(bad, elements, np.full(100, .34), bonds, audit["rebuilt_indices"])
    bad = prepared.copy()
    bad[-2] = bad[-1] + .05
    with pytest.raises(PhysicalReferenceUnavailable, match="steric overlap"):
        admission.geometry(bad, elements, np.full(100, .34), bonds, audit["rebuilt_indices"])


def test_strict_keys_remain_historical_and_only_allocation_guard_can_increase():
    """New repair settings preserve old strict keys and cannot waive scientific identity."""
    original = PhysicsConfig().parameters()
    assert "heavy_atom_policy" not in original
    assert replace(PhysicsConfig(), repair_maximum_atoms=999).parameters() == original
    assert PhysicsConfig(heavy_atom_policy="bounded").parameters() != original
    assert set(StructuralEnrichment.recovery_allocation_guards) == {"physics_maximum_voxels"}


def test_physical_publications_are_distinct_and_training_uses_one_explicit_protocol():
    """Keep strict/bounded publication separate while allowing the reviewed strict VF pilot."""
    root = Path(__file__).resolve().parents[1] / "experiments"
    strict = yaml.safe_load((root / "preprocess/enrich_physics.yaml").read_text())["with"]
    bounded = yaml.safe_load((root / "preprocess/enrich_physics_bounded.yaml").read_text())["with"]
    assert strict["dataset_version"] == "1" and bounded["dataset_version"] == "2"
    assert bounded["dataset"] == {"dataset": "wisdom-dna-reduced-physics@1"}
    assert bounded["physics_options"]["heavy_atom_policy"] == "bounded"
    assert bounded["physics_options"]["repair_maximum_atoms"] == 32
    assert bounded["physics_options"]["repair_maximum_fraction"] == .05
    assert strict["physics_maximum_voxels"] == 8000000
    assert bounded["physics_maximum_voxels"] == 24000000
    assert "physics_maximum_voxels" not in PhysicsConfig(**bounded["physics_options"]).parameters()
    selected = []
    for name in ("wisdom_vf1.yaml", "wisdom_vf_joint.yaml"):
        document = yaml.safe_load((root / "vf" / name).read_text())
        selector = document["with"]["dataset"]
        assert selector in ({"dataset": "wisdom-dna-reduced-physics@1"},
                            {"dataset": "wisdom-dna-reduced-physics@2"})
        selected.append(selector)
    assert selected[0] == selected[1]  # No silent protocol switch between pretraining and joint.


def test_strict_parent_fields_are_readmitted_without_repeating_the_physical_solvers(tmp_path):
    """A typed strict parent supplies unchanged fields, but not reusable unavailable markers."""
    arrays, keys, prepared, elements, _ = _chain(remove_beta=False)
    base = tmp_path / "base.npz"
    np.savez(base, **arrays)
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    for ordinal in range(20):
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(ordinal + 1, " ")
        for index in np.flatnonzero(arrays["residue_indices"] == ordinal):
            atom = gemmi.Atom()
            atom.name = str(arrays["atom_names"][index])
            atom.element = gemmi.Element(int(elements[index]))
            atom.pos = gemmi.Position(*prepared[index])
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    shape = (2, 2, 2)
    targets = np.zeros((*shape, 3), dtype=np.float32)
    targets[..., 0] = 1
    values = {"potential": np.ones(shape, dtype=np.float32), "targets": targets,
        "valid": np.ones((*shape, 3), dtype=bool), "delta": np.ones(3), "origin": np.ones(3),
        "distance": np.ones(shape), "region": np.zeros(shape, dtype=np.int8),
        "accessible": np.ones(shape, dtype=bool), "lj_energy": np.zeros((*shape, 2)),
        "lj_overlap_valid": np.ones((*shape, 2), dtype=bool),
        "heavy_mapping": np.arange(len(prepared), dtype=np.int32), "prepared_positions": prepared,
        "prepared_sigma": np.full(len(prepared), .34),
        "prepared_pdb": np.asarray(structure.make_pdb_string())}
    config = PhysicsConfig()
    settings = config.parameters()
    settings.pop("bank_size")
    audit = {"passed": True, "coverage": 1.0, "valid_queries": 8,
        "region_coverage": {"0": 1.0},
        "comparisons": [{"name": "refined"}, {"name": "larger-box"}]}
    header = {"algorithm": PhysicalFields.ALGORITHM, "units": ["kBT/e", "kJ/mol", "kJ/mol"],
        "potential_convergence": audit, "lj_audits": [{"fixture": True}, {"fixture": True}],
        "provenance": {"fixture": True}, "prepared_identity_keys": keys}
    source = tmp_path / "strict.npz"
    StructuralAsset.write(source, base, values, PhysicalFields.REFERENCE, settings, header)
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = replace(config, heavy_atom_policy="bounded")
    producer.reference_parameters = producer.config.parameters()
    producer.reference_parameters.pop("bank_size")
    producer.preparation = SimpleNamespace(provenance=header["provenance"])
    producer.work = SimpleNamespace(log=Mock())
    target = tmp_path / "bounded.npz"
    assert producer.reuse_strict_reference(base, target, source, None)
    assert producer.validate_reference(target, base, producer.reference_parameters)
    reused, metadata = StructuralAsset.read(target, base, producer.REFERENCE)
    for name, value in values.items():
        np.testing.assert_array_equal(reused[name], value)
    assert metadata["preparation_audit"]["reconstructed_heavy_count"] == 0
    assert metadata["source_reference_sha256"]

    # A cached volume cannot waive the source-chemistry guard just by carrying valid masks.

    conflicting = {key: value.copy() for key, value in arrays.items()}
    conflicting["atom_names"] = conflicting["atom_names"].astype("U3")
    conflicting["atom_names"][1] = "OXT"
    conflicting["atom_positions"][1] = conflicting["atom_positions"][2] + np.array([0, 0, .34])
    invalid_base = tmp_path / "incompatible-base.npz"
    np.savez(invalid_base, **conflicting)
    invalid_reference = tmp_path / "incompatible-reference.npz"
    StructuralAsset.write(invalid_reference, invalid_base, values,
                          PhysicalFields.REFERENCE, settings, header)
    with pytest.raises(PhysicalReferenceUnavailable, match="observed C--OXT"):
        PhysicalFields.validate_reference(invalid_reference, invalid_base, settings)
    # A failed preparation is not suddenly admissible just because a parent contains a file.

    StructuralAsset.write(source, base, {"valid": np.zeros((0, 3), bool)}, producer.REFERENCE,
                          settings, {**header, "availability": "unavailable"})
    assert not producer.reuse_strict_reference(base, target, source, None)


def test_modeled_atom_near_field_is_excluded_without_changing_raw_reference(tmp_path):
    """A bounded query bank keeps only points at least the declared distance from modeled atoms."""
    base = tmp_path / "base.npz"
    np.savez(base, atom_positions=np.zeros((1, 3)))
    producer = PhysicalFields.__new__(PhysicalFields)
    producer.config = PhysicsConfig(heavy_atom_policy="bounded")
    shape = (9, 9, 9)
    raw = {"potential": np.ones(shape, dtype=np.float32), "origin": np.full(3, -4.),
        "delta": np.ones(3), "valid": np.ones((*shape, 3), dtype=bool),
        "targets": np.ones((*shape, 3), dtype=np.float32), "region": np.zeros(shape, np.int8),
        "distance": np.ones(shape), "accessible": np.ones(shape, dtype=bool),
        "prepared_positions": np.zeros((1, 3))}
    reference = tmp_path / "reference.npz"
    StructuralAsset.write(reference, base, raw, producer.REFERENCE, {},
        {"units": ["kBT/e", "kJ/mol", "kJ/mol"],
         "preparation_audit": {"rebuilt_indices": [0]}, "potential_convergence": {"passed": True}})
    before = reference.read_bytes()
    queries = tmp_path / "queries.npz"
    producer.write_queries(base, queries, reference)
    assert producer.validate_queries(queries, base, producer.config.parameters())
    values, header = StructuralAsset.read(queries, base, producer.QUERIES)
    norms = np.linalg.norm(values["positions"], axis=1)
    assert np.all(norms >= 3) and np.any(norms == 3)
    assert reference.read_bytes() == before and values["valid"].all()
    header["repair_exclusion_positions"] = values["positions"][:1].tolist()
    StructuralAsset.write(queries, base, values, producer.QUERIES,
                          producer.config.parameters(), header)
    with pytest.raises(ValueError, match="modeled-atom near field"):
        producer.validate_queries(queries, base)
