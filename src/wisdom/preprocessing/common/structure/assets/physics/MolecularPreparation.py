"""Prepare protein-only physical references with exact retained-heavy-atom correspondence."""

import re
import json
import gemmi
import string
import importlib
import numpy as np
import xml.etree.ElementTree as ET

from typing import Any
from pathlib import Path
from lambdaforge.work.tools import ToolExecutionError
from importlib.metadata import version as package_version
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.physics.PhysicsConfig import PhysicsConfig
from wisdom.preprocessing.common.structure.assets.physics.PreparationAdmission import (
    PreparationAdmission,
)
from wisdom.preprocessing.common.structure.assets.physics.PhysicalReferenceUnavailable import (
    PhysicalReferenceUnavailable,
)


class MolecularPreparation:
    """Own audited physical-source completion without rewriting the model's observed graph."""

    def __init__(self, work: Any, config: PhysicsConfig) -> None:
        """Preflight tools, Python APIs and actual probe parameters before any protein work.

        Args:
            work: Native Work owning external process execution and version provenance.
            config: Explicit physical pilot, including the two force-field probe identities.

        Raises:
            RuntimeError: Missing packages/binaries; install locally with ./install.sh or prepare
                managed targets with lf clusters bootstrap <cluster> --project .
            ValueError: Force field/probes do not support the explicitly implemented LJ rules.
        """
        self.work   = work
        self.config = config
        try:
            self.mm  = importlib.import_module("openmm")
            self.app = importlib.import_module("openmm.app")
            self.pqr_io = importlib.import_module("pdb2pqr.io")
            propka_version = package_version("propka")
            self.pdb2pqr = work.tools.require("pdb2pqr", version_args=("--version",))
            self.apbs = work.tools.require("apbs")
            version = work.tools.run((self.apbs, "--version"), name="vf-apbs-version",
                                     threads=1, check=False)
            # APBS 3.4.1 deliberately exits 13 for --version/--help. Interpret only this documented
            # banner case; arbitrary failures must not be admitted as a working solver.

            match = re.search(r"APBS\s+(\d+\.\d+(?:\.\d+)?)", version.stdout + version.stderr)
            if version.returncode not in {0, 13} or match is None:
                raise RuntimeError("APBS version probe failed without a recognized version banner")
            apbs_version = match.group(1)
        except (ImportError, FileNotFoundError, RuntimeError) as error:
            raise RuntimeError(
                "VF requires PDB2PQR/PROPKA, APBS and OpenMM before extraction. Run ./install.sh "
                "locally or lf clusters bootstrap <cluster> --project . on managed targets. "
                f"Dependency failure: {error}"
            ) from error

        root = Path(str(self.app.__file__)).parent / "data"
        xml = root / config.forcefield
        if not xml.resolve().is_relative_to(root.resolve()):
            raise ValueError("VF forcefield must name an installed OpenMM data XML")
        tree = ET.parse(xml)
        nonbonded = tree.getroot().find("NonbondedForce")
        if nonbonded is None or tree.getroot().find("LennardJonesForce") is not None:
            raise ValueError("VF currently requires an explicit Lorentz--Berthelot NonbondedForce")
        parameters = {a.attrib["type"]: a.attrib for a in nonbonded.findall("Atom")}
        self.probes: list[dict[str, Any]] = []
        for key in (config.probe_small, config.probe_large):
            residue, atom = key.split(":")
            entry = tree.getroot().find(f"Residues/Residue[@name='{residue}']/Atom[@name='{atom}']")
            if entry is None:
                raise ValueError(f"unknown real force-field probe identity: {key}")
            atom_type = entry.attrib["type"]
            values = parameters[atom_type]
            self.probes.append({"identity": key, "type": atom_type, "charge_e": 0.0,
                                "sigma_nm": float(values["sigma"]),
                                "epsilon_kj_mol": float(values["epsilon"])})
        if not 0 < self.probes[0]["sigma_nm"] < self.probes[1]["sigma_nm"]:
            raise ValueError("VF's small and large probes require ordered positive actual sigma")
        self.provenance = {
            "openmm": self.mm.__version__, "forcefield": config.forcefield,
            "forcefield_sha256": ProteinArchive.sha256(xml), "probes": self.probes,
            "pdb2pqr": self.pdb2pqr.version, "apbs": apbs_version,
            "propka": propka_version,
            "electrostatic_forcefield": "AMBER", "combination": "Lorentz--Berthelot",
            "protein_probe_exclusions": "none; unbonded test particle",
            "openmm_platform": "Reference", "coordinate_frame": "immutable-parent-NPZ-centered",
        }

    def prepare(self, base: Path, scratch: Path, threads: int = 1,
                sequence_context: Path | None = None) -> dict[str, Any]:
        """Prepare protein-only charges and admit completion under the selected chemical policy.

        Args:
            base: Immutable schema-3 NPZ with retained heavy atoms only.
            scratch: Disposable native Work directory for Gemmi PDB, PQR and tool outputs.
            threads: Bounded external-process threads; execution only, excluded from identity.
            sequence_context: Optional exact base-bound sequence evidence; modeled terminal OXT
                requires a verified deposited chain end under the bounded repair protocol.

        Returns:
            Prepared coordinates [N_prepared,3] in Å, sigma [N_prepared] nm, epsilon
            [N_prepared] kJ/mol, heavy-to-prepared indices, PQR charges/radii and audit metadata.
            New H or admitted modeled heavy atoms never become graph nodes or input descriptors.

        Raises:
            ValueError: Lost/duplicated/moved heavy atoms, unknown templates, PDB
                address limits, mismatching PQR/PDB rows or unsupported force-field operators.
            PhysicalReferenceUnavailable: Observed terminal chemistry is contradictory or repairs
                exceed the selected chemical admission policy; no physical target is invented.
        """
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in (
                "atom_positions", "atomic_numbers", "atom_names", "residue_names",
                "residue_indices", "chain_indices")}
            metadata = json.loads(str(stored["metadata_json"].item()))
        if np.any(arrays["atomic_numbers"] <= 1):
            raise ValueError("VF currently requires the existing heavy-atom universal graph")

        structure = gemmi.Structure()
        model = gemmi.Model(1)
        alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
        if int(arrays["chain_indices"].max()) >= len(alphabet) or len(arrays["atom_names"]) > 99999:
            raise ValueError("VF preparation exceeds explicit single-character PDB address limits")

        # Reject incompatible observed termini before PDB2PQR can split/relabel the chain. This
        # preserves the immutable parent and records known unsupported chemistry, not lost atoms.

        admission = PreparationAdmission(self.config)
        admission.validate_observed_termini(arrays)

        keys: list[tuple[str, int, str]] = [("", 0, "")] * len(arrays["atom_names"])
        for chain_id in np.unique(arrays["chain_indices"]):
            chain = gemmi.Chain(alphabet[int(chain_id)])
            selected = np.flatnonzero(arrays["chain_indices"] == chain_id)
            for residue_id in np.unique(arrays["residue_indices"][selected]):
                indices = selected[arrays["residue_indices"][selected] == residue_id]
                residue = gemmi.Residue()
                residue.name = str(arrays["residue_names"][indices[0]])
                residue.seqid = gemmi.SeqId(int(residue_id) + 1, " ")
                if int(residue_id) >= 9999:
                    raise ValueError("VF residue address exceeds the PDB preparation contract")
                residue.het_flag = "A"
                for index in indices:
                    atom = gemmi.Atom()
                    atom.name = str(arrays["atom_names"][index])
                    atom.element = gemmi.Element(int(arrays["atomic_numbers"][index]))
                    atom.pos = gemmi.Position(*map(float, arrays["atom_positions"][index]))
                    atom.occ = 1.0
                    residue.add_atom(atom)
                    keys[index] = (chain.name, int(residue_id) + 1, atom.name)
                chain.add_residue(residue)
            model.add_chain(chain)
        structure.add_model(model)
        source = scratch / "protein.pdb"
        write_options = gemmi.PdbWriteOptions()
        write_options.cryst1_record = False
        structure.write_pdb(str(source), write_options)
        pqr = scratch / "prepared.pqr"
        pdb = scratch / "prepared.pdb"
        result = self.work.tools.run((self.pdb2pqr, "--ff=AMBER", "--ffout=AMBER", "--keep-chain",
            "--nodebump", "--noopt", "--titration-state-method=propka",
            f"--with-ph={self.config.ph}", f"--pdb-output={pdb}", source, pqr),
            cwd=scratch, threads=threads, name="vf-protonation", check=False)
        # Known missing-atom/gap limitations do not justify inventing a physical reference.
        # Unknown exit errors (including resource/environment failures) remain retryable failures.

        if result.returncode:
            diagnostic = result.stdout + result.stderr
            limitations = ("missing too many", "Found gap in biomolecule structure",
                           "Too few atoms present to reconstruct or cap residue")
            reason = next((line.strip() for line in diagnostic.splitlines()
                           if any(text in line for text in limitations)), None)
            if result.returncode == 1 and reason is not None:
                raise PhysicalReferenceUnavailable(
                    f"PDB2PQR cannot prepare observed chemistry: {reason}",
                    {"category": "unsupported_pdb2pqr_chemistry", "diagnostic": reason})
            raise ToolExecutionError(result)
        prepared = gemmi.read_structure(str(pdb))[0]
        flat = [(chain, residue, atom)
                for chain in prepared for residue in chain for atom in residue]
        prepared_keys = [(c.name, int(r.seqid.num or 0), a.name) for c, r, a in flat]
        positions = np.asarray([[a.pos.x, a.pos.y, a.pos.z] for _, _, a in flat])
        elements = np.asarray([a.element.atomic_number for _, _, a in flat], dtype=np.int16)
        sequence = StructuralAsset.read(sequence_context, base, "sequence_context")[0] if (
            sequence_context is not None) else None
        preparation_audit = admission.assess(arrays, prepared_keys, positions, elements, sequence)
        mapping = np.asarray(preparation_audit["heavy_mapping"], dtype=np.int32)
        with pqr.open() as stream:
            pqr_atoms = self.pqr_io.read_pqr(stream)
        pqr_lookup = {(a.chain_id, a.res_seq, a.name): a for a in pqr_atoms}
        if set(pqr_lookup) != set(prepared_keys) or len(pqr_atoms) != len(flat):
            raise ValueError("VF PQR and prepared-PDB physical atom domains differ")

        # Gemmi owns coordinate parsing; OpenMM receives an explicit topology and performs its
        # own residue/terminal/disulfide force-field matching, not heuristic NPZ charges/radii.

        topology = self.app.Topology()
        canonical_names = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP",
                           "GLH": "GLU", "LYN": "LYS", "CYX": "CYS", "CYM": "CYS"}
        for chain in prepared:
            target_chain = topology.addChain(chain.name)
            for residue in chain:
                residue_name = canonical_names.get(residue.name, residue.name)
                target_residue = topology.addResidue(residue_name, target_chain,
                                                     id=str(residue.seqid.num))
                for atom in residue:
                    element = self.app.element.Element.getByAtomicNumber(atom.element.atomic_number)
                    # OpenMM's public standard-bond dictionary names the first terminal N-H
                    # atom H, while AMBER-output PDB2PQR calls it H1. Rename topology-only H1;
                    # atom ordering, PQR charges and all retained-heavy identities are unchanged.

                    name = "H" if atom.name == "H1" else atom.name
                    topology.addAtom(name, element, target_residue)
        topology.createStandardBonds()
        topology.createDisulfideBonds(positions * 0.1 * self.mm.unit.nanometer)
        forcefield = self.app.ForceField(self.config.forcefield)
        system = forcefield.createSystem(topology, nonbondedMethod=self.app.NoCutoff,
                                        rigidWater=False)
        forces = [force for force in system.getForces()
                  if isinstance(force, self.mm.NonbondedForce)]
        if len(forces) != 1:
            raise ValueError("VF requires exactly one standard OpenMM nonbonded parameter source")
        sigma, epsilon = [], []
        for index in range(len(flat)):
            _, diameter, depth = forces[0].getParticleParameters(index)
            sigma.append(diameter.value_in_unit(self.mm.unit.nanometer))
            epsilon.append(depth.value_in_unit(self.mm.unit.kilojoule_per_mole))
        try:
            preparation_audit["geometry"] = admission.geometry(
                positions, elements, np.asarray(sigma),
                [(a.index, b.index) for a, b in topology.bonds()],
                preparation_audit["rebuilt_indices"])
        except PhysicalReferenceUnavailable as error:
            error.audit.update(preparation_audit)
            raise
        charges = np.asarray([pqr_lookup[key].charge for key in prepared_keys], dtype=np.float64)
        return {"positions": positions, "sigma": np.asarray(sigma), "epsilon": np.asarray(epsilon),
                "heavy_mapping": mapping, "partial_charges": charges,
                "pqr_radii": np.asarray([pqr_lookup[key].radius for key in prepared_keys]),
                "prepared_keys": prepared_keys, "original_heavy_keys": keys,
                "hydrogen_count": int((elements == 1).sum()),
                "net_charge_e": float(charges.sum()), "preparation_audit": preparation_audit,
                "elements": elements,
                "bonds": np.asarray([(a.index, b.index) for a, b in topology.bonds()],
                                    dtype=np.int32).reshape(-1, 2),
                "coordinate_origin": metadata.get("coordinate_origin"),
                "provenance": self.provenance}
