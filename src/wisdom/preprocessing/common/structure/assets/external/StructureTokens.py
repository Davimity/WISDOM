"""Official Foldseek 3Di extraction with explicit physical residue correspondence."""

import gemmi
import numpy as np
import lambdaforge as lf

from pathlib import Path
from wisdom.utils.structure.models.Protein import Protein
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.context.SequenceContext import SequenceContext


class StructureTokens:
    """Use official structureto3didescriptor, never approximate 3Di from neighbour IDs."""

    def __init__(self, work: lf.Work, executable: str = "foldseek") -> None:
        """Resolve/version-probe Foldseek before source work or model extraction.

        Args:
            work: Active native LF Work exposing tools, logs and disposable scratch.
            executable: Bare specialist executable, default foldseek.

        Raises:
            RuntimeError: Tool is absent; installation/bootstrap instructions are provided.
        """
        self.work = work
        try:
            self.tool = work.tools.require(executable, version_args=["version"])
        except Exception as error:
            raise RuntimeError("Foldseek is required for 3Di. Run ./install.sh locally or "
                "lf clusters bootstrap <cluster> --project . on managed clusters") from error

    def write(self, base: Path, output: Path, sequence: Path, protein: Protein) -> None:
        """Create chain-isolated official descriptors and bind them to verified CA residues.

        Args:
            base: Immutable schema-3 geometry.
            output: LF-managed temporary asset output.
            sequence: Verified residue/token mapping.
            protein: Source hierarchy already matched to every base atom, not a current PDB fetch.

        Raises:
            ValueError: Official descriptor changes the retained residue order or identity.
        """
        SequenceContext().validate(sequence, base)
        context, _ = StructuralAsset.read(sequence, base, "sequence_context")
        tokens = np.full(len(context["residue_chain"]), "#", dtype="U1")
        valid = np.zeros(len(tokens), np.bool_)
        scratch = self.work.temp_dir / "structure-tokens" / ProteinArchive.sha256(base)
        scratch.mkdir(parents=True, exist_ok=True)
        offset = 0
        for index, chain in enumerate(protein.chains):
            model = gemmi.Model(1)
            selected = gemmi.Chain("A")
            rows = []
            for local, residue in enumerate(chain.residues):
                row = offset + local
                if not context["is_protein_polymer"][row] or not any(
                    atom.name == "CA" for atom in residue.atoms
                ):
                    continue
                entry = gemmi.Residue()
                entry.name = residue.name
                entry.het_flag = "A"
                entry.entity_type = gemmi.EntityType.Polymer
                entry.subchain = "A"
                entry.seqid = gemmi.SeqId(residue.number, residue.insertion_code or " ")
                for atom in residue.atoms:
                    value = gemmi.Atom()
                    value.name = atom.name
                    value.element = gemmi.Element(atom.atomic_number)
                    value.pos = gemmi.Position(*atom.position)
                    entry.add_atom(value)
                selected.add_residue(entry)
                rows.append(row)
            offset += len(chain.residues)
            if not rows:
                continue
            model.add_chain(selected)
            structure = gemmi.Structure()
            structure.name = "wisdom-3di"
            structure.add_model(model)
            structure.setup_entities()
            structure.assign_label_seq_id()
            source = scratch / f"chain-{index}.cif"
            descriptor = scratch / f"chain-{index}.tsv"
            try:
                structure.make_mmcif_document().write_file(str(source))
                self.work.tools.run([self.tool, "structureto3didescriptor", "--threads", "1",
                    "--chain-name-mode", "1", str(source), str(descriptor)], threads=1)
                lines = descriptor.read_text().splitlines()
                if not lines:
                    self.work.log(f"Foldseek emitted no descriptor for selected chain {chain.id}; "
                                  "all its structural tokens remain unavailable")
                    continue
                if len(lines) != 1:
                    raise ValueError("Foldseek did not return one selected chain")
                fields = lines[0].split("\t")
                amino, states = fields[1], fields[2]
                expected = "".join(context["canonical_letter"][rows])
                if amino != expected or len(states) != len(rows):
                    raise ValueError("Foldseek residue/sequence correspondence mismatch")
                tokens[rows] = list(states.lower())
                valid[rows] = [state in "acdefghiklmnpqrstvwy" for state in states.lower()]
            finally:
                source.unlink(missing_ok=True)
                descriptor.unlink(missing_ok=True)
                descriptor.with_name(descriptor.name + ".dbtype").unlink(missing_ok=True)
        StructuralAsset.write(output, base, {"token": tokens, "valid": valid}, "structure_tokens",
            {"tool": self.tool.name, "version": self.tool.version,
             "method": "official-structureto3didescriptor",
             "chain_policy": "verified-observed-CA-chains", "confidence_mask": "not inferred"},
            {"algorithm": "saprot-official-3di-join-v1"})

    @staticmethod
    def validate(path: Path, base: Path) -> bool:
        """Verify token alphabet, residue count and validity semantics.

        Args:
            path: Official 3Di sidecar.
            base: Its immutable physical archive.

        Returns:
            True after schema, binding and token-mask validation.

        Raises:
            ValueError: Unsupported descriptor method, size, alphabet or valid mask.
        """
        values, header = StructuralAsset.read(path, base, "structure_tokens")
        with np.load(base, allow_pickle=False) as stored:
            count = int(stored["residue_indices"].max()) + 1
        if values["token"].shape != (count,) or values["valid"].shape != (count,) or \
                header["parameters"].get("method") != "official-structureto3didescriptor":
            raise ValueError("3Di physical residue or method mismatch")
        if values["token"].dtype != np.dtype("U1") or values["valid"].dtype != np.bool_:
            raise ValueError("3Di token/mask dtype mismatch")
        if not np.array_equal(values["valid"], np.isin(values["token"],
                                                     list("acdefghiklmnpqrstvwy"))):
            raise ValueError("3Di token validity mask mismatch")
        return True
