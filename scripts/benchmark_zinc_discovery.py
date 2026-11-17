"""Compare exact per-copy Discovery summaries against the repeated detailed reference path."""

import argparse
import json
import platform
import statistics
import time

import gemmi

from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination
from wisdom.utils.structure.BiologicalAssembly import BiologicalAssembly


def synthetic_assembly(copies: int, residues: int) -> BiologicalAssembly:
    """Build separated synthetic peptide copies with nearby and uncoordinated Zn atoms.

    Args:
        copies: Positive number of physical copies sharing the deposited chain name AQ.
        residues: Positive number of two-donor residues per copy.

    Returns:
        Gemmi-owned test assembly; it is not biological benchmark evidence.
    """
    source = gemmi.Structure()
    model = gemmi.Model(1)
    for copy in range(copies):
        chain = gemmi.Chain("AQ")
        for index in range(residues):
            residue = gemmi.Residue()
            residue.name, residue.seqid = "HIS", gemmi.SeqId(index + 1, " ")
            residue.entity_type, residue.het_flag = gemmi.EntityType.Polymer, "A"
            for name, offset in (("ND1", -1.0), ("NE2", 1.0)):
                atom = gemmi.Atom()
                atom.name, atom.element, atom.occ = name, gemmi.Element("N"), 1.0
                atom.pos = gemmi.Position(copy * 40.0 + offset, index * 2.0, 0)
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    metals = gemmi.Chain("Z")
    for copy in range(copies):
        for offset in (0.0, 20.0):
            residue = gemmi.Residue()
            residue.name, residue.het_flag = "ZN", "H"
            residue.seqid = gemmi.SeqId(len(metals) + 1, " ")
            atom = gemmi.Atom()
            atom.name, atom.element, atom.occ = "ZN", gemmi.Element("Zn"), 1.0
            atom.pos = gemmi.Position(copy * 40.0 + offset, 1, 0)
            residue.add_atom(atom)
            metals.add_residue(residue)
    model.add_chain(metals)
    source.add_model(model)
    return BiologicalAssembly(source, source[0], "1")


def repeated_summaries(assembly: BiologicalAssembly) -> list[dict[str, object]]:
    """Reproduce Discovery's previous full-scan-per-copy computation without dropping fields.

    Args:
        assembly: Synthetic assembly in stable Gemmi chain/copy order.

    Returns:
        The exact four fields consumed by the optimized candidate selection path.
    """
    verifier = ZincCoordination()
    result = []
    for copy, _ in enumerate(assembly.protein_chains(), 1):
        sites = verifier.analyse(assembly, "AQ", copy)
        accepted = [site for site in sites if site["accepted"]]
        result.append({"protein_chain": "AQ", "protein_copy": copy,
                       "accepted_sites": len(accepted),
                       "donor_count": sum(site["selected_donor_count"] for site in accepted)})
    return result


def main() -> None:
    """Print raw CPU timings and exact-equivalence checks; never downloads or submits jobs.

    CLI arguments select copy counts, residue counts and timing repeats. Results establish the
    local algorithmic improvement only, not whole-population network or preprocessing throughput.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--residues", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    records = []
    for copies in args.copies:
        assembly = synthetic_assembly(copies, args.residues)
        expected = repeated_summaries(assembly)
        assert ZincCoordination().summarize_assembly(assembly) == expected
        timings = {}
        for name, function in (("repeated", repeated_summaries),
                               ("single_index", ZincCoordination().summarize_assembly)):
            seconds = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                assert function(assembly) == expected
                seconds.append(time.perf_counter() - started)
            timings[name] = seconds
        records.append({"copies": copies, "residues_per_copy": args.residues,
                        "exact_summary_equality": True, "seconds": timings,
                        "median_speedup": statistics.median(timings["repeated"]) /
                                          statistics.median(timings["single_index"])})
    print(json.dumps({"python": platform.python_version(), "processor": platform.processor(),
                      "scope": "synthetic CPU assembly inspection, excluding network and parsing",
                      "records": records}, indent=2))


if __name__ == "__main__":
    main()
