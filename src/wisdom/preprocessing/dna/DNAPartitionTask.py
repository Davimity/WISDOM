"""Leakage-safe partitioning and phenotype analysis for a fully annotated DNA dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import gemmi
import numpy as np
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from sklearn import __version__ as sklearn_version
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler


class DNAPartitionTask:
    """Build structural leakage groups, physical phenotypes, splits, and train dilutions.

    Leakage groups answer only whether two records are too similar to cross an evaluation
    boundary. Positive and negative phenotype clusters separately describe physical site shape and
    whole-protein morphology. The three concepts remain separate in every output table.
    """

    SPLITS = ("train", "validation", "test")

    def __init__(
        self,
        sequence_identity           : float = 0.30,
        sequence_coverage           : float = 0.80,
        sequence_evalue             : float = 1e-3,
        structure_probability       : float = 0.50,
        structure_evalue            : float = 1e-3,
        train_fraction              : float = 0.70,
        validation_fraction         : float = 0.15,
        test_fraction               : float = 0.15,
        phenotype_min_cluster_size  : int = 15,
        phenotype_min_samples       : int = 5,
        phenotype_stability_minimum : float = 0.60,
        dilution_sizes              : Sequence[int] = (400, 200, 100, 75, 50, 25),
        seed                        : int = 2026,
        threads                     : int = 36,
        mmseqs_executable           : str = "mmseqs",
        foldseek_executable         : str = "foldseek",
    ) -> None:
        """Configure externally computed leakage and deterministic dataset sampling.

        Args:
            sequence_identity: Minimum MMseqs2 aligned-residue identity as a unit fraction.
            sequence_coverage: Minimum coverage required for both aligned sequences.
            sequence_evalue: Largest retained MMseqs2 expectation value.
            structure_probability: Minimum Foldseek homology probability; ``0.5`` means the tool
                estimates at least 50% probability of homology, such as the same SCOPe superfamily.
            structure_evalue: Largest retained Foldseek expectation value.
            train_fraction: Target fraction of logical proteins assigned to training.
            validation_fraction: Target fraction assigned to model development.
            test_fraction: Target fraction reserved for final evaluation.
            phenotype_min_cluster_size: Smallest HDBSCAN physical phenotype cluster.
            phenotype_min_samples: Neighbor count controlling HDBSCAN conservativeness.
            phenotype_stability_minimum: Minimum median adjusted Rand agreement across a small
                neighboring-parameter grid; lower solutions are reported as noise, not discoveries.
            dilution_sizes: Requested total training-set sizes. Evaluation sets never change.
            seed: Deterministic SHA-256 tie-breaking seed.
            threads: CPU threads passed to MMseqs2 and Foldseek.
            mmseqs_executable: Executable name or path for the required MMseqs2 installation.
            foldseek_executable: Executable name or path for the required Foldseek installation.

        Raises:
            ValueError: If a threshold, fraction, size, or executable name is invalid.
        """
        fractions = (train_fraction, validation_fraction, test_fraction)
        if any(value <= 0.0 for value in fractions) or not math.isclose(sum(fractions), 1.0):
            raise ValueError(
                "train, validation, and test fractions must be positive and sum to one"
            )
        if not 0.0 < sequence_identity <= 1.0 or not 0.0 < sequence_coverage <= 1.0:
            raise ValueError("sequence identity and coverage must lie in (0,1]")
        if not 0.0 <= structure_probability <= 1.0:
            raise ValueError("structure_probability must lie in [0,1]")
        if min(sequence_evalue, structure_evalue) <= 0.0:
            raise ValueError("similarity e-value thresholds must be positive")
        if min(phenotype_min_cluster_size, phenotype_min_samples, threads) < 1:
            raise ValueError("cluster sizes, samples, and threads must be positive")
        if not 0.0 <= phenotype_stability_minimum <= 1.0:
            raise ValueError("phenotype_stability_minimum must lie in [0,1]")
        if any(isinstance(size, bool) or int(size) < 1 for size in dilution_sizes):
            raise ValueError("dilution_sizes must contain positive integers")
        if not mmseqs_executable.strip() or not foldseek_executable.strip():
            raise ValueError("external similarity executable names cannot be empty")

        self.sequence_identity = float(sequence_identity)
        self.sequence_coverage = float(sequence_coverage)
        self.sequence_evalue = float(sequence_evalue)
        self.structure_probability = float(structure_probability)
        self.structure_evalue = float(structure_evalue)
        self.fractions = dict(zip(self.SPLITS, fractions, strict=True))
        self.phenotype_min_cluster_size = int(phenotype_min_cluster_size)
        self.phenotype_min_samples = int(phenotype_min_samples)
        self.phenotype_stability_minimum = float(phenotype_stability_minimum)
        self.dilution_sizes = tuple(sorted({int(value) for value in dilution_sizes}, reverse=True))
        self.seed = int(seed)
        self.threads = int(threads)
        self.mmseqs_executable = mmseqs_executable
        self.foldseek_executable = foldseek_executable

    def run(
        self,
        annotated_root: Path,
        output_root: Path | None = None,
        sequence_pairs: Path | None = None,
        structure_pairs: Path | None = None,
    ) -> dict[str, Any]:
        """Create the final split contract from complete geometry and annotation evidence.

        Args:
            annotated_root: Self-contained root with curated/annotated catalogs, base NPZ files,
                DNA sidecars, and selected structures.
            output_root: Optional destination. ``None`` augments ``annotated_root`` in place and
                avoids copying multi-gigabyte immutable arrays.
            sequence_pairs: Optional precomputed MMseqs2 TSV used by offline tests and audited
                rebuilds; production leaves it unset so the required executable is invoked.
            structure_pairs: Optional precomputed Foldseek TSV with the same restricted purpose.

        Returns:
            JSON-compatible summary containing counts, tool provenance, clustering diagnostics,
            split quality, and dilution membership.

        Raises:
            FileNotFoundError: If required data or specialist executables are unavailable.
            ValueError: If catalogs, arrays, pair tables, or constraints are inconsistent.
            RuntimeError: If leakage-free evaluation partitions cannot be constructed.
        """
        source = annotated_root.resolve()
        target = source if output_root is None else output_root.resolve()
        if target != source:
            raise ValueError(
                "output_root must currently equal annotated_root to avoid array duplication"
            )

        curated = self._read_csv(source / "curated-catalog.csv", "base_identifier")
        annotated = self._read_csv(source / "annotated-catalog.csv", "identifier")
        rows: list[dict[str, Any]] = []
        for identifier in sorted(curated):
            if identifier not in annotated:
                raise ValueError(f"annotation coverage is missing {identifier}")
            row = {**curated[identifier], **annotated[identifier]}
            row["identifier"] = identifier
            row["label"] = int(row["label"])
            row["local_gt_available"] = self._boolean(row["local_gt_available"])
            rows.append(row)
        if set(annotated) != set(curated):
            raise ValueError("annotated and curated catalogs must have identical identifier sets")

        tool_root = target / "clusters"
        tool_root.mkdir(parents=True, exist_ok=True)
        if sequence_pairs is None:
            sequence_path = self._run_mmseqs(rows, tool_root)
        else:
            sequence_path = tool_root / "sequence-pairs.tsv"
            if sequence_pairs.resolve() != sequence_path.resolve():
                shutil.copy2(sequence_pairs, sequence_path)
        if structure_pairs is None:
            structure_path = self._run_foldseek(rows, target, tool_root)
        else:
            structure_path = tool_root / "structure-pairs.tsv"
            if structure_pairs.resolve() != structure_path.resolve():
                shutil.copy2(structure_pairs, structure_path)
        sequence_edges = self._sequence_edges(sequence_path, {row["identifier"] for row in rows})
        structure_edges = self._structure_edges(structure_path, {row["identifier"] for row in rows})
        exact_edges = self._exact_edges(rows)

        components = self._components(
            [row["identifier"] for row in rows], sequence_edges | structure_edges | exact_edges
        )
        for number, component in enumerate(components, start=1):
            group = f"L{number:05d}"
            for identifier in component:
                next(row for row in rows if row["identifier"] == identifier)["leakage_group"] = (
                    group
                )

        positive_features, negative_features = self._descriptors(rows, target)
        positive_clusters = self._phenotypes(positive_features, "P")
        negative_clusters = self._phenotypes(negative_features, "N")
        for row in rows:
            identifier = row["identifier"]
            row["positive_phenotype"] = positive_clusters["labels"].get(identifier, "unavailable")
            row["negative_phenotype"] = negative_clusters["labels"].get(identifier, "unavailable")
            row["phenotype_cluster"] = (
                row["positive_phenotype"] if row["label"] == 1 else row["negative_phenotype"]
            )

        rows, balance = self._balance_population(rows)
        retained = {row["identifier"] for row in rows}
        sequence_edges = {
            edge for edge in sequence_edges if edge[0] in retained and edge[1] in retained
        }
        structure_edges = {
            edge for edge in structure_edges if edge[0] in retained and edge[1] in retained
        }
        exact_edges = {edge for edge in exact_edges if edge[0] in retained and edge[1] in retained}
        positive_features = {
            key: value for key, value in positive_features.items() if key in retained
        }
        negative_features = {
            key: value for key, value in negative_features.items() if key in retained
        }

        assignments, optimization = self._assign_splits(rows)
        for row in rows:
            row["split"] = assignments[row["leakage_group"]]
        self._validate_partitions(rows, sequence_edges, structure_edges, exact_edges)

        dilutions = self._dilutions(rows, target / "dilutions")
        self._write_outputs(
            target,
            rows,
            sequence_edges,
            structure_edges,
            exact_edges,
            positive_features,
            negative_features,
        )

        report = {
            "verdict": "PASS",
            "member_count": len(rows),
            "class_counts": self._class_counts(rows),
            "class_balancing": balance,
            "split_counts": {
                split: self._class_counts([row for row in rows if row["split"] == split])
                for split in self.SPLITS
            },
            "leakage_group_count": len({row["leakage_group"] for row in rows}),
            "largest_leakage_group": max(
                Counter(row["leakage_group"] for row in rows).values()
            ),
            "candidate_leakage_group_count_before_balancing": len(components),
            "candidate_largest_leakage_group_before_balancing": max(map(len, components)),
            "sequence_pair_count": len(sequence_edges),
            "structure_pair_count": len(structure_edges),
            "exact_identity_pair_count": len(exact_edges),
            "positive_phenotypes": positive_clusters["diagnostics"],
            "negative_phenotypes": negative_clusters["diagnostics"],
            "split_optimization": optimization,
            "dilutions": dilutions,
            "parameters": self.parameters(),
            "software": {
                "scikit_learn": sklearn_version,
                "mmseqs2": self._version(self.mmseqs_executable)
                if sequence_pairs is None
                else "precomputed",
                "foldseek": self._version(self.foldseek_executable)
                if structure_pairs is None
                else "precomputed",
            },
        }
        self._atomic_json(target / "partition-report.json", report)
        self._materialize_evidence(target)
        self._write_index(target, rows)
        return report

    def parameters(self) -> dict[str, Any]:
        """Return the complete JSON-compatible scientific partition configuration.

        Returns:
            Thresholds, split targets, HDBSCAN settings, dilution sizes, seed, and tool names.
        """
        return {
            "sequence_identity": self.sequence_identity,
            "sequence_coverage": self.sequence_coverage,
            "sequence_evalue": self.sequence_evalue,
            "structure_probability": self.structure_probability,
            "structure_evalue": self.structure_evalue,
            "split_fractions": self.fractions,
            "phenotype_min_cluster_size": self.phenotype_min_cluster_size,
            "phenotype_min_samples": self.phenotype_min_samples,
            "phenotype_stability_minimum": self.phenotype_stability_minimum,
            "dilution_sizes": list(self.dilution_sizes),
            "seed": self.seed,
            "threads": self.threads,
        }

    def _run_mmseqs(self, rows: list[dict[str, Any]], root: Path) -> Path:
        """Run all-versus-all MMseqs2 search while retaining pairwise threshold evidence.

        Args:
            rows: Curated rows containing canonical amino-acid sequences.
            root: Directory receiving FASTA, pair TSV, and temporary databases.

        Returns:
            Path to ``sequence-pairs.tsv``.
        """
        executable = self._require_tool(self.mmseqs_executable)
        fasta = root / "proteins.fasta"
        output = root / "sequence-pairs.tsv"
        content = "".join(f">{row['identifier']}\n{row['canonical_sequence']}\n" for row in rows)
        self._atomic_text(fasta, content)
        command = [
            executable,
            "easy-search",
            str(fasta),
            str(fasta),
            str(output),
            str(root / "mmseqs-tmp"),
            "--min-seq-id",
            str(self.sequence_identity),
            "-c",
            str(self.sequence_coverage),
            "--cov-mode",
            "0",
            "-e",
            str(self.sequence_evalue),
            "--alignment-mode",
            "3",
            "--threads",
            str(self.threads),
            "--format-output",
            "query,target,fident,qcov,tcov,evalue,bits",
        ]
        self._execute(command, "MMseqs2")
        return output

    def _run_foldseek(self, rows: list[dict[str, Any]], dataset: Path, root: Path) -> Path:
        """Run all-versus-all Foldseek search over exact selected protein structures.

        Args:
            rows: Curated rows with identifier and structure digest.
            dataset: Self-contained dataset root containing ``structures``.
            root: Directory receiving tool inputs, output, and temporary databases.

        Returns:
            Path to ``structure-pairs.tsv``.
        """
        executable = self._require_tool(self.foldseek_executable)
        structures = root / "foldseek-input"
        structures.mkdir(exist_ok=True)
        for row in rows:
            source = dataset / "structures" / f"{row['source_structure_sha256']}.cif"
            target = structures / f"{row['identifier']}.cif"
            if not source.is_file():
                raise FileNotFoundError(f"selected structure is missing: {source}")
            if not target.exists():
                self._selected_chain_structure(source, str(row["protein_chain"]), target)
        output = root / "structure-pairs.tsv"
        command = [
            executable,
            "easy-search",
            str(structures),
            str(structures),
            str(output),
            str(root / "foldseek-tmp"),
            "-e",
            str(self.structure_evalue),
            "--threads",
            str(self.threads),
            "--format-output",
            "query,target,prob,evalue,alntmscore,qcov,tcov",
        ]
        self._execute(command, "Foldseek")
        return output

    @staticmethod
    def _selected_chain_structure(source: Path, chain_id: str, target: Path) -> None:
        """Write one exact selected protein chain as a minimal Foldseek mmCIF input.

        The deposited coordinate file can contain DNA, ligands, and unrelated protein chains.
        Comparing that whole deposition would let another chain create a structural leakage edge
        for the selected example. This method copies only the declared chain from model zero; it
        preserves residue/atom coordinates but does not turn assembly copies or alternate models
        into separate benchmark records.

        Args:
            source: Curation-verified deposited mmCIF file.
            chain_id: Exact case-sensitive author/model chain identifier.
            target: Run-owned minimal mmCIF path consumed by Foldseek.

        Raises:
            ValueError: If the structure has no model or the selected chain is absent.
            OSError: If Gemmi cannot read or atomically write the structure.
        """
        structure = gemmi.read_structure(str(source))
        if len(structure) == 0:
            raise ValueError(f"selected structure has no coordinate model: {source}")
        chain = next((value for value in structure[0] if value.name == chain_id), None)
        if chain is None:
            raise ValueError(f"selected chain {chain_id!r} is absent from {source.name}")

        selected = gemmi.Structure()
        selected.name = target.stem
        model = gemmi.Model(1)
        model.add_chain(chain.clone())
        selected.add_model(model)
        selected.setup_entities()

        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp.cif")
        try:
            selected.make_mmcif_document().write_file(str(temporary))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _sequence_edges(self, path: Path, identifiers: set[str]) -> set[tuple[str, str]]:
        """Parse and threshold MMseqs2 evidence using bilateral alignment coverage.

        Args:
            path: Tab-separated query, target, identity, qcov, tcov, e-value, and bit-score table.
            identifiers: Exact allowed logical member IDs.

        Returns:
            Canonically ordered non-self edges meeting every configured threshold.
        """
        edges: set[tuple[str, str]] = set()
        for fields in self._tsv(path, 7):
            query, target = fields[:2]
            self._known_pair(query, target, identifiers, path)
            identity, qcov, tcov, evalue = map(float, fields[2:6])
            if (
                query != target
                and identity >= self.sequence_identity
                and min(qcov, tcov) >= self.sequence_coverage
                and evalue <= self.sequence_evalue
            ):
                edges.add((min(query, target), max(query, target)))
        return edges

    def _structure_edges(self, path: Path, identifiers: set[str]) -> set[tuple[str, str]]:
        """Parse Foldseek homology evidence without treating a visualization score as identity.

        Args:
            path: Tab-separated query, target, probability, e-value, TM-score, qcov, and tcov table.
            identifiers: Exact allowed logical member IDs.

        Returns:
            Canonically ordered non-self edges meeting probability and e-value thresholds.
        """
        edges: set[tuple[str, str]] = set()
        for fields in self._tsv(path, 7):
            query, target = (Path(fields[0]).stem, Path(fields[1]).stem)
            self._known_pair(query, target, identifiers, path)
            probability = float(fields[2])
            if probability > 1.0:
                probability /= 100.0
            if (
                query != target
                and probability >= self.structure_probability
                and float(fields[3]) <= self.structure_evalue
            ):
                edges.add((min(query, target), max(query, target)))
        return edges

    @staticmethod
    def _exact_edges(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
        """Connect exact sequence, accession, deposition, and coordinate duplicates.

        Args:
            rows: Joined curated/annotated rows.

        Returns:
            Exact-identity edges used in the transitive leakage graph.
        """
        edges: set[tuple[str, str]] = set()
        for field in (
            "sequence_sha256",
            "logical_protein_id",
            "pdb_id",
            "protein_structure_sha256",
        ):
            groups: dict[str, list[str]] = defaultdict(list)
            for row in rows:
                value = str(row.get(field, "")).strip()
                if value:
                    groups[value].append(row["identifier"])
            for identifiers in groups.values():
                for index, left in enumerate(sorted(identifiers)):
                    edges.update((left, right) for right in sorted(identifiers)[index + 1 :])
        return edges

    @staticmethod
    def _components(identifiers: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
        """Compute deterministic connected components of an undirected sparse similarity graph.

        Args:
            identifiers: Every logical protein ID.
            edges: Sequence, structure, or exact-identity pairs.

        Returns:
            Components sorted by their lexicographically first member.
        """
        parent = {identifier: identifier for identifier in identifiers}

        def find(value: str) -> str:
            """Return and path-compress the disjoint-set representative.

            Args:
                value: Logical protein identifier already present in ``parent``.

            Returns:
                Canonical root identifier for the current connected component.
            """
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        for left, right in sorted(edges):
            first, second = find(left), find(right)
            if first != second:
                parent[max(first, second)] = min(first, second)
        groups: dict[str, list[str]] = defaultdict(list)
        for identifier in sorted(identifiers):
            groups[find(identifier)].append(identifier)
        return sorted(groups.values(), key=lambda values: values[0])

    def _descriptors(
        self, rows: list[dict[str, Any]], root: Path
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Measure label-specific physical descriptors from immutable base and sidecar arrays.

        Args:
            rows: Joined catalog records with base and annotation filenames.
            root: Self-contained dataset root.

        Returns:
            Positive local-interface descriptors and negative global-morphology descriptors keyed
            by identifier. Positive rows without usable local GT are absent from the first mapping.
        """
        positive: dict[str, dict[str, float]] = {}
        negative: dict[str, dict[str, float]] = {}
        for row in rows:
            with (
                np.load(root / row["portable_base_path"], allow_pickle=False) as base,
                np.load(root / row["output"], allow_pickle=False) as annotation,
            ):
                positions = base["surface_positions"].astype(np.float64)
                weights = base["surface_area_weights"].astype(np.float64)
                curvature = base["surface_curvatures"].astype(np.float64)
                atoms = base["atom_positions"].astype(np.float64)
                surface_edges = base["surface_edge_index"].astype(np.int64)
                surface_residues, surface_residue_indices = self._surface_residues(base)
                hard = annotation["surface_target_hard"].astype(bool)
                local = bool(annotation["local_gt_available"].item())
                common = self._shape_features(
                    positions, weights, curvature, surface_residues, atoms
                )
                common["protein_length"] = float(int(row.get("observed_residue_count", 0)))
                if row["label"] == 0:
                    negative[row["identifier"]] = common
                elif local and hard.any():
                    site = self._shape_features(
                        positions[hard],
                        weights[hard],
                        curvature[hard],
                        surface_residues[hard],
                        atoms,
                    )
                    region_count, largest_region_fraction = self._positive_regions(
                        hard, surface_edges, weights
                    )
                    site["protein_length"] = common["protein_length"]
                    site["total_surface_point_count"] = float(len(weights))
                    site["positive_surface_weight"] = float(weights[hard].sum())
                    site["positive_surface_fraction"] = float(weights[hard].sum() / weights.sum())
                    site["positive_point_count"] = float(hard.sum())
                    site["positive_residue_count"] = float(
                        len(np.unique(surface_residue_indices[hard]))
                    )
                    site["positive_region_count"] = float(region_count)
                    site["largest_positive_region_fraction"] = largest_region_fraction
                    positive[row["identifier"]] = site
        return positive, negative

    @staticmethod
    def _surface_residues(base: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Map each surface point to the residue of its nearest connected heavy atom.

        The universal archive stores a sparse surface-to-atom graph rather than a dense distance
        matrix. For each surface vertex ``s``, this method selects the incident atom ``a`` with
        minimum Euclidean distance ``d(s,a)`` and transfers that atom's residue category and
        residue index. This makes interface composition genuinely local to the selected surface
        patch instead of measuring whole-protein atom composition.

        Args:
            base: Open NPZ-like mapping containing ``surface_positions [M,3]``,
                ``surface_atom_edge_index [2,E]``, ``surface_atom_distance [E]``,
                ``residue_type_ids [N]``, and ``residue_indices [N]``.

        Returns:
            Surface-aligned residue categories and residue indices, each with shape ``[M]``.

        Raises:
            ValueError: If a surface point has no incident atom edge.
        """
        surface_count = len(base["surface_positions"])
        edges          = base["surface_atom_edge_index"].astype(np.int64)
        distances      = base["surface_atom_distance"].astype(np.float64)
        atom_types     = base["residue_type_ids"].astype(np.int64)
        atom_residues  = base["residue_indices"].astype(np.int64)

        nearest_atoms     = np.full(surface_count, -1, dtype=np.int64)
        nearest_distances = np.full(surface_count, np.inf, dtype=np.float64)
        for edge_index in range(edges.shape[1]):
            surface = int(edges[0, edge_index])
            atom    = int(edges[1, edge_index])
            distance = float(distances[edge_index])
            if distance < nearest_distances[surface]:
                nearest_distances[surface] = distance
                nearest_atoms[surface]     = atom
        if np.any(nearest_atoms < 0):
            raise ValueError("every surface point must have at least one surface-to-atom edge")
        return atom_types[nearest_atoms], atom_residues[nearest_atoms]

    @staticmethod
    def _positive_regions(
        hard: np.ndarray, edges: np.ndarray, weights: np.ndarray
    ) -> tuple[int, float]:
        """Measure connected positive interface regions on the sparse surface graph.

        Two positive points belong to the same region when a path of positive surface edges joins
        them. The largest-region fraction is its represented area divided by total positive area;
        isolated positive points therefore remain explicit one-vertex regions rather than being
        discarded as noise.

        Args:
            hard: Boolean positive mask with shape ``[M]``.
            edges: Undirected surface pairs stored once as ``int [2,E]``.
            weights: Positive dimensionless represented-area weights summing to one, shape ``[M]``.

        Returns:
            Number of connected positive regions and largest positive-region area fraction.
        """
        positive = set(np.flatnonzero(hard).tolist())
        adjacency: dict[int, list[int]] = defaultdict(list)
        for left, right in edges.T:
            first, second = int(left), int(right)
            if first in positive and second in positive:
                adjacency[first].append(second)
                adjacency[second].append(first)

        region_areas: list[float] = []
        remaining = set(positive)
        while remaining:
            start = min(remaining)
            stack = [start]
            remaining.remove(start)
            area = 0.0
            while stack:
                current = stack.pop()
                area += float(weights[current])
                for neighbor in adjacency[current]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            region_areas.append(area)
        total = sum(region_areas)
        largest = max(region_areas, default=0.0) / max(total, np.finfo(float).eps)
        return len(region_areas), float(largest)

    @staticmethod
    def _shape_features(
        positions: np.ndarray,
        weights: np.ndarray,
        curvature: np.ndarray,
        residue_types: np.ndarray,
        atoms: np.ndarray,
    ) -> dict[str, float]:
        """Compute translation-invariant surface size, shape, and curvature summaries.

        Args:
            positions: Surface coordinates ``float [M,3]`` in ångströms.
            weights: Dimensionless represented-area weights ``float [M]`` summing to one.
            curvature: Mean, Gaussian, and curvedness values ``float [M,S,3]``.
            residue_types: Nearest-residue category for every selected surface point, shape ``[M]``.
            atoms: Heavy-atom coordinates ``float [N,3]``.

        Returns:
            Finite scalar descriptors suitable for median/IQR robust scaling.
        """
        normalized = weights / max(float(weights.sum()), np.finfo(float).eps)
        center = np.sum(positions * normalized[:, None], axis=0)
        centered = positions - center
        covariance = (centered * normalized[:, None]).T @ centered
        eigen = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
        mean_curvature = curvature[:, :, 0].mean(axis=1)
        gaussian = curvature[:, :, 1].mean(axis=1)
        values = {
            "surface_point_count": float(len(positions)),
            "atom_count": float(len(atoms)),
            "compactness_radius": float(np.sqrt(np.sum(eigen))),
            "aspect_ratio": float(np.sqrt(eigen[0] / max(eigen[-1], 1e-12))),
            "principal_spread_1": float(np.sqrt(eigen[0])),
            "principal_spread_2": float(np.sqrt(eigen[1])),
            "principal_spread_3": float(np.sqrt(eigen[2])),
            "mean_curvature_mean": float(np.average(mean_curvature, weights=weights)),
            "mean_curvature_std": float(np.std(mean_curvature)),
            "mean_curvature_q25": float(np.quantile(mean_curvature, 0.25)),
            "mean_curvature_median": float(np.median(mean_curvature)),
            "mean_curvature_q75": float(np.quantile(mean_curvature, 0.75)),
            "gaussian_curvature_mean": float(np.average(gaussian, weights=weights)),
            "gaussian_curvature_std": float(np.std(gaussian)),
            "gaussian_curvature_median": float(np.median(gaussian)),
            "concave_fraction": float(np.average(mean_curvature < 0.0, weights=weights)),
            "convex_fraction": float(np.average(mean_curvature > 0.0, weights=weights)),
        }
        for category, residue_ids in {
            "hydrophobic": {1, 5, 8, 10, 11, 13, 14, 15, 18, 19, 20},
            "aromatic": {14, 18, 19},
            "polar": {3, 6, 16, 17},
            "positive": {2, 9, 12},
            "negative": {4, 7},
        }.items():
            values[f"residue_{category}_fraction"] = float(
                np.mean(np.isin(residue_types, list(residue_ids)))
            )
        return values

    def _phenotypes(self, features: dict[str, dict[str, float]], prefix: str) -> dict[str, Any]:
        """Fit robust-scaled HDBSCAN and reject unstable apparent phenotypes.

        Args:
            features: Finite physical descriptors keyed by logical protein.
            prefix: ``P`` for positive local sites or ``N`` for negative global morphology.

        Returns:
            Stable human labels and diagnostics including noise, membership, and parameter-grid ARI.
        """
        identifiers = sorted(features)
        if len(identifiers) < self.phenotype_min_cluster_size * 2:
            return {
                "labels": {identifier: f"{prefix}_NOISE" for identifier in identifiers},
                "diagnostics": {
                    "robust": False,
                    "reason": "too_few_samples",
                    "eligible_count": len(identifiers),
                    "cluster_count": 0,
                    "noise_fraction": 1.0,
                },
            }
        columns = sorted(next(iter(features.values())))
        matrix = np.asarray([[features[key][name] for name in columns] for key in identifiers])
        scaled = RobustScaler(quantile_range=(25.0, 75.0)).fit_transform(matrix)
        settings = sorted(
            {
                (
                    max(2, self.phenotype_min_cluster_size + delta_size),
                    max(1, self.phenotype_min_samples + delta_samples),
                )
                for delta_size in (-5, 0, 5)
                for delta_samples in (-2, 0, 2)
            }
        )
        labels_grid = [
            HDBSCAN(
                min_cluster_size=size,
                min_samples=samples,
                n_jobs=self.threads,
                copy=True,
            ).fit_predict(scaled)
            for size, samples in settings
        ]
        canonical_index = settings.index(
            (self.phenotype_min_cluster_size, self.phenotype_min_samples)
        )
        labels = labels_grid[canonical_index]
        ari = [
            adjusted_rand_score(labels, candidate)
            for candidate in labels_grid
            if not np.array_equal(candidate, labels)
        ]
        cluster_values = sorted(set(labels) - {-1})
        stability = float(np.median(ari)) if ari else 1.0
        robust = len(cluster_values) >= 2 and stability >= self.phenotype_stability_minimum
        if not robust:
            labels = np.full(len(labels), -1)
            cluster_values = []
        remap = {
            value: f"{prefix}{index:03d}" for index, value in enumerate(cluster_values, start=1)
        }
        named = {
            identifier: remap.get(int(label), f"{prefix}_NOISE")
            for identifier, label in zip(identifiers, labels, strict=True)
        }
        return {
            "labels": named,
            "diagnostics": {
                "robust": robust,
                "reason": "stable" if robust else "no_stable_multi_cluster_solution",
                "eligible_count": len(identifiers),
                "feature_names": columns,
                "cluster_count": len(cluster_values),
                "noise_fraction": float(np.mean(labels == -1)),
                "median_adjusted_rand": stability,
                "parameter_grid": [
                    {"min_cluster_size": size, "min_samples": samples} for size, samples in settings
                ],
            },
        }

    def _balance_population(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Select an exactly balanced benchmark while retaining diverse majority examples.

        Leakage groups and physical phenotypes are computed on the complete annotated population
        first. If one class is larger, all members of the minority class are retained and the
        majority is deterministically reduced to the same size. Selection prioritizes stable
        phenotype coverage, distinct leakage groups, distinct public sources, usable positive
        local targets, and finally the configured seeded rank. Dropping a member cannot introduce
        leakage because all retained pair edges keep their original connected-component label.

        Args:
            rows: Complete annotated population with leakage and phenotype assignments.

        Returns:
            Balanced rows and an audit containing original counts and omitted identifiers.

        Raises:
            RuntimeError: If either defensible class is empty.
        """
        by_label = {
            label: [row for row in rows if row["label"] == label] for label in (0, 1)
        }
        target = min(len(by_label[0]), len(by_label[1]))
        if target == 0:
            raise RuntimeError("partitioning requires non-empty positive and negative populations")

        selected: list[dict[str, Any]] = []
        for label in (0, 1):
            candidates = list(by_label[label])
            chosen: list[dict[str, Any]] = []
            phenotypes: set[str] = set()
            groups: set[str] = set()
            sources: set[str] = set()
            while len(chosen) < target:
                row = min(
                    candidates,
                    key=lambda value: (
                        value["phenotype_cluster"] in phenotypes,
                        value["leakage_group"] in groups,
                        str(value.get("source_dataset", "")) in sources,
                        label == 1 and not value["local_gt_available"],
                        self._rank(value["identifier"]),
                    ),
                )
                candidates.remove(row)
                chosen.append(row)
                phenotypes.add(row["phenotype_cluster"])
                groups.add(row["leakage_group"])
                sources.add(str(row.get("source_dataset", "")))
            selected.extend(chosen)

        selected_ids = {row["identifier"] for row in selected}
        selected.sort(key=lambda value: value["identifier"])
        return selected, {
            "method": "deterministic_majority_reduction_after_full_similarity_and_phenotyping",
            "input_counts": self._class_counts(rows),
            "retained_counts": self._class_counts(selected),
            "omitted_count": len(rows) - len(selected),
            "omitted_identifiers": sorted(
                row["identifier"] for row in rows if row["identifier"] not in selected_ids
            ),
        }

    def _assign_splits(self, rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
        """Greedily assign indivisible leakage groups using explicit multi-objective costs.

        Args:
            rows: Fully annotated rows with leakage and phenotype group labels.

        Returns:
            Leakage-group assignments and an auditable objective summary.

        Raises:
            RuntimeError: If local-GT constraints leave validation or test without both classes.
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["leakage_group"]].append(row)
        targets = {
            split: {
                "total": len(rows) * fraction,
                "positive": sum(row["label"] == 1 for row in rows) * fraction,
                "negative": sum(row["label"] == 0 for row in rows) * fraction,
                **{
                    f"phenotype:{phenotype}": sum(
                        row["phenotype_cluster"] == phenotype for row in rows
                    )
                    * fraction
                    for phenotype in sorted({row["phenotype_cluster"] for row in rows})
                },
                **{
                    f"source:{source}": sum(
                        str(row.get("source_dataset", "")) == source for row in rows
                    )
                    * fraction
                    for source in sorted({str(row.get("source_dataset", "")) for row in rows})
                },
            }
            for split, fraction in self.fractions.items()
        }
        counts: dict[str, Counter[str]] = {split: Counter() for split in self.SPLITS}
        assignments: dict[str, str] = {}

        # Seed each sufficiently supported stable phenotype across the three partitions before
        # optimizing sizes. Leakage groups remain indivisible, and groups containing a positive
        # without local GT are excluded because they are legally restricted to training.
        phenotype_groups: dict[str, set[str]] = defaultdict(set)
        train_only_groups = {
            group
            for group, members in groups.items()
            if any(row["label"] == 1 and not row["local_gt_available"] for row in members)
        }
        for group, members in groups.items():
            if group in train_only_groups:
                continue
            for row in members:
                phenotype = row["phenotype_cluster"]
                if not phenotype.endswith(("NOISE", "unavailable")):
                    phenotype_groups[phenotype].add(group)
        for _phenotype, candidates in sorted(
            phenotype_groups.items(), key=lambda value: (len(value[1]), value[0])
        ):
            if len(candidates) < len(self.SPLITS):
                continue
            for split in self.SPLITS:
                if any(assignments.get(group) == split for group in candidates):
                    continue
                available = [group for group in candidates if group not in assignments]
                if not available:
                    continue
                group = min(available, key=lambda value: (len(groups[value]), self._rank(value)))
                assignments[group] = split
                members = groups[group]
                counts[split]["total"] += len(members)
                counts[split]["positive"] += sum(row["label"] == 1 for row in members)
                counts[split]["negative"] += sum(row["label"] == 0 for row in members)
                for row in members:
                    counts[split][f"phenotype:{row['phenotype_cluster']}"] += 1
                    counts[split][f"source:{row.get('source_dataset', '')}"] += 1

        ordered = sorted(
            (group for group in groups if group not in assignments),
            key=lambda group: (-len(groups[group]), self._rank(group)),
        )
        for group in ordered:
            members = groups[group]
            train_only = any(row["label"] == 1 and not row["local_gt_available"] for row in members)
            choices = ("train",) if train_only else self.SPLITS
            scored: list[tuple[float, str]] = []
            for split in choices:
                prospective = counts[split].copy()
                prospective["total"] += len(members)
                prospective["positive"] += sum(row["label"] == 1 for row in members)
                prospective["negative"] += sum(row["label"] == 0 for row in members)
                cost = sum(
                    ((prospective[key] - targets[split][key]) / max(targets[split][key], 1.0)) ** 2
                    for key in ("total", "positive", "negative")
                )
                for row in members:
                    phenotype_key = f"phenotype:{row['phenotype_cluster']}"
                    source_key = f"source:{row.get('source_dataset', '')}"
                    prospective[phenotype_key] += 1
                    prospective[source_key] += 1
                phenotype_keys = [key for key in targets[split] if key.startswith("phenotype:")]
                source_keys = [key for key in targets[split] if key.startswith("source:")]
                cost += 0.25 * sum(
                    (
                        (prospective[key] - targets[split][key])
                        / max(targets[split][key], 1.0)
                    )
                    ** 2
                    for key in phenotype_keys
                )
                cost += 0.10 * sum(
                    (
                        (prospective[key] - targets[split][key])
                        / max(targets[split][key], 1.0)
                    )
                    ** 2
                    for key in source_keys
                )
                # Reward missing stable phenotype coverage without forcing impossible singleton
                # groups to fragment across evaluation boundaries.
                missing = {
                    row["phenotype_cluster"]
                    for row in members
                    if not row["phenotype_cluster"].endswith(("NOISE", "unavailable"))
                }
                represented = {
                    row["phenotype_cluster"]
                    for other, assigned in assignments.items()
                    if assigned == split
                    for row in groups[other]
                }
                cost -= 0.25 * len(missing - represented)
                scored.append((cost, split))
            assignments[group] = min(
                scored, key=lambda value: (value[0], self._rank(group + value[1]))
            )[1]
            split = assignments[group]
            counts[split]["total"] += len(members)
            counts[split]["positive"] += sum(row["label"] == 1 for row in members)
            counts[split]["negative"] += sum(row["label"] == 0 for row in members)
            for row in members:
                counts[split][f"phenotype:{row['phenotype_cluster']}"] += 1
                counts[split][f"source:{row.get('source_dataset', '')}"] += 1
        for split in ("validation", "test"):
            labels = {
                row["label"]
                for group, assigned in assignments.items()
                if assigned == split
                for row in groups[group]
            }
            if labels != {0, 1}:
                raise RuntimeError(
                    f"{split} cannot contain both labels under indivisible "
                    "leakage/local-GT constraints"
                )
        phenotype_feasibility = {
            phenotype: {
                "leakage_group_count": len(candidate_groups),
                "movable_group_count": len(candidate_groups - train_only_groups),
                "representable_in_all_splits": (
                    len(candidate_groups) >= len(self.SPLITS)
                    and len(candidate_groups - train_only_groups) >= len(self.SPLITS)
                ),
                "observed_splits": sorted(
                    {
                        assignments[group]
                        for group in candidate_groups
                        if group in assignments
                    }
                ),
            }
            for phenotype, candidate_groups in sorted(phenotype_groups.items())
        }
        return assignments, {
            "method": "deterministic_greedy_group_assignment",
            "cost_weights": {
                "size_and_class": 1.0,
                "phenotype_distribution": 0.25,
                "source_distribution": 0.10,
                "new_stable_phenotype_reward": 0.25,
            },
            "targets": targets,
            "observed": {split: dict(value) for split, value in counts.items()},
            "phenotype_feasibility": phenotype_feasibility,
        }

    def _dilutions(self, rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
        """Create nested, balanced, leakage-group-aware training subsets of absolute size.

        Args:
            rows: Final partition rows.
            root: Destination ``dilutions`` directory.

        Returns:
            Requested/realized sizes and class/phenotype counts for every canonical subset.
        """
        train = [row for row in rows if row["split"] == "train"]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in train:
            groups[row["leakage_group"]].append(row)

        # Construct one deterministic group order shared by every dilution. Each next group is the
        # one that best preserves the full training class ratio and adds uncovered stable physical
        # phenotypes. Rows are then interleaved across groups, so exact-size prefixes are nested and
        # maximize group diversity; a train-only dilution may contain part of a train leakage group.
        remaining       = set(groups)
        ranked          : list[str] = []
        selected_rows   : list[dict[str, Any]] = []
        seen_phenotypes : set[str] = set()
        target_positive = sum(row["label"] == 1 for row in train) / max(len(train), 1)
        while remaining:
            candidates: list[tuple[float, str]] = []
            for group in remaining:
                prospective = selected_rows + groups[group]
                positive_fraction = sum(row["label"] == 1 for row in prospective) / len(
                    prospective
                )
                stable = {
                    row["phenotype_cluster"]
                    for row in groups[group]
                    if not row["phenotype_cluster"].endswith(("NOISE", "unavailable"))
                }
                score = abs(positive_fraction - target_positive)
                score -= 0.02 * len(stable - seen_phenotypes)
                candidates.append((score, group))
            chosen = min(
                candidates, key=lambda value: (value[0], self._rank(value[1]))
            )[1]
            ranked.append(chosen)
            selected_rows.extend(groups[chosen])
            seen_phenotypes.update(
                row["phenotype_cluster"]
                for row in groups[chosen]
                if not row["phenotype_cluster"].endswith(("NOISE", "unavailable"))
            )
            remaining.remove(chosen)
        rows_by_group = {
            group: sorted(groups[group], key=lambda value: self._rank(value["identifier"]))
            for group in ranked
        }
        ordered_rows = [
            rows_by_group[group][index]
            for index in range(max(map(len, rows_by_group.values()), default=0))
            for group in ranked
            if index < len(rows_by_group[group])
        ]
        result: dict[str, Any] = {}
        previous: set[str] = set()
        canonical = root / "canonical"
        canonical.mkdir(parents=True, exist_ok=True)
        for requested in sorted(self.dilution_sizes):
            target = min(requested, len(ordered_rows))
            selected = list(ordered_rows[:target])
            selected_groups = {row["leakage_group"] for row in selected}
            selected_ids = {row["identifier"] for row in selected}
            if not previous.issubset(selected_ids):
                raise RuntimeError("dilution nesting invariant failed")
            previous = selected_ids
            name = f"train-{requested}"
            self._atomic_text(
                canonical / f"{name}.txt", "".join(f"{value}\n" for value in sorted(selected_ids))
            )
            result[name] = {
                "requested": requested,
                "realized": len(selected),
                "class_counts": self._class_counts(selected),
                "leakage_group_count": len(selected_groups),
                "phenotype_counts": dict(Counter(row["phenotype_cluster"] for row in selected)),
            }
        return result

    def _validate_partitions(
        self, rows: list[dict[str, Any]], *edge_sets: set[tuple[str, str]]
    ) -> None:
        """Enforce every hard partition and local-ground-truth invariant.

        Args:
            rows: Final rows with assigned splits.
            edge_sets: Independent MMseqs2, Foldseek, and exact-identity edge collections.

        Raises:
            RuntimeError: If identifiers, leakage groups, edges, or evaluation local GT cross a
                forbidden boundary.
        """
        by_id = {row["identifier"]: row for row in rows}
        if len(by_id) != len(rows):
            raise RuntimeError("logical identifiers are not unique")
        group_splits: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            group_splits[row["leakage_group"]].add(row["split"])
            if (
                row["split"] in {"validation", "test"}
                and row["label"] == 1
                and not row["local_gt_available"]
            ):
                raise RuntimeError("evaluation positive lacks usable local ground truth")
        if any(len(values) != 1 for values in group_splits.values()):
            raise RuntimeError("one leakage group crosses final partitions")

        # A stable phenotype with three independently movable groups has enough support for one
        # representative per split. Treat omission as a construction failure instead of silently
        # publishing a benchmark whose physical diversity could have been preserved.
        phenotype_groups: dict[str, set[str]] = defaultdict(set)
        phenotype_splits: dict[str, set[str]] = defaultdict(set)
        train_only_groups = {
            row["leakage_group"]
            for row in rows
            if row["label"] == 1 and not row["local_gt_available"]
        }
        for row in rows:
            phenotype = row["phenotype_cluster"]
            if phenotype.endswith(("NOISE", "unavailable")):
                continue
            phenotype_groups[phenotype].add(row["leakage_group"])
            phenotype_splits[phenotype].add(row["split"])
        required_splits = set(self.SPLITS)
        for phenotype, groups in phenotype_groups.items():
            movable = groups - train_only_groups
            if (
                len(groups) >= len(self.SPLITS)
                and len(movable) >= len(self.SPLITS)
                and phenotype_splits[phenotype] != required_splits
            ):
                raise RuntimeError(
                    f"stable phenotype {phenotype} has sufficient group support but is absent "
                    "from one or more splits"
                )
        for edges in edge_sets:
            for left, right in edges:
                if by_id[left]["split"] != by_id[right]["split"]:
                    raise RuntimeError(f"similar pair crosses final partitions: {left}, {right}")

    def _write_outputs(
        self,
        root: Path,
        rows: list[dict[str, Any]],
        sequence_edges: set[tuple[str, str]],
        structure_edges: set[tuple[str, str]],
        exact_edges: set[tuple[str, str]],
        positive: dict[str, dict[str, float]],
        negative: dict[str, dict[str, float]],
    ) -> None:
        """Publish canonical membership, pair, cluster, and descriptor tables.

        Args:
            root: Final self-contained dataset root.
            rows: Final catalog records.
            sequence_edges: Thresholded MMseqs2 pairs.
            structure_edges: Thresholded Foldseek pairs.
            exact_edges: Exact identity/provenance pairs.
            positive: Positive local-site descriptor mappings.
            negative: Negative global-morphology descriptor mappings.
        """
        self._write_csv(root / "catalog.csv", rows)
        for split in self.SPLITS:
            self._atomic_text(
                root / f"{split}.txt",
                "".join(f"{row['identifier']}\n" for row in rows if row["split"] == split),
            )
        clusters = root / "clusters"
        self._write_csv(
            clusters / "sequence-pairs.csv",
            [{"left": left, "right": right} for left, right in sorted(sequence_edges)],
        )
        self._write_csv(
            clusters / "structure-pairs.csv",
            [{"left": left, "right": right} for left, right in sorted(structure_edges)],
        )
        self._write_csv(
            clusters / "exact-identity-pairs.csv",
            [{"left": left, "right": right} for left, right in sorted(exact_edges)],
        )
        self._write_csv(
            clusters / "leakage-groups.csv",
            [
                {
                    "identifier": row["identifier"],
                    "leakage_group": row["leakage_group"],
                    "split": row["split"],
                }
                for row in rows
            ],
        )
        self._write_csv(
            clusters / "positive-phenotypes.csv",
            [
                {
                    "identifier": key,
                    "phenotype": next(
                        row["positive_phenotype"] for row in rows if row["identifier"] == key
                    ),
                    **value,
                }
                for key, value in sorted(positive.items())
            ],
        )
        self._write_csv(
            clusters / "negative-phenotypes.csv",
            [
                {
                    "identifier": key,
                    "phenotype": next(
                        row["negative_phenotype"] for row in rows if row["identifier"] == key
                    ),
                    **value,
                }
                for key, value in sorted(negative.items())
            ],
        )

    @staticmethod
    def _materialize_evidence(root: Path) -> None:
        """Collect dataset-wide audit files into one publishable LambdaForge asset.

        LambdaForge 0.12 streams member assets and has no separate global-assets argument. WISDOM
        therefore copies only its small catalogs, reports, pair evidence, and dilution manifests
        into one ``evidence`` directory. The universal NPZ files, sidecars, and structures remain
        per-protein assets and are never duplicated here.

        Args:
            root: Final run-owned dataset root containing all completed partition outputs.

        Raises:
            OSError: If an evidence file cannot be copied into the compact directory.
        """
        evidence = root / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        for name in (
            "annotation-report.json",
            "annotated-catalog.csv",
            "catalog.csv",
            "curated-catalog.csv",
            "partition-report.json",
            "train.txt",
            "validation.txt",
            "test.txt",
        ):
            source = root / name
            if source.is_file():
                shutil.copy2(source, evidence / name)
        cluster_evidence = evidence / "clusters"
        cluster_evidence.mkdir(exist_ok=True)
        for name in (
            "sequence-pairs.tsv",
            "structure-pairs.tsv",
            "sequence-pairs.csv",
            "structure-pairs.csv",
            "exact-identity-pairs.csv",
            "leakage-groups.csv",
            "positive-phenotypes.csv",
            "negative-phenotypes.csv",
        ):
            source = root / "clusters" / name
            if source.is_file():
                shutil.copy2(source, cluster_evidence / name)

        dilution_source = root / "dilutions"
        dilution_target = evidence / "dilutions"
        if dilution_target.exists():
            shutil.rmtree(dilution_target)
        shutil.copytree(dilution_source, dilution_target)

    @staticmethod
    def _write_index(root: Path, rows: list[dict[str, Any]]) -> None:
        """Write LambdaForge's checksummed logical-member index.

        Args:
            root: Final dataset root holding every member asset.
            rows: Final rows with partitions, labels, and asset provenance.
        """
        dilution_members = {
            path.stem: {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            for path in sorted((root / "dilutions" / "canonical").glob("train-*.txt"))
        }
        members: list[DatasetMember] = []
        for row in sorted(rows, key=lambda value: value["identifier"]):
            base = root / row["portable_base_path"]
            annotation = root / row["output"]
            structure = root / "structures" / f"{row['source_structure_sha256']}.cif"
            assets = {
                "universal_npz": DatasetAsset(
                    path=str(row["portable_base_path"]),
                    sha256=f"sha256:{row['base_npz_sha256']}",
                    size_bytes=base.stat().st_size,
                    media_type="application/x-npz",
                ),
                "dna_annotation": DatasetAsset(
                    path=str(row["output"]),
                    sha256=f"sha256:{row['sidecar_sha256']}",
                    size_bytes=annotation.stat().st_size,
                    media_type="application/x-npz",
                ),
                "source_structure": DatasetAsset(
                    path=structure.relative_to(root).as_posix(),
                    sha256=f"sha256:{row['source_structure_sha256']}",
                    size_bytes=structure.stat().st_size,
                    media_type="chemical/x-mmcif",
                ),
            }
            members.append(
                DatasetMember(
                    member_id=row["identifier"],
                    partitions={
                        "split": row["split"],
                        "leakage_group": row["leakage_group"],
                        "phenotype": row["phenotype_cluster"],
                    },
                    targets={
                        "dna_binding": row["label"],
                        "local_ground_truth": row["local_gt_available"],
                    },
                    metadata={
                        "tier": row.get("tier", "core"),
                        "dilutions": sorted(
                            name
                            for name, identifiers in dilution_members.items()
                            if row["identifier"] in identifiers
                        ),
                    },
                    assets=assets,
                )
            )
        DatasetIndex.write(root / "members.jsonl", members)

    @staticmethod
    def _read_csv(path: Path, key: str) -> dict[str, dict[str, Any]]:
        """Read a unique-key CSV table.

        Args:
            path: Input table.
            key: Required unique identifier column.

        Returns:
            Rows indexed by the stripped key value.
        """
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if key not in (reader.fieldnames or ()):
                raise ValueError(f"{path.name} lacks required column {key}")
            rows = {str(row[key]).strip(): dict(row) for row in reader}
        if not rows or "" in rows:
            raise ValueError(f"{path.name} contains no valid keyed rows")
        return rows

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        """Atomically write the stable union of mapping fields as CSV.

        Args:
            path: Destination table.
            rows: Ordered mappings; nested values are compact JSON.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({str(name) for row in rows for name in row})
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            name: json.dumps(value, sort_keys=True)
                            if isinstance(value, (dict, list, tuple))
                            else value
                            for name, value in row.items()
                        }
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _tsv(path: Path, fields: int) -> list[list[str]]:
        """Read a strict field-count TSV table.

        Args:
            path: Pair-evidence table.
            fields: Exact expected number of columns.

        Returns:
            Non-empty split rows.
        """
        rows = [
            line.rstrip("\n").split("\t")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(len(row) != fields for row in rows):
            raise ValueError(f"{path.name} does not have exactly {fields} columns")
        return rows

    @staticmethod
    def _known_pair(left: str, right: str, identifiers: set[str], path: Path) -> None:
        """Reject pair evidence that refers to a non-member.

        Args:
            left: Query identifier.
            right: Target identifier.
            identifiers: Canonical allowed IDs.
            path: Evidence path used in the diagnostic.
        """
        if left not in identifiers or right not in identifiers:
            raise ValueError(f"{path.name} references an unknown identifier: {left}, {right}")

    @staticmethod
    def _boolean(value: Any) -> bool:
        """Parse a strict Boolean value from CSV-compatible data.

        Args:
            value: Boolean or case-insensitive ``true``/``false`` string.

        Returns:
            Parsed Boolean.
        """
        if isinstance(value, bool):
            return value
        if str(value).lower() in {"true", "1"}:
            return True
        if str(value).lower() in {"false", "0"}:
            return False
        raise ValueError(f"invalid Boolean value: {value!r}")

    @staticmethod
    def _class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        """Count positive and negative rows.

        Args:
            rows: Catalog subset.

        Returns:
            Positive, negative, and total counts.
        """
        return {
            "positive": sum(row["label"] == 1 for row in rows),
            "negative": sum(row["label"] == 0 for row in rows),
            "total": len(rows),
        }

    def _rank(self, value: str) -> str:
        """Return a seed-dependent stable SHA-256 ordering key.

        Args:
            value: Identifier or group/split composite.

        Returns:
            Hexadecimal deterministic rank.
        """
        return hashlib.sha256(f"{self.seed}:{value}".encode()).hexdigest()

    @staticmethod
    def _require_tool(executable: str) -> str:
        """Resolve a required specialist executable without silent fallback.

        Args:
            executable: Name or path selected by configuration.

        Returns:
            Resolved executable path.
        """
        path = shutil.which(executable)
        if path is None:
            raise FileNotFoundError(f"required specialist tool is unavailable: {executable}")
        return path

    @staticmethod
    def _execute(command: list[str], name: str) -> None:
        """Execute one bounded external scientific tool and expose its diagnostic output.

        Args:
            command: Argument vector with no shell interpolation.
            name: Human-readable tool name.

        Raises:
            RuntimeError: If the tool exits unsuccessfully.
        """
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(
                f"{name} failed with exit {result.returncode}: {result.stderr[-2000:]}"
            )

    @staticmethod
    def _version(executable: str) -> str:
        """Capture the first version line of a required specialist tool.

        Args:
            executable: Configured executable name or path.

        Returns:
            Non-empty first output line from ``version``.
        """
        path = DNAPartitionTask._require_tool(executable)
        result = subprocess.run([path, "version"], check=False, text=True, capture_output=True)
        value = (result.stdout or result.stderr).strip().splitlines()
        return value[0] if value else "unknown"

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        """Atomically publish UTF-8 text.

        Args:
            path: Destination path.
            content: Complete text content.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        """Atomically publish standards-compliant, ordered JSON.

        Args:
            path: Destination JSON path.
            payload: JSON-compatible report mapping.
        """
        DNAPartitionTask._atomic_text(
            path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
