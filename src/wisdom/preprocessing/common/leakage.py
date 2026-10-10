"""Build full-RAW leakage groups from exact and specialist similarity evidence."""

from typing import Any
from collections import defaultdict
from collections.abc import Mapping, Sequence


def assign_leakage_groups(
    rows          : Sequence[Mapping[str, Any]],
    similarity    : Mapping[str, Any],
    group_same_pdb: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign one transitive leakage component to every RAW protein.

    Args:
        rows: Complete structurally analysed RAW population.
        similarity: MMseqs2 and Foldseek hard edges.
        group_same_pdb: Join members from the same deposited structure.

    Returns:
        Rows carrying ``leakage_group`` plus auditable edge/component evidence.
    """
    identifiers = sorted(str(row["identifier"]) for row in rows)
    exact_pairs = _exact_pairs(rows, group_same_pdb)
    edges = set(similarity["sequence_edges"]) | set(similarity["structure_edges"]) | {
        (pair["left"], pair["right"]) for pair in exact_pairs
    }
    components = _components(identifiers, edges)
    group_by_id = {
        identifier: f"L{number:05d}"
        for number, component in enumerate(components, start=1)
        for identifier in component
    }
    assigned = [{**dict(row), "leakage_group": group_by_id[str(row["identifier"])]} for row in rows]

    return assigned, {
        "components":        components,
        "exact_pairs":       exact_pairs,
        "sequence_edges":    sorted(similarity["sequence_edges"]),
        "largest_component": max(map(len, components)),
        "structure_edges":   sorted(similarity["structure_edges"]),
    }


def _exact_pairs(
    rows          : Sequence[Mapping[str, Any]],
    group_same_pdb: bool,
) -> list[dict[str, Any]]:
    """Create sparse star edges for exact-sequence and provenance equivalence classes.

    Args:
        rows: Complete RAW rows.
        group_same_pdb: Include one equivalence class per PDB deposition.

    Returns:
        Canonical pairs with every reason that connects the pair.
    """
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    fields = {"sequence_sha256": "exact_sequence"}
    if group_same_pdb:
        fields["pdb_id"] = "same_pdb_deposition"
    for field, reason in fields.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(str(row["identifier"]))
        for identifiers in groups.values():
            ordered = sorted(set(identifiers))
            if len(ordered) < 2:
                continue
            # A star has the same connected component as all pair combinations while keeping the
            # evidence table linear for large homologous families.

            for identifier in ordered[1:]:
                reasons[(ordered[0], identifier)].add(reason)
    return [
        {"left": left, "right": right, "reasons": sorted(values)}
        for (left, right), values in sorted(reasons.items())
    ]


def _components(
    identifiers: Sequence[str],
    edges      : set[tuple[str, str]],
) -> list[list[str]]:
    """Compute deterministic connected components with a sparse disjoint-set structure.

    Args:
        identifiers: Every legal RAW identifier.
        edges: Union of all hard similarity and provenance relations.

    Returns:
        Lexically ordered components with lexically ordered members.
    """
    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        """Return and path-compress the component root for ``identifier``."""
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    for left, right in sorted(edges):
        if left not in parent or right not in parent:
            raise ValueError(f"leakage edge names an unknown protein: {left}, {right}")
        first, second = find(left), find(right)
        if first != second:
            parent[max(first, second)] = min(first, second)
    groups: dict[str, list[str]] = defaultdict(list)
    for identifier in identifiers:
        groups[find(identifier)].append(identifier)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])
