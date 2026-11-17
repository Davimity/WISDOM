"""Compact deterministic neighbour chemistry derived exclusively from persisted covalent edges."""

import gemmi
import numpy as np

from pathlib import Path
from typing import ClassVar
from collections.abc import Mapping
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset


class CovalentEnvironment:
    """Count chemical neighbourhoods once rather than infer missing chemical bond orders."""

    FAMILY = "covalent_environment"
    parameters: ClassVar[dict[str, float]] = {}

    def build(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Accumulate two directed contributions for each retained undirected covalent pair.

        Args:
            arrays: Atomic numbers [N], covalent flags/bond orders [E], undirected pairs [2,E].

        Returns:
            Integer element counts [N,8] for H/C/N/O/P/S/metals/other and order counts [N,5]
            for unavailable/single/aromatic/double/triple. Unknown orders stay unavailable; these
            counts describe the persisted reconstruction, not experimental certainty of bonds.
        """
        elements = arrays["atomic_numbers"]
        pairs = arrays["atom_edge_index"][:, arrays["atom_edge_is_covalent"].astype(bool)]
        orders = arrays["atom_edge_bond_order"][arrays["atom_edge_is_covalent"].astype(bool)]
        source, target = pairs
        centers = np.concatenate((source, target))
        neighbors = np.concatenate((target, source))
        category = np.full(len(elements), 7, np.int32)
        for index, number in enumerate((1, 6, 7, 8, 15, 16)):
            category[elements == number] = index
        metals = np.asarray([gemmi.Element(int(number)).is_metal for number in elements])
        category[metals] = 6
        elemental = np.zeros((len(elements), 8), np.uint16)
        np.add.at(elemental, (centers, category[neighbors]), 1)
        order_category = np.zeros(len(orders), np.int32)
        for index, order in enumerate((1., 1.5, 2., 3.), start=1):
            order_category[orders == order] = index
        histogram = np.zeros((len(elements), 5), np.uint16)
        np.add.at(histogram, (centers, np.tile(order_category, 2)), 1)
        return {"neighbor_elements": elemental, "bond_orders": histogram}

    def write(self, base: Path, output: Path) -> None:
        """Persist optional compact covalent statistics without duplicating AtomicContext.

        Args:
            base: Immutable universal atomic graph.
            output: Managed temporary destination.
        """
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in ("atomic_numbers", "atom_edge_index",
                "atom_edge_is_covalent", "atom_edge_bond_order")}
        StructuralAsset.write(output, base, self.build(arrays), self.FAMILY, self.parameters,
            {"algorithm": "persisted-covalent-neighbor-histograms-v1",
             "element_columns": ["H", "C", "N", "O", "P", "S", "metal", "other"],
             "order_columns": ["unavailable", "single", "aromatic", "double", "triple"]})

    def validate(self, path: Path, base: Path) -> bool:
        """Reconstruct every neighbour count at immutable publication/resume boundaries.

        Args:
            path: Compact covalent statistics.
            base: Exact universal source archive.

        Returns:
            True if all counts match persisted physical relations.

        Raises:
            ValueError: Wrong binding, counts or array schema.
        """
        actual, header = StructuralAsset.read(path, base, self.FAMILY)
        if header["parameters"] or header.get("algorithm") != (
            "persisted-covalent-neighbor-histograms-v1"
        ):
            raise ValueError("covalent environment algorithm/configuration mismatch")
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in ("atomic_numbers", "atom_edge_index",
                "atom_edge_is_covalent", "atom_edge_bond_order")}
        StructuralAsset.compare(actual, self.build(arrays))
        return True
