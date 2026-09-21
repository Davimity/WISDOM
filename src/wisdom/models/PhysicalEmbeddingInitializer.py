"""Task-independent physical initialization for trainable element and residue embeddings."""

import gemmi
import torch

from torch import Tensor, nn
from typing import ClassVar


class PhysicalEmbeddingInitializer:
    """Project standardized chemical descriptor tables into trainable embedding spaces."""

    _RESIDUES: ClassVar[tuple[str, ...]] = (
        "UNK", "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    )

    # Kyte-Doolittle hydropathy, charge class, polarity, aromaticity, side-chain volume (A^3),
    # hydrogen-bond donor tendency, and acceptor tendency. The unknown row is imputed below.
    _RESIDUE_DESCRIPTORS: ClassVar[dict[str, tuple[float, ...]]] = {
        "ALA": (1.8, 0.0, 0.0, 0.0, 88.6, 0.0, 0.0),
        "ARG": (-4.5, 1.0, 1.0, 0.0, 173.4, 1.0, 0.0),
        "ASN": (-3.5, 0.0, 1.0, 0.0, 114.1, 1.0, 1.0),
        "ASP": (-3.5, -1.0, 1.0, 0.0, 111.1, 0.0, 1.0),
        "CYS": (2.5, 0.0, 0.5, 0.0, 108.5, 1.0, 1.0),
        "GLN": (-3.5, 0.0, 1.0, 0.0, 143.8, 1.0, 1.0),
        "GLU": (-3.5, -1.0, 1.0, 0.0, 138.4, 0.0, 1.0),
        "GLY": (-0.4, 0.0, 0.0, 0.0, 60.1, 0.0, 0.0),
        "HIS": (-3.2, 0.5, 1.0, 1.0, 153.2, 1.0, 1.0),
        "ILE": (4.5, 0.0, 0.0, 0.0, 166.7, 0.0, 0.0),
        "LEU": (3.8, 0.0, 0.0, 0.0, 166.7, 0.0, 0.0),
        "LYS": (-3.9, 1.0, 1.0, 0.0, 168.6, 1.0, 0.0),
        "MET": (1.9, 0.0, 0.0, 0.0, 162.9, 0.0, 1.0),
        "PHE": (2.8, 0.0, 0.0, 1.0, 189.9, 0.0, 0.0),
        "PRO": (-1.6, 0.0, 0.0, 0.0, 112.7, 0.0, 0.0),
        "SER": (-0.8, 0.0, 1.0, 0.0, 89.0, 1.0, 1.0),
        "THR": (-0.7, 0.0, 1.0, 0.0, 116.1, 1.0, 1.0),
        "TRP": (-0.9, 0.0, 0.5, 1.0, 227.8, 1.0, 0.0),
        "TYR": (-1.3, 0.0, 1.0, 1.0, 193.6, 1.0, 1.0),
        "VAL": (4.2, 0.0, 0.0, 0.0, 140.0, 0.0, 0.0),
    }

    def initialize(
        self,
        element_embedding: nn.Embedding,
        residue_embedding: nn.Embedding,
    ) -> None:
        """Initialize two lookup tables from deterministic task-independent descriptors.

        Each descriptor column is mean-imputed and standardized. Singular-value decomposition
        then gives an orthogonal projection ordered by descriptor variance. If the requested
        embedding is wider than the physical descriptor rank, remaining coordinates start at zero
        and remain fully trainable. Unknown rows equal the descriptor mean, hence map to zero.

        Args:
            element_embedding: Element table indexed by atomic number, normally ``[119,D_Z]``.
            residue_embedding: Residue table in WISDOM's ``UNK, ALA, ..., VAL`` order.

        Raises:
            ValueError: If either vocabulary is too small for its public WISDOM index contract.
        """
        if element_embedding.num_embeddings < 119:
            raise ValueError(
                "physical element initialization requires atomic numbers 0 through 118"
            )
        if residue_embedding.num_embeddings < len(self._RESIDUES):
            raise ValueError("physical residue initialization requires the 21-entry WISDOM table")

        elements = self._element_descriptors(element_embedding.num_embeddings)
        residues = self._residue_descriptors(residue_embedding.num_embeddings)

        with torch.no_grad():
            element_embedding.weight.copy_(
                self._project(elements, element_embedding.embedding_dim)
            )
            residue_embedding.weight.copy_(
                self._project(residues, residue_embedding.embedding_dim)
            )

    @staticmethod
    def _element_descriptors(count: int) -> Tensor:
        """Construct general element descriptors from Gemmi's periodic-table data.

        Args:
            count: Lookup-table row count; rows above atomic number 118 are unknown padding.

        Returns:
            Float matrix ``[count,5]`` containing atomic number, square-root atomic number, atomic
            mass, covalent radius, and van der Waals radius. Row zero and padding rows are imputed
            with column means and therefore become neutral after standardization.
        """
        values = torch.full((count, 5), float("nan"), dtype=torch.float32)
        for atomic_number in range(1, min(count, 119)):
            element = gemmi.Element(atomic_number)
            values[atomic_number] = torch.tensor(
                (
                    float(atomic_number),
                    float(atomic_number) ** 0.5,
                    float(element.weight),
                    float(element.covalent_r),
                    float(element.vdw_r),
                )
            )
        return values

    def _residue_descriptors(self, count: int) -> Tensor:
        """Construct the fixed general amino-acid descriptor matrix.

        Args:
            count: Residue lookup-table row count.

        Returns:
            Float matrix ``[count,7]``. Unknown and extra padding rows are missing values to be
            mean-imputed by the common projection routine.
        """
        values = torch.full((count, 7), float("nan"), dtype=torch.float32)
        for index, residue in enumerate(self._RESIDUES[1:], start=1):
            values[index] = torch.tensor(self._RESIDUE_DESCRIPTORS[residue])
        return values

    @staticmethod
    def _project(descriptors: Tensor, width: int) -> Tensor:
        """Mean-impute, standardize, and orthogonally project one descriptor table.

        Args:
            descriptors: Physical table ``[categories,properties]`` with optional NaNs.
            width: Positive target embedding width.

        Returns:
            Deterministic float matrix ``[categories,width]`` with per-coordinate scale bounded by
            ``1/sqrt(width)`` so physical initialization remains comparable to fan-scaled random
            lookup tables.
        """
        observed = torch.isfinite(descriptors)
        means    = torch.nanmean(descriptors, dim=0)
        imputed  = torch.where(observed, descriptors, means)
        centered = imputed - imputed.mean(dim=0, keepdim=True)
        scales   = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1.0e-6)
        standardized = centered / scales

        _, _, right = torch.linalg.svd(standardized, full_matrices=False)
        rank        = min(width, right.shape[0])
        projected   = standardized @ right[:rank].T
        output      = descriptors.new_zeros((len(descriptors), width))
        output[:, :rank] = projected

        maximum = output.abs().amax().clamp_min(1.0)
        return output / maximum / (width**0.5)
