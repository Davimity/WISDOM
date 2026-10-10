"""Fixed physicochemical projection through the existing bounded atom neighborhood."""

import numpy as np

from collections.abc import Mapping, Sequence
from wisdom.features.SurfaceFeatureSchema import SurfaceFeatureSchema
from wisdom.utils.structure.AtomicDescriptors import AtomicDescriptors


class SurfaceFeatureProjector:
    """Project protein chemistry, never ligand positions or annotation arrays."""

    def __init__(self, sigma: float = 2.0) -> None:
        """Configure the Gaussian distance kernel.

        Args:
            sigma: Positive kernel width in ångströms.

        Raises:
            ValueError: If sigma is not finite and positive.
        """
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("surface feature sigma must be finite and positive")
        self.sigma = sigma

    def project(self, arrays: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
        """Calculate fixed fields in the exact base surface point order.

        For point p and stored valid atom a, w_pa=exp(-d_pa^2/(2 sigma^2)). Average
        fields use sum(w f)/sum(w); density proxies use sum(w f), without division by
        volume. Only the persisted radius/Jmax neighborhood participates, not all atoms.

        Args:
            arrays: Protein-only structural arrays; atom identities [N], descriptors [N],
                neighbor IDs/distances/mask [M,J]. No GT arrays are consulted.
            names: Ordered known channels, generic and/or zinc-motivated.

        Returns:
            Finite float32 [M,K] matrix, with zero for an empty neighborhood.

        Raises:
            ValueError: If names are unknown or a Zn atom contaminates the protein input.
        """
        names    = SurfaceFeatureSchema.resolve(names)
        elements = arrays["atomic_numbers"]
        if np.any(elements == 30):
            raise ValueError("observed Zn must not enter a protein surface feature projection")
        descriptors = AtomicDescriptors.derive(
            elements, arrays["atom_names"], arrays["residue_names"], arrays["formal_charges"],
            arrays["atom_edge_index"], arrays["atom_edge_bond_order"],
            arrays["atom_edge_is_covalent"],
        )
        atom_names = arrays["atom_names"].astype(str)
        residues   = arrays["residue_names"].astype(str)
        charges    = arrays["formal_charges"].astype(np.float64)
        indices    = arrays["surface_atom_neighbors"].clip(min=0)
        mask       = arrays["surface_atom_mask"]
        distances  = arrays["surface_atom_distances"].astype(np.float64)
        weights    = np.where(mask, np.exp(-distances**2 / (2 * self.sigma**2)), 0.0)
        denominator = weights.sum(axis=1)

        # Lewis eligibility is a conservative protein-identity rule, not a measured donor
        # strength. It excludes amide N, positive formal charges and unknown modifications.

        strict_pairs = {
            ("HIS", "ND1"), ("HIS", "NE2"), ("CYS", "SG"),
            ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
        }
        strict = np.asarray([
            (residue, atom) in strict_pairs
            for residue, atom in zip(residues, atom_names, strict=True)
        ]) & (charges <= 0)
        ched = np.isin(residues, ["CYS", "HIS", "GLU", "ASP"])
        atom_fields = {
            "formal_charge_density": charges,
            "hbond_donor_density": descriptors["atom_hbond_donor"],
            "hbond_acceptor_density": descriptors["atom_hbond_acceptor"],
            "aromatic_density": descriptors["atom_aromaticity"],
            "hydropathy": descriptors["residue_hydropathy"] * (atom_names == "CA"),
            "polarity": descriptors["residue_polarity"] * (atom_names == "CA"),
            "N_density": elements == 7,
            "O_density": elements == 8,
            "S_density": elements == 16,
            "zn_lewis_strict": strict,
            "zn_donor_N_density": strict & (elements == 7),
            "zn_donor_O_density": strict & (elements == 8),
            "zn_donor_S_density": strict & (elements == 16),
            "ched_residue_density": ched & (atom_names == "CA"),
        }
        fields = {name: (weights * field[indices]).sum(axis=1)
                  for name, field in atom_fields.items()}
        for name in ("hydropathy", "polarity"):
            # One CA supplies one residue contribution: large sidechains must not receive
            # more votes simply because they contain more atoms in the bounded neighborhood.

            ca_denominator = (weights * (atom_names == "CA")[indices]).sum(axis=1)
            fields[name] = np.divide(fields[name], ca_denominator,
                                     out=np.zeros_like(denominator), where=ca_denominator > 0)
        donor_density = fields["N_density"] + fields["O_density"] + fields["S_density"]
        fields["zn_specificity_ratio"] = np.divide(
            fields["zn_lewis_strict"], donor_density, out=np.zeros_like(denominator),
            where=donor_density > 0,
        )
        # CA compactness is an interpretable neighborhood hypothesis, not a coordination
        # detector. Only stored protein candidates contribute; no observed metal is consulted.

        if any(name in names for name in ("ched_ca_compactness", "ched_constellation_score")):
            ca = (atom_names == "CA") & ched
            ca_weights = weights * ca[indices]
            totals = ca_weights.sum(axis=1)
            xyz = arrays["atom_positions"].astype(np.float64)[indices]
            center = np.divide((ca_weights[:, :, None] * xyz).sum(axis=1), totals[:, None],
                               out=np.zeros((len(weights), 3)), where=totals[:, None] > 0)
            squared_radius = ((xyz - center[:, None, :])**2).sum(axis=2)
            spread = np.divide((ca_weights * squared_radius).sum(axis=1), totals,
                               out=np.zeros_like(totals), where=totals > 0)
            supported = (ca_weights > 0).sum(axis=1) >= 2
            compactness = np.where(supported, self.sigma / (self.sigma + np.sqrt(spread)), 0)
            diversity = sum(np.any(mask & ca[indices] & (residues[indices] == residue), axis=1)
                            for residue in ("CYS", "HIS", "GLU", "ASP")) / 4
            fields["ched_ca_compactness"] = compactness
            fields["ched_constellation_score"] = compactness * diversity
        return np.column_stack([fields[name] for name in names]).astype(np.float32)
