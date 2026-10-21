"""Closed, single-factor atomic-encoder candidates for the V9 construction studies."""

from enum import Enum


class AtomicEncoderType(str, Enum):
    """Identify an exact operator/control, rather than an unspecified GNN family."""

    BASELINE = "baseline"
    BASELINE_NO_ATTRIBUTES = "baseline_no_attributes"
    BASELINE_RADIAL = "baseline_radial"
    GCN_STRUCTURAL = "gcn_structural"
    GCN_CONDITIONED = "gcn_conditioned"
    GCN_CANONICAL = "gcn_canonical"
    SAGE_STRUCTURAL = "sage_structural"
    SAGE_CONDITIONED = "sage_conditioned"
    GIN = "gin"
    RGCN = "rgcn"
    RGCN_CANONICAL = "rgcn_canonical"
    MPNN_MEAN = "mpnn_mean"
    MPNN_SUM = "mpnn_sum"
    GINE = "gine"
    GINE_NO_ATTRIBUTES = "gine_no_attributes"
    NNCONV_BASES = "nnconv_bases"
    NNCONV_FULL = "nnconv_full"
    PNA_LIGHT = "pna_light"
    PNA_IDENTITY = "pna_identity"
    PNA_FULL = "pna_full"
    GATV2_2 = "gatv2_2"
    GATV2_4 = "gatv2_4"
    GATV2_NO_ATTRIBUTES = "gatv2_no_attributes"
    SCHNET = "schnet"
    SCHNET_RADIAL = "schnet_radial"
    DIMENET = "dimenet"
    DIMENET_PLUS_PLUS = "dimenet_plus_plus"
    GVP = "gvp"
    PAINN = "painn"
    EGNN_FIXED = "egnn_fixed"
    EGNN_LATENT = "egnn_latent"
    TRANSFORMER_SPARSE = "transformer_sparse"
    TRANSFORMER_NO_ATTRIBUTES = "transformer_no_attributes"
    STRUCTURAL_GLOBAL = "structural_global"
    GLOBAL_NO_STRUCTURE = "global_no_structure"

    def inactive_sources(self) -> tuple[str, ...]:
        """Return edge sources unavailable to this candidate, never optional missing data.

        Returns:
            Existing registry names whose samples and L0 costs must be exactly zero.
            Directional/vector geometry is a declared additional input, not an edge scalar.
        """
        spatial = ("distance", "same_residue", "same_chain", "residue_separation")
        covalent = ("distance", "bond_order", "same_residue")
        structural = self in {
            self.BASELINE_NO_ATTRIBUTES,
            self.GCN_STRUCTURAL,
            self.GCN_CANONICAL,
            self.SAGE_STRUCTURAL,
            self.GIN,
            self.RGCN,
            self.RGCN_CANONICAL,
            self.GINE_NO_ATTRIBUTES,
            self.GATV2_NO_ATTRIBUTES,
            self.TRANSFORMER_NO_ATTRIBUTES,
            self.STRUCTURAL_GLOBAL,
            self.GLOBAL_NO_STRUCTURE,
        }
        radial_only = self in {self.BASELINE_RADIAL, self.SCHNET_RADIAL}
        if not structural and not radial_only:
            return ()
        return tuple(
            f"edge.{relation}.{source}"
            for relation, sources in (("spatial", spatial), ("covalent", covalent))
            for source in sources
            if structural or source != "distance"
        )

    def needs_positions(self) -> bool:
        """Return whether this mechanism reads physical geometry beyond scalar edge records."""
        return self in {
            self.DIMENET,
            self.DIMENET_PLUS_PLUS,
            self.GVP,
            self.PAINN,
            self.EGNN_FIXED,
            self.EGNN_LATENT,
        }
