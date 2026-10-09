"""Versioned, label-free vocabulary for optional protein surface fields."""

from typing import ClassVar
from collections.abc import Sequence


class SurfaceFeatureSchema:
    """Declare channel order, units, and scientific interpretation independently of tasks."""

    VERSION = "1.0"
    GENERIC = (
        "formal_charge_density", "hbond_donor_density", "hbond_acceptor_density",
        "aromatic_density", "hydropathy", "polarity", "N_density", "O_density", "S_density",
    )
    ZINC = (
        "zn_lewis_strict", "zn_N_density", "zn_O_density", "zn_S_density",
        "zn_ched_density", "zn_specificity_ratio",
        "ched_ca_compactness", "ched_constellation_score",
    )
    GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "generic_minimal": GENERIC[:3],
        "generic_chemistry": GENERIC[:6],
        "generic_elemental": GENERIC[6:],
        "generic_basic": GENERIC,
        "zinc_interpretable": ZINC,
    }

    @classmethod
    def resolve(
        cls, names: Sequence[str] = (), group: str | None = None,
        exclude: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Resolve a deterministic ordered subset without inventing unknown channels.

        Args:
            names: Explicit channel names, appended after a group's channels.
            group: Optional named group from ``GROUPS``.
            exclude: Channels to remove after expansion; unknown names are errors.

        Returns:
            Deduplicated names preserving researcher-authored order.

        Raises:
            ValueError: If a feature or group is unknown.
        """
        if group is not None and group not in cls.GROUPS:
            raise ValueError(f"unknown surface feature group: {group!r}")
        requested = [*cls.GROUPS.get(group or "", ()), *names]
        unknown   = set([*requested, *exclude]) - set(cls.GENERIC + cls.ZINC)
        if unknown:
            raise ValueError(f"unknown surface features: {sorted(unknown)}")
        return tuple(name for name in dict.fromkeys(requested) if name not in exclude)

    @classmethod
    def describe(cls, name: str) -> dict[str, str]:
        """Describe one field's units, reduction and approximation.

        Args:
            name: Valid channel name.

        Returns:
            JSON-compatible provenance, units and reduction convention.

        Raises:
            ValueError: If ``name`` is unknown.
        """
        cls.resolve([name])
        average = name in {"hydropathy", "polarity"}
        ratio   = name == "zn_specificity_ratio"
        return {
            "name": name,
            "units": "Kyte-Doolittle scale / 4.5" if name == "hydropathy" else (
                "elementary charge weighted sum" if name == "formal_charge_density"
                else "dimensionless"
            ),
            "reduction": "normalized_CA_spread" if name == "ched_ca_compactness" else (
                "normalized_CA_spread_times_type_coverage" if name == "ched_constellation_score"
                else "average" if average else "ratio" if ratio else "weighted_sum"
            ),
            "source": "protein-only universal NPZ; AtomicDescriptors residue/atom rules",
            "limitations": (
                "Gaussian neighborhood proxy, not volume-normalized density or electrostatic "
                "potential; donor/acceptor identities do not resolve protonation or tautomerism"
            ),
            "scope": "zinc_motivated" if name in cls.ZINC else "generic",
            "formula": (
                "sigma/(sigma+weighted_CA_radius_of_gyration), >=2 CHED CAs; otherwise zero"
                if name == "ched_ca_compactness" else (
                    "CHED-type coverage/4 times CA compactness; geometric heuristic, not affinity"
                    if name == "ched_constellation_score" else "Gaussian weighted atom reduction"
                )
            ),
        }
