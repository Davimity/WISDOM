"""VF reconstruction evidence in original units, with explicit protein/region support counts."""

import numpy as np

from typing import Any
from scipy.stats import spearmanr


class PhysicalMetricSuite:
    """Aggregate whole-bank evidence without interpreting low reconstruction loss as function."""

    def __init__(self, sign_band: float = 0.1) -> None:
        """Configure the electrostatic near-zero exclusion band for sign agreement.

        Args:
            sign_band: Absolute potential in kBT/e below which reference sign is not scored.
        """
        self.sign_band = sign_band
        self.rows: list[dict[str, Any]] = []

    def add(self, identifier: str, predicted: np.ndarray, target: np.ndarray,
            valid: np.ndarray, regions: np.ndarray) -> None:
        """Record one protein's metrics separately for each valid field and geometric region.

        Args:
            identifier: Stable physical member identity.
            predicted: Original-unit decoder outputs [Q,3], potential kBT/e and LJ kJ/mol.
            target: Original-unit immutable reference [Q,3].
            valid: Independent Boolean field masks [Q,3].
            regions: Disjoint geometric stratum IDs [Q], not function/evidence labels.

        Raises:
            ValueError: A valid prediction is non-finite; overflow is not an acceptable target fit.
        """
        if not np.isfinite(predicted[valid]).all():
            raise ValueError("VF produced non-finite predictions in original physical units")
        for channel, field in enumerate(("potential", "lj1", "lj2")):
            for region in (-1, 0, 1, 2, 3):
                mask = valid[:, channel] & ((regions == region) if region >= 0 else True)
                if not mask.any():
                    continue
                actual, expected = predicted[mask, channel], target[mask, channel]
                difference = actual - expected
                correlation = (float(np.corrcoef(actual, expected)[0, 1])
                    if len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(expected) > 0 else None)
                rank = (float(spearmanr(actual, expected).statistic)
                    if len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(expected) > 0 else None)
                outside = np.abs(expected) > self.sign_band
                self.rows.append({"identifier": identifier, "field": field, "region": region,
                    "queries": int(mask.sum()), "mae": float(np.abs(difference).mean()),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "pearson": correlation, "spearman": rank,
                    "sign_accuracy": (float(np.mean(np.sign(actual[outside]) ==
                        np.sign(expected[outside]))) if channel == 0 and outside.any() else None)})

    def metrics(self) -> dict[str, float]:
        """Return macro-protein metrics and supporting counts for native LF curves.

        Returns:
            Available metric names only. Missing/constant correlations remain unavailable rather
            than zero; each averaged metric has a protein support count. Region -1 is overall.
        """
        result: dict[str, float] = {}
        for field in ("potential", "lj1", "lj2"):
            for region in (-1, 0, 1, 2, 3):
                rows = [row for row in self.rows
                        if row["field"] == field and row["region"] == region]
                prefix = f"physics_{field}" + (f"_region_{region}" if region >= 0 else "")
                result[prefix + "_proteins"] = float(len(rows))
                result[prefix + "_queries"] = float(sum(row["queries"] for row in rows))
                for metric in ("mae", "rmse", "pearson", "spearman", "sign_accuracy"):
                    available = [row[metric] for row in rows if row[metric] is not None]
                    if available:
                        result[prefix + "_" + metric] = float(np.mean(available))
                        result[prefix + "_" + metric + "_proteins"] = float(len(available))
        return result
