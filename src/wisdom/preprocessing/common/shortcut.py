"""Train-only covariate baselines for acquisition and morphology shortcut audits."""

import numpy as np

from typing import Any
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from collections.abc import Mapping, Sequence
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


def shortcut_baseline(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    """Measure whether simple acquisition/morphology covariates predict validation labels.

    Args:
        rows: Fixed split records, already checked for indivisible leakage groups.
        seed: Deterministic logistic-regression seed, not a model-selection sweep.

    Returns:
        Validation AUROC/AUPRC/balanced accuracy and support, or an explicit unsupported
        verdict when a class has fewer than two train/validation proteins. Test is not opened.
        The baseline is descriptive; high discrimination flags a possible shortcut, not proof.
    """
    train = [row for row in rows if row["split"] == "train"]
    val   = [row for row in rows if row["split"] == "validation"]
    support = {name: dict(Counter(int(row["label"]) for row in split))
               for name, split in (("train", train), ("validation", val))}
    if any(counts.get(label, 0) < 2 for counts in support.values() for label in (0, 1)):
        return {"support": support, "available": False,
                "reason": "fewer_than_two_proteins_per_class_and_split"}
    numeric = ("sequence_length", "resolution", "radius_of_gyration_normalized",
               "aspect_ratio", "packing_density")
    categorical = ("origin", "method")
    matrices = [np.asarray([
        [row.get(name, np.nan) if row.get(name) is not None else np.nan for name in numeric]
        + [str(row.get(name, "unknown")) for name in categorical]
        for row in split
    ], dtype=object) for split in (train, val)]
    transform = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         list(range(len(numeric)))),
        (OneHotEncoder(handle_unknown="ignore"),
         list(range(len(numeric), len(numeric) + len(categorical)))),
    )
    predictor = make_pipeline(transform, LogisticRegression(
        max_iter=1000, random_state=seed, class_weight="balanced"))
    predictor.fit(matrices[0], [row["label"] for row in train])
    labels = np.asarray([row["label"] for row in val])
    probabilities = predictor.predict_proba(matrices[1])[:, 1]
    return {
        "available": True, "support": support, "features": [*numeric, *categorical],
        "fit_split": "train", "evaluation_split": "validation",
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, probabilities >= 0.5)),
        "interpretation": "High discrimination warrants acquisition/shape-confound review; "
                          "this is not a functional model and has no surface GT input.",
    }
