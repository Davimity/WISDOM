import pytest
import torch

from wisdom.evaluation.SurfaceFaithfulnessAudit import SurfaceFaithfulnessAudit
from wisdom.models.WisdomV2 import WisdomV2


def _model(pooling_type: str) -> WisdomV2:
    """Build a tiny pooling-capable model for boundary-only audits.

    Args:
        pooling_type: Public WISDOM pooling name.

    Returns:
        Small V2 instance; its encoder is not executed by these tests.
    """
    return WisdomV2(
        hidden_dim=4,
        embedding_dim=2,
        atomic_layers=1,
        projection_depth=1,
        surface_layers=1,
        dropout=0.0,
        curvature_features=6,
        pooling_type=pooling_type,
    )


def test_surface_faithfulness_reports_equal_area_controls_without_ground_truth() -> None:
    model  = _model("max")
    logits = torch.tensor([-2.0, -1.0, 0.0, 5.0, 1.0, -3.0, 0.0, 4.0, 1.0, -1.0])
    output = {
        "logits":             torch.tensor([5.0, 4.0]),
        "surface_logits":     logits,
        "surface_embeddings": torch.zeros(10, 4),
    }
    batch = {
        "surface_area_weights": torch.ones(10),
        "surface_batch":        torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        "surface_ptr":          torch.tensor([0, 5, 10]),
        "target":               torch.tensor([1.0, 1.0]),
    }

    sums, count = SurfaceFaithfulnessAudit().compute(model, output, batch)

    assert count == 2
    assert sums["faithfulness_deletion_top_20"] > sums["faithfulness_deletion_bottom_20"]
    assert sums["faithfulness_insertion_top_20"] > sums["faithfulness_insertion_bottom_20"]


def test_surface_faithfulness_rejects_operator_changing_regional_deletion() -> None:
    model = _model("local_mean_max")
    output = {
        "logits":             torch.tensor([1.0, -1.0]),
        "surface_logits":     torch.tensor([1.0, -1.0]),
        "surface_embeddings": torch.zeros(2, 4),
    }
    batch = {
        "surface_area_weights": torch.ones(2),
        "surface_batch":        torch.tensor([0, 0]),
        "surface_ptr":          torch.tensor([0, 2]),
        "target":               torch.tensor([1.0]),
    }

    with pytest.raises(ValueError, match="diffusion operator"):
        SurfaceFaithfulnessAudit().compute(model, output, batch)
