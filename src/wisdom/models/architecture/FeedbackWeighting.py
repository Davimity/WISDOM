"""FeedbackWeighting vocabulary; scientific values are unchanged."""

from enum import Enum


class FeedbackWeighting(str, Enum):
    """Select the integration measure of surface-to-atom aggregation."""

    POINT = "point"
    AREA  = "area"
