"""TransferMessage vocabulary; scientific values are unchanged."""

from enum import Enum


class TransferMessage(str, Enum):
    """Change transmitted values independently of the historical compatibility score."""

    ATOM = "atom"
    PAIR = "pair"
    POINT_PAIR = "point_pair"
    LOW_RANK = "low_rank"
