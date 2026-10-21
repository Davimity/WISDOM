"""Closed V8d controls, separate from historical V8b interaction profiles."""

from enum import Enum
from wisdom.models.InteractionRound import InteractionRound


class FeedbackPropagationMode(str, Enum):
    """Declare exactly one finite feedback experiment; NONE preserves archived constructors."""

    NONE = "none"
    D0   = "d0"
    D1   = "d1"
    D2   = "d2"
    D3   = "d3"
    D4   = "d4"
    D5   = "d5"
    D6   = "d6"
    D7   = "d7"
    D8   = "d8"

    def interaction(self) -> InteractionRound:
        """Return the existing V8b route reused by this explicitly selected control.

        Returns:
            Single pass, depth-only, null-message update, or real feedback. NONE has no override.

        Raises:
            ValueError: NONE was treated as an experiment instead of an absent override.
        """
        if self is self.NONE:
            raise ValueError("feedback_propagation=none does not override interaction_round")
        if self is self.D0:
            return InteractionRound.SINGLE
        if self in {self.D1, self.D7}:
            return InteractionRound.DEPTH_CONTROL
        if self in {self.D2, self.D6}:
            return InteractionRound.DEPTH_CONTROL_PARAMETER_MATCHED
        return InteractionRound.BIDIRECTIONAL

    def layers(self) -> int:
        """Return post-feedback graph depth; D8 uses an explicit nodewise capacity control.

        Returns:
            One for D4, two for D5-D7, and zero when no second GNN is executed.
        """
        return 1 if self is self.D4 else 2 if self in {self.D5, self.D6, self.D7} else 0
