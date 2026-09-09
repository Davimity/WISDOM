"""Semantic Hard-Concrete gates used by every trainable WISDOM generation."""

from wisdom.models.gating.HardConcreteGate import HardConcreteGate
from wisdom.models.gating.SemanticGateRegistry import SemanticGateRegistry

__all__ = ["HardConcreteGate", "SemanticGateRegistry"]
