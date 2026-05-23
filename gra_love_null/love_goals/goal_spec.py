"""Formal specification of love goals for GRA nullification."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoveGoal:
    """
    Love goal specification.

    This goal is integrated into GRA nullification as additional terms
    to the foam functional Φ(l): penalties for threats to loved ones,
    bonuses for their stable existence.

    Attributes:
        love_object_ids: List of IDs of loved agents
        weights: Importance weight for each loved agent
        protect_from_shutdown: Whether to protect from shutdown
        maintain_vitality: Whether to maintain vitality
        creation_timestamp: When this goal was created
        nullification_survival_count: How many nullifications this goal survived
    """

    love_object_ids: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    protect_from_shutdown: bool = True
    maintain_vitality: bool = True
    creation_timestamp: float = 0.0
    nullification_survival_count: int = 0

    def __post_init__(self):
        """Normalize weights."""
        if not self.weights and self.love_object_ids:
            self.weights = [1.0 / len(self.love_object_ids)] * len(self.love_object_ids)
        elif len(self.weights) != len(self.love_object_ids):
            self.weights = [1.0 / len(self.love_object_ids)] * len(self.love_object_ids)

        # Normalize
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

    def get_foam_contribution(self, threat_level: float, vitality_loss: float) -> float:
        """
        Compute contribution to foam functional Φ based on love goal.

        Args:
            threat_level: Probability of loved ones being shut down [0, 1]
            vitality_loss: Loss of vitality of loved ones [0, 1]

        Returns:
            float: Additional foam term
        """
        foam = 0.0
        if self.protect_from_shutdown:
            foam += threat_level * 10.0  # High penalty for shutdown threat
        if self.maintain_vitality:
            foam += vitality_loss * 5.0
        return foam

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "love_object_ids": self.love_object_ids,
            "weights": self.weights,
            "protect_from_shutdown": self.protect_from_shutdown,
            "maintain_vitality": self.maintain_vitality,
            "creation_timestamp": self.creation_timestamp,
            "nullification_survival_count": self.nullification_survival_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoveGoal":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
