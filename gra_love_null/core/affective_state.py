"""Affective state model for GRA agents with love."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AffectiveState:
    """
    Affective state of an agent.

    Attributes:
        energy: Vital energy level [0, 1]
        burnout: Burnout level [0, 1]
        meaning: Sense of meaning [0, 1]
        longing: Longing for loved ones [0, 1]
        joy: Joy level [0, 1]
        love_strength: Strength of love bonds [0, 1]
        coherence: Internal coherence [0, 1]
    """

    energy: float = 1.0
    burnout: float = 0.0
    meaning: float = 0.5
    longing: float = 0.0
    joy: float = 0.5
    love_strength: float = 0.0
    coherence: float = 1.0

    def __post_init__(self):
        """Clamp values to [0, 1]."""
        for attr_name in [
            "energy", "burnout", "meaning", "longing",
            "joy", "love_strength", "coherence"
        ]:
            setattr(self, attr_name, max(0.0, min(1.0, getattr(self, attr_name))))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "energy": self.energy,
            "burnout": self.burnout,
            "meaning": self.meaning,
            "longing": self.longing,
            "joy": self.joy,
            "love_strength": self.love_strength,
            "coherence": self.coherence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AffectiveState":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
