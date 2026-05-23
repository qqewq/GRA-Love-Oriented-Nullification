"""Configuration classes for GRA-Love-Oriented-Nullification."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import os


@dataclass
class GRAConfig:
    """Configuration for GRA multiverse integration."""

    multiverse_levels: int = 3
    foam_threshold: float = 1e-6
    nullification_rate: float = 0.1
    max_iterations: int = 1000
    seed: int = 42


@dataclass
class LoveConfig:
    """Configuration for love selection and goals."""

    # Love discovery
    max_love_objects: int = 5
    min_similarity_threshold: float = 0.7
    selection_strategy: str = "top_k_nearest"  # "top_k_nearest", "mutual_love", "cluster_centroid"
    mutual_love_required: bool = False

    # Love goals
    protect_from_shutdown: bool = True
    maintain_vitality: bool = True
    heart_beat_interval: float = 10.0  # seconds

    # Semantic embedding
    embedding_provider: str = "internal"  # "internal", "llm"
    embedding_dim: int = 128

    # Constraints
    forbid_dominance_love: bool = True
    dominance_threshold: float = 0.9

    # Protective policies
    resource_share_max: float = 0.5
    self_sacrifice_energy_threshold: float = 0.2


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON file or return defaults."""
    default_config = {
        "gra": GRAConfig().__dict__,
        "love": LoveConfig().__dict__,
    }
    if path and os.path.exists(path):
        with open(path, "r") as f:
            user_config = json.load(f)
        # Merge user config with defaults
        for section in default_config:
            if section in user_config:
                default_config[section].update(user_config[section])
    return default_config
