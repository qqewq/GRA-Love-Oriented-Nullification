"""
Adapter for integrating with GRA-Multiverse-Final.

Provides interface for reading foam functional Φ(l),
calling meta-nullification, and integrating love goals
into the multiverse framework.
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np


class GRAAdapter:
    """
    Adapter for GRA-Multiverse-Final integration.

    Connects love-oriented agents to the GRA multiverse:
    - Reads Φ(l) from multiverse
    - Injects love constraints into nullification
    - Tracks loved agents across multiverse levels
    """

    def __init__(self, multiverse_config: Optional[Dict[str, Any]] = None):
        self.multiverse_config = multiverse_config or {}
        self.levels: int = self.multiverse_config.get("multiverse_levels", 3)
        self.foam_values: List[float] = [0.0] * self.levels
        self.love_constraints: Dict[int, List[Any]] = {}  # level -> [love constraints]

    def read_foam(self, level: int, multiverse_state: Any) -> float:
        """
        Read Φ(l) from multiverse state.

        Args:
            level: Multiverse level
            multiverse_state: State of the multiverse at that level

        Returns:
            float: Foam value Φ(l)
        """
        if hasattr(multiverse_state, "foam"):
            return float(multiverse_state.foam)
        elif isinstance(multiverse_state, dict) and "foam" in multiverse_state:
            return float(multiverse_state["foam"])
        else:
            # Estimate foam from state complexity
            return self._estimate_foam(multiverse_state)

    def _estimate_foam(self, state: Any) -> float:
        """Estimate foam from state complexity."""
        if isinstance(state, dict):
            return 0.1 * len(state)
        elif isinstance(state, (list, tuple)):
            return 0.1 * len(state)
        elif isinstance(state, np.ndarray):
            return float(np.std(state))
        else:
            return 0.5

    def inject_love_constraints(
        self,
        level: int,
        love_goal: Optional[Any] = None,
        loved_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Inject love constraints into nullification process.

        Args:
            level: Multiverse level
            love_goal: Love goal to preserve
            loved_ids: IDs of loved agents

        Returns:
            Dict with constraint specifications
        """
        constraints = {
            "level": level,
            "preserve_agents": loved_ids or [],
            "preserve_goal": love_goal.to_dict() if love_goal and hasattr(love_goal, "to_dict") else None,
            "foam_regularization": {
                "shutdown_penalty": 10.0 if love_goal and love_goal.protect_from_shutdown else 0.0,
                "vitality_penalty": 5.0 if love_goal and love_goal.maintain_vitality else 0.0,
            },
        }
        self.love_constraints[level] = constraints
        return constraints

    def call_meta_nullification(
        self,
        agent: Any,
        foam: float,
        nullification_fn: Optional[Callable] = None,
    ) -> Any:
        """
        Call meta-nullification with love constraints.

        Args:
            agent: Agent to nullify
            foam: Current foam level
            nullification_fn: External nullification function

        Returns:
            Nullified agent
        """
        if nullification_fn is not None:
            return nullification_fn(agent, foam)
        else:
            # Fallback: internal nullification
            from ..core.nullification import nullify_with_love
            return nullify_with_love(
                agent,
                foam,
                nullifier=None,  # Will use agent's nullifier
                love_goal=agent.love_goal if hasattr(agent, "love_goal") else None,
                loved_ids=agent.loved_ids if hasattr(agent, "loved_ids") else None,
            )
