"""
Hooks that preserve love goals during GRA nullification.

These hooks ensure that when nullification occurs:
- Love goals are not erased
- Semantic connections to loved ones are preserved
- Love strength may even be amplified as a "through-invariant"
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
import copy


class LoveNullificationHook(ABC):
    """Abstract hook for love-preserving nullification."""

    @abstractmethod
    def pre_nullify(self, agent: Any) -> Dict[str, Any]:
        """Called before nullification. Returns state to preserve."""
        ...

    @abstractmethod
    def post_nullify(self, agent: Any, preserved_state: Dict[str, Any]) -> Any:
        """Called after nullification. Restores preserved state."""
        ...


class PreserveLoveGoalHook(LoveNullificationHook):
    """Hook that preserves the LoveGoal across nullification."""

    def pre_nullify(self, agent: Any) -> Dict[str, Any]:
        """Save love goal state."""
        preserved = {}
        if hasattr(agent, "love_goal") and agent.love_goal is not None:
            preserved["love_goal"] = copy.deepcopy(agent.love_goal)
        if hasattr(agent, "loved_ids"):
            preserved["loved_ids"] = list(agent.loved_ids)
        return preserved

    def post_nullify(self, agent: Any, preserved_state: Dict[str, Any]) -> Any:
        """Restore love goal and increment survival count."""
        if "love_goal" in preserved_state:
            agent.love_goal = preserved_state["love_goal"]
            agent.love_goal.nullification_survival_count += 1
        if "loved_ids" in preserved_state:
            agent.loved_ids = preserved_state["loved_ids"]
        return agent


class PreserveLovedConnectionsHook(LoveNullificationHook):
    """
    Hook that preserves and may amplify semantic connections
    to loved ones during nullification.
    """

    def __init__(self, amplification_factor: float = 1.1):
        self.amplification_factor = amplification_factor

    def pre_nullify(self, agent: Any) -> Dict[str, Any]:
        """Save love connection strengths."""
        preserved = {}
        if hasattr(agent, "affect") and agent.affect is not None:
            preserved["love_strength"] = agent.affect.love_strength
        return preserved

    def post_nullify(self, agent: Any, preserved_state: Dict[str, Any]) -> Any:
        """Restore and amplify love strength."""
        if "love_strength" in preserved_state and hasattr(agent, "affect") and agent.affect is not None:
            # Amplify love as a "through-invariant"
            agent.affect.love_strength = min(
                1.0,
                preserved_state["love_strength"] * self.amplification_factor,
            )
        return agent
