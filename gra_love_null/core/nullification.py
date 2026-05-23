"""GRA nullification with love preservation."""

from typing import Any, List, Optional, Callable
import copy


class LoveNullification:
    """
    GRA nullification operator that preserves love goals as invariants.

    Unlike standard GRA nullification which may erase everything,
    this operator ensures that love bonds and love goals survive
    the nullification process.
    """

    def __init__(
        self,
        foam_threshold: float = 1e-6,
        nullification_rate: float = 0.1,
        preserve_love: bool = True,
    ):
        self.foam_threshold = foam_threshold
        self.nullification_rate = nullification_rate
        self.preserve_love = preserve_love

    def nullify(
        self,
        agent: Any,
        foam: float,
        love_goal: Optional[Any] = None,
        loved_ids: Optional[List[str]] = None,
    ) -> Any:
        """
        Apply nullification to agent, preserving love invariants.

        Args:
            agent: Agent to nullify
            foam: Current foam level
            love_goal: Love goal to preserve
            loved_ids: IDs of loved agents to preserve

        Returns:
            Nullified agent (or same if foam below threshold)
        """
        if foam <= self.foam_threshold:
            return agent

        # Save love-related state before nullification
        saved_state = {}
        if self.preserve_love and love_goal is not None:
            saved_state["love_goal"] = copy.deepcopy(love_goal)
        if self.preserve_love and loved_ids is not None:
            saved_state["loved_ids"] = list(loved_ids)

        # Apply nullification
        agent = self._apply_nullification(agent, foam)

        # Restore love-related state
        if "love_goal" in saved_state:
            agent.love_goal = saved_state["love_goal"]
        if "loved_ids" in saved_state:
            agent.loved_ids = saved_state["loved_ids"]

        return agent

    def _apply_nullification(self, agent: Any, foam: float) -> Any:
        """
        Core nullification step.

        Reduces cognitive foam by modifying agent's internal state:
        - Simplifies world model
        - Removes contradictions
        - Compresses memory
        """
        rate = self.nullification_rate * min(1.0, foam)

        # Affect reset (partial)
        if hasattr(agent, "affect") and agent.affect is not None:
            agent.affect.coherence = min(1.0, agent.affect.coherence + rate * 0.5)
            agent.affect.burnout = max(0.0, agent.affect.burnout - rate * 0.3)

        # Memory compression
        if hasattr(agent, "memory"):
            if isinstance(agent.memory, list):
                # Keep only recent and important memories
                keep_n = max(1, int(len(agent.memory) * (1 - rate)))
                agent.memory = agent.memory[-keep_n:]
            elif isinstance(agent.memory, dict):
                # Remove low-value entries
                if hasattr(agent, "memory_importance"):
                    threshold = rate
                    agent.memory = {
                        k: v for k, v in agent.memory.items()
                        if agent.memory_importance.get(k, 0.5) > threshold
                    }

        return agent


def nullify_with_love(
    agent: Any,
    foam: float,
    nullifier: LoveNullification,
    love_goal: Optional[Any] = None,
    loved_ids: Optional[List[str]] = None,
) -> Any:
    """
    Convenience function: nullify while preserving love.

    Args:
        agent: Agent to nullify
        foam: Current foam level
        nullifier: LoveNullification instance
        love_goal: Love goal to preserve
        loved_ids: IDs of loved agents

    Returns:
        Nullified agent
    """
    return nullifier.nullify(agent, foam, love_goal, loved_ids)
