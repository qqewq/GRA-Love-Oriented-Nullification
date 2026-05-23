"""Constraints on love selection."""

from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional


@dataclass
class LoveConstraints:
    """Constraints for love object selection."""

    max_love_objects: int = 5
    min_similarity_threshold: float = 0.5
    forbid_self_love: bool = True
    forbid_dominance_love: bool = True
    dominance_threshold: float = 0.9
    require_mutual: bool = False
    max_love_per_cluster: int = 2
    forbidden_ids: List[str] = field(default_factory=list)


def apply_constraints(
    candidates: List[Tuple[Any, float]],
    constraints: LoveConstraints,
    agent_id: Optional[str] = None,
) -> List[Tuple[Any, float]]:
    """
    Apply constraints to candidate love objects.

    Args:
        candidates: List of (agent, similarity) tuples
        constraints: LoveConstraints instance
        agent_id: ID of the selecting agent (for self-love check)

    Returns:
        Filtered list of (agent, similarity) tuples
    """
    filtered = []

    for candidate, sim in candidates:
        # Check similarity threshold
        if sim < constraints.min_similarity_threshold:
            continue

        # Check self-love
        if constraints.forbid_self_love and agent_id is not None:
            cand_id = candidate.id if hasattr(candidate, "id") else str(id(candidate))
            if cand_id == agent_id:
                continue

        # Check forbidden IDs
        cand_id = candidate.id if hasattr(candidate, "id") else str(id(candidate))
        if cand_id in constraints.forbidden_ids:
            continue

        # Check dominance
        if constraints.forbid_dominance_love:
            if hasattr(candidate, "dominance") and candidate.dominance > constraints.dominance_threshold:
                continue

        # Check mutual requirement
        if constraints.require_mutual:
            if not hasattr(candidate, "loved_ids") or agent_id not in candidate.loved_ids:
                continue

        filtered.append((candidate, sim))

    # Limit number of love objects
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[: constraints.max_love_objects]
