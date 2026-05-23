"""Metrics for love-oriented nullification."""

from typing import Any, List, Optional
import numpy as np


def burnout_risk(agent: Any) -> float:
    """
    Compute burnout risk for an agent.

    High burnout + low energy + low meaning = high risk.

    Args:
        agent: Agent with affective state

    Returns:
        float: Burnout risk in [0, 1]
    """
    if not hasattr(agent, "affect") or agent.affect is None:
        return 0.5

    affect = agent.affect
    risk = (
        0.4 * affect.burnout
        + 0.3 * (1.0 - affect.energy)
        + 0.3 * (1.0 - affect.meaning)
    )
    return max(0.0, min(1.0, risk))


def love_goal_satisfaction(
    agent: Any,
    loved_agents: List[Any],
) -> float:
    """
    Compute how well the love goal is satisfied.

    Measures: are loved agents alive, energetic, coherent?

    Args:
        agent: The loving agent
        loved_agents: List of loved agents

    Returns:
        float: Satisfaction in [0, 1]
    """
    if not loved_agents:
        return 0.0

    scores = []
    for loved in loved_agents:
        score = 1.0
        if hasattr(loved, "is_alive"):
            score *= float(loved.is_alive)
        if hasattr(loved, "affect") and loved.affect is not None:
            score *= 0.5 + 0.5 * loved.affect.energy
            score *= 0.5 + 0.5 * loved.affect.coherence
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def semantic_coherence_with_loved_ones(
    agent: Any,
    loved_agents: List[Any],
    embedding_provider: Any,
) -> float:
    """
    Compute semantic coherence between agent and its loved ones.

    Args:
        agent: The loving agent
        loved_agents: List of loved agents
        embedding_provider: Embedding provider

    Returns:
        float: Mean semantic similarity with loved ones in [0, 1]
    """
    if not loved_agents:
        return 0.0

    from ..semantics.similarity import semantic_similarity

    similarities = []
    for loved in loved_agents:
        sim = semantic_similarity(agent, loved, embedding_provider)
        similarities.append(max(0.0, sim))

    return float(np.mean(similarities)) if similarities else 0.0
