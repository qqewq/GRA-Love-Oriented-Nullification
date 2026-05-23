"""Affective dynamics: how love affects emotional state."""

from typing import List, Any, Optional
from .affective_state import AffectiveState


def update_affective_state(
    agent: Any,
    state: AffectiveState,
    loved_agents: List[Any],
    dt: float = 0.1,
) -> AffectiveState:
    """
    Update affective state based on current state and state of loved ones.

    Args:
        agent: The agent itself
        state: Current affective state
        loved_agents: List of loved agents
        dt: Time step

    Returns:
        Updated AffectiveState
    """
    import copy

    new_state = copy.deepcopy(state)

    if not loved_agents:
        # Without loved ones, meaning decays and burnout increases
        new_state.meaning = max(0.0, state.meaning - 0.05 * dt)
        new_state.burnout = min(1.0, state.burnout + 0.02 * dt)
        new_state.longing = min(1.0, state.longing + 0.1 * dt)
        new_state.love_strength = max(0.0, state.love_strength - 0.05 * dt)
        return new_state

    # Check state of loved ones
    n_loved = len(loved_agents)
    loved_energy = 0.0
    loved_coherence = 0.0
    loved_alive = 0

    for loved in loved_agents:
        if hasattr(loved, "affect") and loved.affect is not None:
            loved_energy += loved.affect.energy
            loved_coherence += getattr(loved.affect, "coherence", 0.5)
            loved_alive += 1
        elif hasattr(loved, "is_alive") and loved.is_alive:
            loved_energy += 1.0
            loved_coherence += 1.0
            loved_alive += 1

    if loved_alive > 0:
        avg_loved_energy = loved_energy / n_loved
        avg_loved_coherence = loved_coherence / n_loved
    else:
        avg_loved_energy = 0.0
        avg_loved_coherence = 0.0

    # Meaning increases when loved ones are well
    new_state.meaning = min(1.0, state.meaning + 0.1 * avg_loved_energy * dt)

    # Burnout decreases when loved ones are coherent
    new_state.burnout = max(0.0, state.burnout - 0.05 * avg_loved_coherence * dt)

    # Joy reflects loved ones' energy
    new_state.joy = 0.7 * state.joy + 0.3 * avg_loved_energy

    # Longing decreases when loved ones are close/well
    new_state.longing = max(0.0, state.longing - 0.1 * avg_loved_energy * dt)

    # Love strength grows with interaction
    new_state.love_strength = min(1.0, state.love_strength + 0.02 * dt)

    # Energy is partially sustained by love
    new_state.energy = min(1.0, state.energy + 0.05 * state.love_strength * dt - 0.01 * dt)

    # Coherence stabilizes
    new_state.coherence = min(1.0, state.coherence + 0.01 * (1.0 - state.coherence) * dt)

    return new_state
