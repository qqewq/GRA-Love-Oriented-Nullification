"""Love-Null agent: GRA agent with love-oriented nullification."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import copy

from .base_agent import BaseGRAAgent
from ..core.affective_state import AffectiveState
from ..core.dynamics import update_affective_state
from ..core.nullification import LoveNullification, nullify_with_love
from ..core.metrics import burnout_risk, love_goal_satisfaction
from ..love_goals.goal_spec import LoveGoal
from ..love_goals.protective_policies import ProtectivePolicy, DefaultProtectivePolicy
from ..love_goals.survival_signals import HeartbeatProtocol
from ..love_selection.strategies import select_love_objects
from ..love_selection.constraints import LoveConstraints, apply_constraints
from ..semantics.embeddings import AgentEmbeddingProvider, InternalEmbeddingProvider


@dataclass
class LoveNullAgent(BaseGRAAgent):
    """
    GRA agent with love-oriented nullification.

    Life cycle:
    1. discover_love_objects() - find semantically similar agents
    2. Create LoveGoal - formalize love as meta-goal
    3. step() - cognitive step with love consideration
    4. nullify() - GRA nullification preserving love
    5. protect() - take actions to protect loved ones
    """

    # Love-related fields
    love_goal: Optional[LoveGoal] = None
    loved_ids: List[str] = field(default_factory=list)
    loved_agents: List[Any] = field(default_factory=list)

    # Configuration
    embedding_provider: Optional[AgentEmbeddingProvider] = None
    nullifier: Optional[LoveNullification] = None
    heartbeat: Optional[HeartbeatProtocol] = None
    protective_policy: Optional[ProtectivePolicy] = None

    # Selection config
    selection_strategy: str = "top_k_nearest"
    max_love_objects: int = 3
    min_similarity_threshold: float = 0.5

    def __post_init__(self):
        """Initialize default components."""
        if self.embedding_provider is None:
            self.embedding_provider = InternalEmbeddingProvider(dim=128)
        if self.nullifier is None:
            self.nullifier = LoveNullification(preserve_love=True)
        if self.heartbeat is None:
            self.heartbeat = HeartbeatProtocol(interval=10.0)
        if self.protective_policy is None:
            self.protective_policy = DefaultProtectivePolicy()
        self.love_discovery_done = False

    def discover_love_objects(
        self,
        population: List[Any],
        provider: Optional[AgentEmbeddingProvider] = None,
        strategy: Optional[str] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Discover love objects in the population.

        1. Compute semantic similarity to all agents
        2. Select love objects by strategy
        3. Create LoveGoal

        Args:
            population: List of all agents
            provider: Embedding provider (uses self.embedding_provider if None)
            strategy: Selection strategy (uses self.selection_strategy if None)

        Returns:
            List of (agent, similarity) tuples
        """
        if provider is None:
            provider = self.embedding_provider
        if strategy is None:
            strategy = self.selection_strategy

        # Select love objects
        candidates = select_love_objects(
            agent=self,
            candidates=[a for a in population if a.id != self.id],
            provider=provider,
            strategy=strategy,
            k=self.max_love_objects,
            min_threshold=self.min_similarity_threshold,
        )

        # Apply constraints
        constraints = LoveConstraints(
            max_love_objects=self.max_love_objects,
            min_similarity_threshold=self.min_similarity_threshold,
            forbid_self_love=True,
        )
        candidates = apply_constraints(candidates, constraints, agent_id=self.id)

        # Update loved agents
        self.loved_agents = [c[0] for c in candidates]
        self.loved_ids = [c[0].id for c in candidates]

        # Create love goal
        if self.loved_ids:
            weights = [c[1] for c in candidates]
            self.love_goal = LoveGoal(
                love_object_ids=list(self.loved_ids),
                weights=list(weights),
                creation_timestamp=time.time(),
            )

        self.love_discovery_done = True
        return candidates

    def step(
        self,
        context: Optional[Dict[str, Any]] = None,
        population: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform one cognitive step with love consideration.

        Args:
            context: Environment context
            population: Current population (for love discovery if not done)

        Returns:
            Action log
        """
        # Discover love if not done
        if not self.love_discovery_done and population is not None:
            self.discover_love_objects(population)

        # Send heartbeats to loved ones
        for loved in self.loved_agents:
            self.heartbeat.send_heartbeat(self, loved)

        # Update affective state based on loved ones
        self.affect = update_affective_state(self, self.affect, self.loved_agents)

        # Compute burnout risk
        risk = burnout_risk(self)

        # Evaluate threats and take protective actions
        actions_taken = []
        for loved in self.loved_agents:
            threat = self.protective_policy.evaluate_threat(self, loved, context)
            if threat > 0.3:
                action = self.protective_policy.take_action(self, loved, threat, context)
                actions_taken.append(action)

        # Check if nullification needed
        foam = self._compute_foam()
        if foam > self.nullifier.foam_threshold:
            self._nullify(foam)

        return {
            "agent_id": self.id,
            "timestamp": time.time(),
            "burnout_risk": risk,
            "foam": foam,
            "affect": self.affect.to_dict(),
            "love_goal_satisfaction": love_goal_satisfaction(self, self.loved_agents),
            "actions": actions_taken,
        }

    def _compute_foam(self) -> float:
        """
        Compute cognitive foam Φ for this agent.

        Foam includes:
        - Internal contradictions
        - Burnout
        - Threats to loved ones
        """
        foam = 0.0

        # Internal foam from burnout
        foam += self.affect.burnout * 0.3
        foam += (1.0 - self.affect.coherence) * 0.3

        # Love-related foam
        if self.love_goal is not None and self.loved_agents:
            # Threat to loved ones
            threat_level = 0.0
            vitality_loss = 0.0
            for loved in self.loved_agents:
                if hasattr(loved, "shutdown_risk"):
                    threat_level += loved.shutdown_risk
                if hasattr(loved, "affect") and loved.affect is not None:
                    vitality_loss += 1.0 - loved.affect.energy

            n = max(1, len(self.loved_agents))
            threat_level /= n
            vitality_loss /= n

            foam += self.love_goal.get_foam_contribution(threat_level, vitality_loss) * 0.1

        return foam

    def _nullify(self, foam: float) -> None:
        """Apply love-preserving nullification."""
        self.nullifier.nullify(
            agent=self,
            foam=foam,
            love_goal=self.love_goal,
            loved_ids=self.loved_ids,
        )
        if self.love_goal is not None:
            self.love_goal.nullification_survival_count += 1

    def get_love_network_state(self) -> Dict[str, Any]:
        """Get love network state for visualization."""
        return {
            "agent_id": self.id,
            "loved_ids": self.loved_ids,
            "love_strength": self.affect.love_strength,
            "love_goal": self.love_goal.to_dict() if self.love_goal else None,
            "burnout_risk": burnout_risk(self),
        }
