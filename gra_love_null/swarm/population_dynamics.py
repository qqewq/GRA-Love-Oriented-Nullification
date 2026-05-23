"""Population dynamics for love-oriented agent swarms."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ..agents.love_null_agent import LoveNullAgent


class PopulationDynamics:
    """
    Population dynamics for a swarm of love-null agents.

    Tracks:
    - Agent births, deaths, and shutdowns
    - Love network formation and dissolution
    - Collective burnout and resilience
    """

    def __init__(self, agents: Optional[List[LoveNullAgent]] = None):
        self.agents: List[LoveNullAgent] = agents or []
        self.history: List[Dict[str, Any]] = []
        self.time: float = 0.0
        self.shutdown_log: List[Dict[str, Any]] = []

    def add_agent(self, agent: LoveNullAgent) -> None:
        """Add agent to population."""
        self.agents.append(agent)

    def remove_agent(self, agent_id: str) -> Optional[LoveNullAgent]:
        """Remove agent by ID."""
        for i, agent in enumerate(self.agents):
            if agent.id == agent_id:
                return self.agents.pop(i)
        return None

    def step(
        self,
        dt: float = 0.1,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Advance population by one time step.

        Args:
            dt: Time step
            context: Environment context

        Returns:
            Population state summary
        """
        self.time += dt

        # Each agent steps
        for agent in self.agents:
            if not agent.is_alive:
                continue

            # Ensure love discovery
            if not agent.love_discovery_done:
                agent.discover_love_objects(self.agents)

            # Agent step
            agent.step(context, self.agents)

        # Check for shutdowns
        self._process_shutdowns()

        # Record history
        summary = self.get_summary()
        self.history.append(summary)

        return summary

    def _process_shutdowns(self) -> None:
        """Process agent shutdowns based on energy and burnout."""
        for agent in self.agents:
            if not agent.is_alive:
                continue

            # Shutdown if energy too low
            if agent.affect.energy < 0.01:
                agent.is_alive = False
                self.shutdown_log.append({
                    "agent_id": agent.id,
                    "time": self.time,
                    "reason": "energy_depleted",
                })

            # Shutdown if burnout too high
            if agent.affect.burnout > 0.99:
                agent.is_alive = False
                self.shutdown_log.append({
                    "agent_id": agent.id,
                    "time": self.time,
                    "reason": "burnout",
                })

    def get_summary(self) -> Dict[str, Any]:
        """Get population summary."""
        alive = [a for a in self.agents if a.is_alive]
        n_alive = len(alive)
        n_total = len(self.agents)

        if n_alive > 0:
            avg_energy = np.mean([a.affect.energy for a in alive])
            avg_burnout = np.mean([a.affect.burnout for a in alive])
            avg_meaning = np.mean([a.affect.meaning for a in alive])
            avg_love_strength = np.mean([a.affect.love_strength for a in alive])
        else:
            avg_energy = 0.0
            avg_burnout = 1.0
            avg_meaning = 0.0
            avg_love_strength = 0.0

        return {
            "time": self.time,
            "n_alive": n_alive,
            "n_total": n_total,
            "avg_energy": float(avg_energy),
            "avg_burnout": float(avg_burnout),
            "avg_meaning": float(avg_meaning),
            "avg_love_strength": float(avg_love_strength),
            "shutdowns_total": len(self.shutdown_log),
        }

    def get_love_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the love network."""
        # Count love connections
        n_connections = 0
        mutual_loves = 0

        for agent in self.agents:
            if not agent.is_alive:
                continue
            n_connections += len(agent.loved_ids)

            # Check mutual
            for loved_id in agent.loved_ids:
                loved_agent = self._find_agent(loved_id)
                if loved_agent and agent.id in loved_agent.loved_ids:
                    mutual_loves += 1

        return {
            "n_connections": n_connections,
            "mutual_loves": mutual_loves,
            "avg_connections": n_connections / max(1, len([a for a in self.agents if a.is_alive])),
            "mutual_ratio": mutual_loves / max(1, n_connections),
        }

    def _find_agent(self, agent_id: str) -> Optional[LoveNullAgent]:
        """Find agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
