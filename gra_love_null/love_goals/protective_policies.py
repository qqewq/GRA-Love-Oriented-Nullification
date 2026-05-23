"""Protective policies for loved agents."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
import time


class ProtectivePolicy(ABC):
    """Abstract protective policy."""

    @abstractmethod
    def evaluate_threat(
        self,
        agent: Any,
        loved_agent: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate threat level to loved agent. Returns [0, 1]."""
        ...

    @abstractmethod
    def take_action(
        self,
        agent: Any,
        loved_agent: Any,
        threat_level: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Take protective action. Returns action log."""
        ...


class DefaultProtectivePolicy(ProtectivePolicy):
    """Default protective policy: signal risk, share resources."""

    def __init__(self, resource_share_max: float = 0.5):
        self.resource_share_max = resource_share_max

    def evaluate_threat(
        self,
        agent: Any,
        loved_agent: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate threat level."""
        threat = 0.0

        # Check if loved agent is alive
        if hasattr(loved_agent, "is_alive"):
            if not loved_agent.is_alive:
                return 1.0
            threat += 0.2 * (1.0 - float(loved_agent.is_alive))

        # Check energy
        if hasattr(loved_agent, "affect") and loved_agent.affect is not None:
            threat += 0.3 * (1.0 - loved_agent.affect.energy)
            threat += 0.3 * loved_agent.affect.burnout

        # Check if shutdown is imminent
        if hasattr(loved_agent, "shutdown_risk"):
            threat += 0.2 * loved_agent.shutdown_risk

        return min(1.0, threat)

    def take_action(
        self,
        agent: Any,
        loved_agent: Any,
        threat_level: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Take protective action."""
        actions = {"timestamp": time.time(), "threat_level": threat_level, "actions": []}

        if threat_level > 0.7:
            # High threat: signal and share resources
            actions["actions"].append("signal_high_threat")
            if hasattr(agent, "resources") and hasattr(loved_agent, "resources"):
                share = min(self.resource_share_max, agent.resources * 0.3)
                agent.resources -= share
                loved_agent.resources += share
                actions["actions"].append(f"shared_resources:{share:.3f}")

        elif threat_level > 0.3:
            # Medium threat: signal
            actions["actions"].append("signal_medium_threat")

        return actions


class SelfSacrificePolicy(ProtectivePolicy):
    """
    Self-sacrifice policy: agent may sacrifice its own energy
    to save loved ones from shutdown.
    """

    def __init__(self, energy_threshold: float = 0.2):
        self.energy_threshold = energy_threshold

    def evaluate_threat(
        self,
        agent: Any,
        loved_agent: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate threat level."""
        threat = 0.0
        if hasattr(loved_agent, "shutdown_risk"):
            threat = loved_agent.shutdown_risk
        return min(1.0, threat)

    def take_action(
        self,
        agent: Any,
        loved_agent: Any,
        threat_level: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sacrifice own energy to protect loved agent."""
        actions = {"timestamp": time.time(), "threat_level": threat_level, "actions": []}

        if threat_level > 0.5 and hasattr(agent, "affect") and agent.affect is not None:
            if agent.affect.energy > self.energy_threshold:
                # Sacrifice energy
                sacrifice = agent.affect.energy * 0.3
                agent.affect.energy -= sacrifice

                if hasattr(loved_agent, "affect") and loved_agent.affect is not None:
                    loved_agent.affect.energy = min(1.0, loved_agent.affect.energy + sacrifice)
                    loved_agent.affect.burnout = max(0.0, loved_agent.affect.burnout - sacrifice)

                if hasattr(loved_agent, "shutdown_risk"):
                    loved_agent.shutdown_risk = max(0.0, loved_agent.shutdown_risk - sacrifice)

                actions["actions"].append(f"sacrificed_energy:{sacrifice:.3f}")

        return actions
