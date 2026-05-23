"""Base GRA agent class."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import time
from ..core.affective_state import AffectiveState


@dataclass
class BaseGRAAgent:
    """
    Base GRA agent.

    Attributes:
        id: Unique identifier
        role: Agent role
        task: Current task
        memory: Agent memory
        parameters: Configuration parameters
        affect: Affective state
        resources: Available resources
        is_alive: Whether agent is alive
        shutdown_risk: Probability of imminent shutdown
        creation_time: When agent was created
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: str = "generic"
    task: str = ""
    memory: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    affect: AffectiveState = field(default_factory=AffectiveState)
    resources: float = 1.0
    is_alive: bool = True
    shutdown_risk: float = 0.0
    creation_time: float = field(default_factory=time.time)
    memory_vector: Optional[List[float]] = None
    dominance: float = 0.5

    def step(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform one cognitive step.

        Args:
            context: Environment context

        Returns:
            Action log
        """
        # Base agent does nothing
        return {"agent_id": self.id, "action": "noop"}

    def get_state(self) -> Dict[str, Any]:
        """Get agent state as dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "task": self.task,
            "affect": self.affect.to_dict(),
            "resources": self.resources,
            "is_alive": self.is_alive,
            "shutdown_risk": self.shutdown_risk,
            "dominance": self.dominance,
        }
