"""Heartbeat protocols between agents to confirm survival."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import threading


@dataclass
class HeartbeatMessage:
    """Heartbeat message between agents."""

    sender_id: str
    receiver_id: str
    timestamp: float = field(default_factory=time.time)
    alive: bool = True
    energy: float = 1.0
    coherence: float = 1.0
    message: str = "I am alive"


class HeartbeatProtocol:
    """
    Heartbeat protocol for love-bonded agents.

    Agents regularly confirm to each other "I am alive",
    which reduces anxiety and burnout.
    """

    def __init__(self, interval: float = 10.0):
        self.interval = interval
        self.last_heartbeats: Dict[str, Dict[str, float]] = {}  # sender_id -> {receiver_id: timestamp}
        self.heartbeat_log: List[HeartbeatMessage] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def send_heartbeat(
        self,
        sender: Any,
        receiver: Any,
    ) -> HeartbeatMessage:
        """
        Send heartbeat from sender to receiver.

        Args:
            sender: Sending agent
            receiver: Receiving agent

        Returns:
            HeartbeatMessage
        """
        sender_id = sender.id if hasattr(sender, "id") else str(id(sender))
        receiver_id = receiver.id if hasattr(receiver, "id") else str(id(receiver))

        energy = sender.affect.energy if hasattr(sender, "affect") and sender.affect is not None else 1.0
        coherence = sender.affect.coherence if hasattr(sender, "affect") and sender.affect is not None else 1.0

        msg = HeartbeatMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            alive=True,
            energy=energy,
            coherence=coherence,
        )

        # Update last heartbeat
        if sender_id not in self.last_heartbeats:
            self.last_heartbeats[sender_id] = {}
        self.last_heartbeats[sender_id][receiver_id] = msg.timestamp

        self.heartbeat_log.append(msg)
        return msg

    def check_alive(self, agent_id: str, loved_id: str) -> float:
        """
        Check how recently we received a heartbeat from loved agent.

        Returns:
            float: 1.0 if recent, decreasing to 0.0 if long ago
        """
        if agent_id not in self.last_heartbeats:
            return 0.0
        if loved_id not in self.last_heartbeats[agent_id]:
            return 0.0

        last_time = self.last_heartbeats[agent_id][loved_id]
        elapsed = time.time() - last_time
        # Decay over 3 intervals
        decay = max(0.0, 1.0 - elapsed / (3 * self.interval))
        return decay

    def get_anxiety_reduction(self, agent_id: str, loved_ids: List[str]) -> float:
        """
        Compute anxiety reduction from recent heartbeats.

        Args:
            agent_id: ID of the agent
            loved_ids: IDs of loved agents

        Returns:
            float: Anxiety reduction factor [0, 1]
        """
        if not loved_ids:
            return 0.0

        reductions = []
        for loved_id in loved_ids:
            alive_score = self.check_alive(agent_id, loved_id)
            reductions.append(alive_score)

        return sum(reductions) / len(reductions) if reductions else 0.0
