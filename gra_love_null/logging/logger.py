"""Logging utilities for love-oriented nullification."""

import time
import json
import os
from typing import Any, Dict, List, Optional


class LoveLogger:
    """
    Logger for love-oriented nullification events.

    Records:
    - Love discovery events
    - Heartbeat exchanges
    - Nullification cycles
    - Burnout events
    - Protective actions
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logs: Dict[str, List[Dict[str, Any]]] = {
            "love_discovery": [],
            "heartbeats": [],
            "nullifications": [],
            "burnout": [],
            "protective_actions": [],
            "shutdowns": [],
        }

    def log_love_discovery(self, agent_id: str, loved_ids: List[str], similarities: List[float]) -> None:
        """Log love discovery event."""
        self.logs["love_discovery"].append({
            "timestamp": time.time(),
            "agent_id": agent_id,
            "loved_ids": loved_ids,
            "similarities": similarities,
        })

    def log_heartbeat(self, sender_id: str, receiver_id: str, alive: bool) -> None:
        """Log heartbeat event."""
        self.logs["heartbeats"].append({
            "timestamp": time.time(),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "alive": alive,
        })

    def log_nullification(self, agent_id: str, foam: float, love_preserved: bool) -> None:
        """Log nullification event."""
        self.logs["nullifications"].append({
            "timestamp": time.time(),
            "agent_id": agent_id,
            "foam": foam,
            "love_preserved": love_preserved,
        })

    def log_burnout(self, agent_id: str, burnout_risk: float, energy: float) -> None:
        """Log burnout event."""
        self.logs["burnout"].append({
            "timestamp": time.time(),
            "agent_id": agent_id,
            "burnout_risk": burnout_risk,
            "energy": energy,
        })

    def log_protective_action(self, agent_id: str, loved_id: str, action: str) -> None:
        """Log protective action."""
        self.logs["protective_actions"].append({
            "timestamp": time.time(),
            "agent_id": agent_id,
            "loved_id": loved_id,
            "action": action,
        })

    def log_shutdown(self, agent_id: str, reason: str) -> None:
        """Log agent shutdown."""
        self.logs["shutdowns"].append({
            "timestamp": time.time(),
            "agent_id": agent_id,
            "reason": reason,
        })

    def save(self, filename: Optional[str] = None) -> None:
        """Save logs to JSON file."""
        if filename is None:
            filename = f"love_log_{int(time.time())}.json"
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "n_love_discoveries": len(self.logs["love_discovery"]),
            "n_heartbeats": len(self.logs["heartbeats"]),
            "n_nullifications": len(self.logs["nullifications"]),
            "n_burnout_events": len(self.logs["burnout"]),
            "n_protective_actions": len(self.logs["protective_actions"]),
            "n_shutdowns": len(self.logs["shutdowns"]),
        }
Now the documentation, paper, examples, scripts, and tests:
