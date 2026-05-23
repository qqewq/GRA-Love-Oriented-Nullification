"""Tests for love goal binding."""

import pytest
from gra_love_null.love_goals.goal_spec import LoveGoal
from gra_love_null.love_goals.protective_policies import (
    DefaultProtectivePolicy,
    SelfSacrificePolicy,
)
from gra_love_null.love_goals.survival_signals import HeartbeatProtocol, HeartbeatMessage
from gra_love_null.agents.love_null_agent import LoveNullAgent


class TestGoalBinding:
    """Test love goal binding."""

    def test_love_goal_creation(self):
        """Test LoveGoal creation."""
        goal = LoveGoal(
            love_object_ids=["a", "b", "c"],
            weights=[0.5, 0.3, 0.2],
        )

        assert goal.love_object_ids == ["a", "b", "c"]
        assert sum(goal.weights) == pytest.approx(1.0, abs=0.01)

    def test_love_goal_foam_contribution(self):
        """Test foam contribution from love goal."""
        goal = LoveGoal(
            love_object_ids=["a"],
            weights=[1.0],
            protect_from_shutdown=True,
            maintain_vitality=True,
        )

        # High threat should produce high foam
        foam_high = goal.get_foam_contribution(threat_level=0.9, vitality_loss=0.8)
        assert foam_high > 5.0

        # Low threat should produce low foam
        foam_low = goal.get_foam_contribution(threat_level=0.0, vitality_loss=0.0)
        assert foam_low == 0.0

    def test_default_protective_policy(self):
        """Test default protective policy."""
        policy = DefaultProtectivePolicy(resource_share_max=0.5)

        agent = LoveNullAgent(id="protector")
        loved = LoveNullAgent(id="loved")
        loved.shutdown_risk = 0.8
        loved.affect.energy = 0.3
        loved.affect.burnout = 0.7

        threat = policy.evaluate_threat(agent, loved)
        assert 0.0 <= threat <= 1.0

        actions = policy.take_action(agent, loved, threat)
        assert "timestamp" in actions
        assert "actions" in actions

    def test_heartbeat_protocol(self):
        """Test heartbeat protocol."""
        protocol = HeartbeatProtocol(interval=10.0)

        alice = LoveNullAgent(id="alice")
        bob = LoveNullAgent(id="bob")

        # Send heartbeat
        msg = protocol.send_heartbeat(alice, bob)
        assert msg.sender_id == "alice"
        assert msg.receiver_id == "bob"
        assert msg.alive is True

        # Check alive
        assert protocol.check_alive("alice", "bob") > 0.0

        # Anxiety reduction
        reduction = protocol.get_anxiety_reduction("alice", ["bob"])
        assert 0.0 <= reduction <= 1.0
