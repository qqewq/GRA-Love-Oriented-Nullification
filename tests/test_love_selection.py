"""Tests for love selection strategies."""

import pytest
import numpy as np
from gra_love_null.semantics.embeddings import InternalEmbeddingProvider
from gra_love_null.love_selection.strategies import (
    select_by_top_k_nearest,
    select_love_objects,
)
from gra_love_null.love_selection.constraints import LoveConstraints, apply_constraints
from gra_love_null.agents.love_null_agent import LoveNullAgent


class TestLoveSelection:
    """Test love selection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = InternalEmbeddingProvider(dim=128)

        # Create population
        self.agents = []
        for i in range(20):
            agent = LoveNullAgent(id=f"agent_{i}")
            agent.memory_vector = np.random.randn(128) * 0.1
            agent.memory_vector[i % 5] += 1.0  # 5 groups
            self.agents.append(agent)

    def test_top_k_nearest(self):
        """Test top-k nearest selection."""
        agent = self.agents[0]
        candidates = [a for a in self.agents if a.id != agent.id]

        loved = select_by_top_k_nearest(
            agent, candidates, self.provider, k=3, min_threshold=0.0
        )

        assert len(loved) == 3
        # Similarities should be sorted descending
        sims = [s for _, s in loved]
        assert sims == sorted(sims, reverse=True)

    def test_select_love_objects(self):
        """Test select_love_objects with different strategies."""
        agent = self.agents[0]
        candidates = [a for a in self.agents if a.id != agent.id]

        # Top-k
        loved = select_love_objects(
            agent, candidates, self.provider, strategy="top_k_nearest", k=5
        )
        assert len(loved) == 5

        # Mutual (none initially since no love goals set)
        loved_mutual = select_love_objects(
            agent, candidates, self.provider, strategy="mutual_love", k=5
        )
        assert len(loved_mutual) == 0  # No mutual love yet

    def test_constraints_max_love_objects(self):
        """Test max love objects constraint."""
        agent = self.agents[0]
        candidates = [(self.agents[i], 0.9 - i * 0.01) for i in range(1, 11)]

        constraints = LoveConstraints(max_love_objects=3, min_similarity_threshold=0.0)
        filtered = apply_constraints(candidates, constraints, agent_id=agent.id)

        assert len(filtered) == 3

    def test_constraints_forbid_self_love(self):
        """Test self-love is forbidden."""
        agent = self.agents[0]
        candidates = [(agent, 1.0)]  # Self-love

        constraints = LoveConstraints(forbid_self_love=True)
        filtered = apply_constraints(candidates, constraints, agent_id=agent.id)

        assert len(filtered) == 0
