"""Tests for semantic similarity module."""

import pytest
import numpy as np
from gra_love_null.semantics.embeddings import InternalEmbeddingProvider
from gra_love_null.semantics.similarity import (
    semantic_similarity,
    structural_similarity,
    combined_similarity,
)
from gra_love_null.agents.base_agent import BaseGRAAgent


class TestSemanticSimilarity:
    """Test semantic similarity functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = InternalEmbeddingProvider(dim=128)

        # Create test agents
        self.agent_a = BaseGRAAgent(id="a")
        self.agent_a.memory_vector = np.ones(128) / np.sqrt(128)

        self.agent_b = BaseGRAAgent(id="b")
        self.agent_b.memory_vector = np.ones(128) / np.sqrt(128)

        self.agent_c = BaseGRAAgent(id="c")
        self.agent_c.memory_vector = -np.ones(128) / np.sqrt(128)

    def test_identical_agents_similarity_1(self):
        """Identical agents should have similarity ~1."""
        sim = semantic_similarity(self.agent_a, self.agent_b, self.provider)
        assert abs(sim - 1.0) < 0.01, f"Expected ~1.0, got {sim}"

    def test_opposite_agents_similarity_minus_1(self):
        """Opposite agents should have similarity ~-1."""
        sim = semantic_similarity(self.agent_a, self.agent_c, self.provider)
        assert abs(sim - (-1.0)) < 0.01, f"Expected ~-1.0, got {sim}"

    def test_self_similarity_1(self):
        """Agent with itself should have similarity 1."""
        sim = semantic_similarity(self.agent_a, self.agent_a, self.provider)
        assert abs(sim - 1.0) < 0.01

    def test_similarity_range(self):
        """Similarity should be in [-1, 1]."""
        for _ in range(100):
            agent_x = BaseGRAAgent()
            agent_x.memory_vector = np.random.randn(128)
            agent_y = BaseGRAAgent()
            agent_y.memory_vector = np.random.randn(128)
            sim = semantic_similarity(agent_x, agent_y, self.provider)
            assert -1.0 <= sim <= 1.0, f"Similarity {sim} out of range"

    def test_combined_similarity(self):
        """Test combined similarity."""
        sim = combined_similarity(
            self.agent_a, self.agent_b, self.provider, alpha=0.7
        )
        assert 0.0 <= sim <= 1.0
