"""Embedding providers for agent semantic representation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class AgentEmbeddingProvider(ABC):
    """Abstract interface for obtaining embeddings of agents."""

    @abstractmethod
    def get_embedding(self, agent: Any) -> np.ndarray:
        """Return embedding vector for an agent."""
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return dimension of embedding vectors."""
        ...


class InternalEmbeddingProvider(AgentEmbeddingProvider):
    """
    Embedding from agent's internal state:
    - memory vector
    - parameters
    - task descriptions
    - log statistics
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def get_embedding(self, agent: Any) -> np.ndarray:
        """
        Extract embedding from agent's internal state.

        The agent is expected to have:
        - agent.memory_vector: np.ndarray
        - agent.parameters: dict
        - agent.task_history: list
        """
        vec = np.zeros(self.dim)

        # Use memory vector if available
        if hasattr(agent, "memory_vector") and agent.memory_vector is not None:
            mem = np.asarray(agent.memory_vector, dtype=float).flatten()
            n = min(len(mem), self.dim)
            vec[:n] = mem[:n]

        # Use parameter hash
        elif hasattr(agent, "parameters"):
            param_bytes = str(sorted(agent.parameters.items())).encode("utf-8")
            hash_vec = np.frombuffer(param_bytes[: self.dim * 8], dtype=float)
            vec[: len(hash_vec)] = hash_vec

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def get_dimension(self) -> int:
        return self.dim


class LLMEmbeddingProvider(AgentEmbeddingProvider):
    """
    Embedding from external LLM encoder (e.g., sentence-transformers).

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            self._enabled = True
        except ImportError:
            print("Warning: sentence-transformers not installed. LLM embeddings disabled.")
            self.model = None
            self.dim = 384
            self._enabled = False

    def get_embedding(self, agent: Any) -> np.ndarray:
        if not self._enabled:
            # Fallback to random
            return np.random.randn(self.dim) / np.sqrt(self.dim)

        # Build text description of agent
        description = self._build_description(agent)
        embedding = self.model.encode(description)
        return embedding.astype(float)

    def _build_description(self, agent: Any) -> str:
        """Build textual description of agent for LLM encoding."""
        parts = []
        if hasattr(agent, "id"):
            parts.append(f"Agent {agent.id}")
        if hasattr(agent, "task"):
            parts.append(f"Task: {agent.task}")
        if hasattr(agent, "role"):
            parts.append(f"Role: {agent.role}")
        if hasattr(agent, "memory"):
            parts.append(f"Memory: {agent.memory}")
        if hasattr(agent, "goals"):
            parts.append(f"Goals: {agent.goals}")
        return " ".join(parts) if parts else f"Agent_{id(agent)}"

    def get_dimension(self) -> int:
        return self.dim
