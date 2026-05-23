"""Semantic similarity metrics between agents."""

from typing import Any, Optional, Callable
import numpy as np
from .embeddings import AgentEmbeddingProvider


def semantic_similarity(
    agent_a: Any,
    agent_b: Any,
    provider: AgentEmbeddingProvider,
) -> float:
    """
    Compute cosine similarity between agent embeddings.

    Args:
        agent_a: First agent
        agent_b: Second agent
        provider: Embedding provider

    Returns:
        float: Cosine similarity in [-1, 1]
    """
    emb_a = provider.get_embedding(agent_a)
    emb_b = provider.get_embedding(agent_b)

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def structural_similarity(
    agent_a: Any,
    agent_b: Any,
    task_graph: Optional[Any] = None,
) -> float:
    """
    Compute structural similarity based on task graph proximity.

    Args:
        agent_a: First agent
        agent_b: Second agent
        task_graph: NetworkX graph of task relationships

    Returns:
        float: Structural similarity score in [0, 1]
    """
    if task_graph is None:
        return 0.5  # Neutral if no graph available

    id_a = agent_a.id if hasattr(agent_a, "id") else id(agent_a)
    id_b = agent_b.id if hasattr(agent_b, "id") else id(agent_b)

    if id_a not in task_graph or id_b not in task_graph:
        return 0.0

    try:
        import networkx as nx
        # Jaccard similarity of neighbors
        neighbors_a = set(task_graph.neighbors(id_a))
        neighbors_b = set(task_graph.neighbors(id_b))
        intersection = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)
        if union == 0:
            return 0.0
        return intersection / union
    except ImportError:
        return 0.5


def combined_similarity(
    agent_a: Any,
    agent_b: Any,
    provider: AgentEmbeddingProvider,
    task_graph: Optional[Any] = None,
    alpha: float = 0.7,
) -> float:
    """
    Combined similarity score: α * semantic + (1-α) * structural.

    Args:
        agent_a: First agent
        agent_b: Second agent
        provider: Embedding provider
        task_graph: Optional task graph
        alpha: Weight for semantic similarity (default: 0.7)

    Returns:
        float: Combined similarity in [0, 1]
    """
    sem_sim = max(0.0, semantic_similarity(agent_a, agent_b, provider))
    struct_sim = structural_similarity(agent_a, agent_b, task_graph)
    return alpha * sem_sim + (1 - alpha) * struct_sim
