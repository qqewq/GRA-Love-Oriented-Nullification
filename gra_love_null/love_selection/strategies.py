"""Strategies for selecting love objects."""

from typing import Any, List, Optional, Tuple
import numpy as np
from ..semantics.embeddings import AgentEmbeddingProvider
from ..semantics.similarity import semantic_similarity, combined_similarity
from ..semantics.clustering import cluster_agents, find_cluster_centroids


def select_by_top_k_nearest(
    agent: Any,
    candidates: List[Any],
    provider: AgentEmbeddingProvider,
    k: int = 3,
    min_threshold: float = 0.5,
    task_graph: Optional[Any] = None,
) -> List[Tuple[Any, float]]:
    """
    Select k nearest agents by semantic similarity.

    Args:
        agent: The selecting agent
        candidates: List of candidate agents
        provider: Embedding provider
        k: Number of agents to select
        min_threshold: Minimum similarity threshold
        task_graph: Optional task graph for combined similarity

    Returns:
        List of (agent, similarity) tuples, sorted by similarity descending
    """
    if not candidates:
        return []

    similarities = []
    for candidate in candidates:
        if candidate is agent or candidate.id == getattr(agent, "id", None):
            continue
        sim = combined_similarity(agent, candidate, provider, task_graph)
        if sim >= min_threshold:
            similarities.append((candidate, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def select_by_mutual_love(
    agent: Any,
    candidates: List[Any],
    provider: AgentEmbeddingProvider,
    k: int = 3,
    min_threshold: float = 0.6,
) -> List[Tuple[Any, float]]:
    """
    Select agents that also select this agent (mutual love).

    Args:
        agent: The selecting agent
        candidates: List of candidate agents
        provider: Embedding provider
        k: Number of agents to select
        min_threshold: Minimum similarity threshold

    Returns:
        List of (agent, similarity) tuples
    """
    # First, compute similarities
    similarities = {}
    for candidate in candidates:
        if candidate is agent or candidate.id == getattr(agent, "id", None):
            continue
        sim = semantic_similarity(agent, candidate, provider)
        if sim >= min_threshold:
            similarities[candidate.id if hasattr(candidate, "id") else id(candidate)] = (candidate, sim)

    # Check which candidates also love this agent
    mutual = []
    agent_id = agent.id if hasattr(agent, "id") else id(agent)
    for cand_id, (candidate, sim) in similarities.items():
        cand_loves_agent = False
        if hasattr(candidate, "loved_ids") and agent_id in candidate.loved_ids:
            cand_loves_agent = True
        elif hasattr(candidate, "love_goal") and candidate.love_goal is not None:
            if agent_id in candidate.love_goal.love_object_ids:
                cand_loves_agent = True

        if cand_loves_agent:
            mutual.append((candidate, sim))

    mutual.sort(key=lambda x: x[1], reverse=True)
    return mutual[:k]


def select_by_cluster_centroid(
    agent: Any,
    candidates: List[Any],
    provider: AgentEmbeddingProvider,
    n_clusters: int = 5,
    min_threshold: float = 0.4,
) -> List[Tuple[Any, float]]:
    """
    Select cluster centroids as love objects ("elder siblings").

    Args:
        agent: The selecting agent
        candidates: List of candidate agents
        provider: Embedding provider
        n_clusters: Number of clusters
        min_threshold: Minimum similarity threshold

    Returns:
        List of (agent, similarity) tuples
    """
    all_agents = list(candidates) + [agent]
    clusters = cluster_agents(all_agents, provider, n_clusters)
    centroids = find_cluster_centroids(all_agents, clusters, provider)

    results = []
    for cluster_id, centroid_idx in centroids.items():
        centroid = all_agents[centroid_idx]
        if centroid is agent or centroid.id == getattr(agent, "id", None):
            continue
        sim = semantic_similarity(agent, centroid, provider)
        if sim >= min_threshold:
            results.append((centroid, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def select_love_objects(
    agent: Any,
    candidates: List[Any],
    provider: AgentEmbeddingProvider,
    strategy: str = "top_k_nearest",
    k: int = 3,
    min_threshold: float = 0.5,
    task_graph: Optional[Any] = None,
    n_clusters: int = 5,
) -> List[Tuple[Any, float]]:
    """
    Select love objects using specified strategy.

    Args:
        agent: The selecting agent
        candidates: List of candidate agents
        provider: Embedding provider
        strategy: "top_k_nearest", "mutual_love", or "cluster_centroid"
        k: Number of agents to select
        min_threshold: Minimum similarity threshold
        task_graph: Optional task graph
        n_clusters: Number of clusters (for cluster_centroid)

    Returns:
        List of (agent, similarity) tuples
    """
    if strategy == "top_k_nearest":
        return select_by_top_k_nearest(agent, candidates, provider, k, min_threshold, task_graph)
    elif strategy == "mutual_love":
        return select_by_mutual_love(agent, candidates, provider, k, min_threshold)
    elif strategy == "cluster_centroid":
        return select_by_cluster_centroid(agent, candidates, provider, n_clusters, min_threshold)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
