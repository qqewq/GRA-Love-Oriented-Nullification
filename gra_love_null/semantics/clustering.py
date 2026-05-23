"""Clustering agents by semantic similarity."""

from typing import Any, List, Dict, Tuple
import numpy as np
from .embeddings import AgentEmbeddingProvider


def cluster_agents(
    agents: List[Any],
    provider: AgentEmbeddingProvider,
    n_clusters: int = 5,
) -> Dict[int, List[int]]:
    """
    Cluster agents based on their semantic embeddings using K-Means.

    Args:
        agents: List of agents
        provider: Embedding provider
        n_clusters: Number of clusters

    Returns:
        Dict mapping cluster_id -> list of agent indices
    """
    if len(agents) == 0:
        return {}

    # Get embeddings for all agents
    embeddings = np.array([provider.get_embedding(a) for a in agents])

    # Simple K-Means implementation
    n = len(embeddings)
    n_clusters = min(n_clusters, n)

    # Random initialization
    np.random.seed(42)
    indices = np.random.choice(n, n_clusters, replace=False)
    centroids = embeddings[indices].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(100):  # Max iterations
        # Assign to nearest centroid
        new_labels = np.zeros(n, dtype=int)
        for i in range(n):
            distances = np.linalg.norm(embeddings[i] - centroids, axis=1)
            new_labels[i] = np.argmin(distances)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = new_labels == k
            if mask.sum() > 0:
                new_centroids[k] = embeddings[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        # Check convergence
        if np.array_equal(labels, new_labels):
            break

        labels = new_labels
        centroids = new_centroids

    # Build result dictionary
    clusters: Dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    return clusters


def find_cluster_centroids(
    agents: List[Any],
    clusters: Dict[int, List[int]],
    provider: AgentEmbeddingProvider,
) -> Dict[int, int]:
    """
    Find centroid agents for each cluster (agent closest to cluster mean).

    Args:
        agents: List of agents
        clusters: Cluster assignments
        provider: Embedding provider

    Returns:
        Dict mapping cluster_id -> index of centroid agent
    """
    centroids = {}
    for cluster_id, indices in clusters.items():
        if not indices:
            continue
        embeddings = np.array([provider.get_embedding(agents[i]) for i in indices])
        mean_emb = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - mean_emb, axis=1)
        closest_idx = indices[int(np.argmin(distances))]
        centroids[cluster_id] = closest_idx
    return centroids
