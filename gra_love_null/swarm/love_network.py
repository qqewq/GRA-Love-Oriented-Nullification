"""Love network: graph of who loves whom."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np


@dataclass
class LoveEdge:
    """Edge in the love network."""

    source_id: str
    target_id: str
    weight: float = 1.0
    is_mutual: bool = False
    love_type: str = "primary"  # "primary", "secondary", "cluster_centroid"


class LoveNetwork:
    """
    Graph representation of love relationships between agents.

    Nodes: agents
    Edges: love relationships (directed, weighted)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.agent_data: Dict[str, Any] = {}

    def add_agent(self, agent: Any) -> None:
        """Add agent to network."""
        node_id = agent.id if hasattr(agent, "id") else str(id(agent))
        self.graph.add_node(node_id)
        self.agent_data[node_id] = {
            "role": getattr(agent, "role", "unknown"),
            "energy": agent.affect.energy if hasattr(agent, "affect") and agent.affect is not None else 1.0,
            "burnout": agent.affect.burnout if hasattr(agent, "affect") and agent.affect is not None else 0.0,
            "is_alive": getattr(agent, "is_alive", True),
        }

    def add_love_edge(
        self,
        source_agent: Any,
        target_agent: Any,
        weight: float = 1.0,
        love_type: str = "primary",
    ) -> None:
        """Add a love edge from source to target."""
        source_id = source_agent.id if hasattr(source_agent, "id") else str(id(source_agent))
        target_id = target_agent.id if hasattr(target_agent, "id") else str(id(target_agent))

        # Check if mutual
        is_mutual = False
        if self.graph.has_edge(target_id, source_id):
            is_mutual = True
            self.graph.edges[target_id, source_id]["is_mutual"] = True

        self.graph.add_edge(
            source_id,
            target_id,
            weight=weight,
            is_mutual=is_mutual,
            love_type=love_type,
        )

    def build_from_agents(self, agents: List[Any]) -> "LoveNetwork":
        """Build love network from list of agents."""
        # Add all agents
        for agent in agents:
            self.add_agent(agent)

        # Add love edges
        for agent in agents:
            source_id = agent.id if hasattr(agent, "id") else str(id(agent))
            if hasattr(agent, "loved_ids"):
                for loved_id in agent.loved_ids:
                    if loved_id in self.agent_data:
                        self.add_love_edge(agent, {"id": loved_id})

            if hasattr(agent, "loved_agents"):
                for loved in agent.loved_agents:
                    self.add_love_edge(agent, loved)

        return self

    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Compute stability metrics of the love network.

        Returns:
            Dict with:
            - modularity: How clustered the love network is
            - reciprocity: Fraction of mutual loves
            - density: Network density
            - avg_clustering: Average clustering coefficient
        """
        n = self.graph.number_of_nodes()
        e = self.graph.number_of_edges()

        if n == 0:
            return {
                "modularity": 0.0,
                "reciprocity": 0.0,
                "density": 0.0,
                "avg_clustering": 0.0,
            }

        # Density
        max_edges = n * (n - 1)  # Directed
        density = e / max_edges if max_edges > 0 else 0.0

        # Reciprocity
        mutual_count = sum(
            1 for u, v in self.graph.edges()
            if self.graph.has_edge(v, u)
        )
        reciprocity = mutual_count / e if e > 0 else 0.0

        # Average clustering (undirected version)
        undirected = self.graph.to_undirected()
        avg_clustering = nx.average_clustering(undirected) if n > 1 else 0.0

        return {
            "density": float(density),
            "reciprocity": float(reciprocity),
            "avg_clustering": float(avg_clustering),
            "n_nodes": n,
            "n_edges": e,
            "mutual_edges": mutual_count,
        }

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get weighted adjacency matrix."""
        return nx.to_numpy_array(self.graph, weight="weight")

    def visualize(self, ax=None):
        """Visualize the love network."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        pos = nx.spring_layout(self.graph, seed=42)

        # Node colors based on energy
        node_colors = [
            self.agent_data.get(node, {}).get("energy", 0.5)
            for node in self.graph.nodes()
        ]

        # Edge colors: red for mutual, gray for one-way
        edge_colors = [
            "red" if data.get("is_mutual", False) else "gray"
            for _, _, data in self.graph.edges(data=True)
        ]

        # Edge widths based on weight
        edge_widths = [
            data.get("weight", 1.0) * 2
            for _, _, data in self.graph.edges(data=True)
        ]

        nx.draw_networkx(
            self.graph,
            pos=pos,
            ax=ax,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            edge_color=edge_colors,
            width=edge_widths,
            with_labels=True,
            node_size=500,
            font_size=8,
        )

        ax.set_title("Love Network")
        return ax
