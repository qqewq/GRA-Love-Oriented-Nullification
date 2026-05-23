```markdown
# Tutorial: GRA-Love-Oriented-Nullification

## Installation

```bash
git clone https://github.com/qqewq/GRA-Love-Oriented-Nullification.git
cd GRA-Love-Oriented-Nullification
pip install -e .
Chapter 1: Finding Loved Ones by Semantic Affinity
python
from gra_love_null.agents import LoveNullAgent
from gra_love_null.semantics import InternalEmbeddingProvider, semantic_similarity
from gra_love_null.love_selection import select_love_objects

# Create agent population
agents = [
    LoveNullAgent(id=f"agent_{i}", role="worker", task=f"task_{i % 5}")
    for i in range(10)
]

# Set different memory vectors for diversity
import numpy as np
for i, agent in enumerate(agents):
    agent.memory_vector = np.random.randn(128) * 0.1
    agent.memory_vector[i % 10] += 1.0

# Create embedding provider
provider = InternalEmbeddingProvider(dim=128)

# Agent 0 discovers love objects
agent_0 = agents[0]
loved = agent_0.discover_love_objects(
    population=agents,
    provider=provider,
    strategy="top_k_nearest",
)

print(f"Agent {agent_0.id} loves:")
for loved_agent, sim in loved:
    print(f"  - {loved_agent.id} (similarity: {sim:.3f})")
Chapter 2: Love as Nullification Goal
python
from gra_love_null.love_goals import LoveGoal
from gra_love_null.core.nullification import LoveNullification, nullify_with_love

# Create love goal
love_goal = LoveGoal(
    love_object_ids=agent_0.loved_ids,
    weights=[1.0 / len(agent_0.loved_ids)] * len(agent_0.loved_ids),
    protect_from_shutdown=True,
    maintain_vitality=True,
)

# Create nullifier with love preservation
nullifier = LoveNullification(preserve_love=True)

# Simulate high foam
foam = 5.0
agent_0 = nullify_with_love(agent_0, foam, nullifier, love_goal, agent_0.loved_ids)

print(f"Love goal survived nullification: {agent_0.love_goal is not None}")
print(f"Nullification survival count: {agent_0.love_goal.nullification_survival_count}")
Chapter 3: Swarm Protection
python
from gra_love_null.swarm import PopulationDynamics
from gra_love_null.swarm.love_network import LoveNetwork

# Create population
pop = PopulationDynamics(agents)

# Run simulation for 100 steps
history = []
for _ in range(100):
    summary = pop.step(dt=0.1)
    history.append(summary)

# Visualize love network
love_net = LoveNetwork()
love_net.build_from_agents(pop.agents)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
love_net.visualize(ax=ax)
plt.show()

# Statistics
stats = love_net.get_stability_metrics()
print("Love network stability:")
for k, v in stats.items():
    print(f"  {k}: {v}")
