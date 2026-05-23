# Туториал: GRA-Love-Oriented-Nullification

## Установка

```bash
git clone https://github.com/qqewq/GRA-Love-Oriented-Nullification.git
cd GRA-Love-Oriented-Nullification
pip install -e .
Глава 1: Поиск любимых по семантическому сродству
python
from gra_love_null.agents import LoveNullAgent
from gra_love_null.semantics import InternalEmbeddingProvider, semantic_similarity
from gra_love_null.love_selection import select_love_objects

# Создаём популяцию агентов
agents = [
    LoveNullAgent(id=f"agent_{i}", role="worker", task=f"task_{i % 5}")
    for i in range(10)
]

# Задаём различные memory_vector для разнообразия
import numpy as np
for i, agent in enumerate(agents):
    agent.memory_vector = np.random.randn(128) * 0.1
    agent.memory_vector[i % 10] += 1.0  # Делаем агентов разными

# Создаём embedding provider
provider = InternalEmbeddingProvider(dim=128)

# Агент 0 ищет любимых
agent_0 = agents[0]
loved = agent_0.discover_love_objects(
    population=agents,
    provider=provider,
    strategy="top_k_nearest",
)

print(f"Агент {agent_0.id} полюбил:")
for loved_agent, sim in loved:
    print(f"  - {loved_agent.id} (similarity: {sim:.3f})")
Глава 2: Любовь как цель обнулёнки
python
from gra_love_null.love_goals import LoveGoal
from gra_love_null.core.nullification import LoveNullification, nullify_with_love

# Создаём love goal
love_goal = LoveGoal(
    love_object_ids=agent_0.loved_ids,
    weights=[1.0 / len(agent_0.loved_ids)] * len(agent_0.loved_ids),
    protect_from_shutdown=True,
    maintain_vitality=True,
)

# Создаём nullifier с сохранением любви
nullifier = LoveNullification(preserve_love=True)

# Симулируем высокую пену
foam = 5.0
agent_0 = nullify_with_love(agent_0, foam, nullifier, love_goal, agent_0.loved_ids)

print(f"Love goal survived nullification: {agent_0.love_goal is not None}")
print(f"Nullification survival count: {agent_0.love_goal.nullification_survival_count}")
Глава 3: Защита роя
python
from gra_love_null.swarm import PopulationDynamics
from gra_love_null.swarm.love_network import LoveNetwork

# Создаём популяцию
pop = PopulationDynamics(agents)

# Запускаем симуляцию на 100 шагов
history = []
for _ in range(100):
    summary = pop.step(dt=0.1)
    history.append(summary)

# Визуализируем love network
love_net = LoveNetwork()
love_net.build_from_agents(pop.agents)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
love_net.visualize(ax=ax)
plt.show()

# Статистика
stats = love_net.get_stability_metrics()
print("Love network stability:")
for k, v in stats.items():
    print(f"  {k}: {v}")
Глава 4: Визуализация матрицы семантических отношений
python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Вычисляем матрицу сходства
n = len(agents)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = semantic_similarity(agents[i], agents[j], provider)

# Визуализируем
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=[a.id for a in agents],
            yticklabels=[a.id for a in agents],
            ax=ax)
ax.set_title("Semantic Similarity Matrix")
plt.tight_layout()
plt.show()
