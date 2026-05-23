#!/bin/bash
# run_swarm_protection.sh — Защита роя через любовь

echo "=== GRA-Love-Oriented-Nullification: Swarm Protection ==="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

python3 -c "
from gra_love_null.agents import LoveNullAgent
from gra_love_null.swarm import PopulationDynamics
import numpy as np

# Create swarm
n = 30
agents = [LoveNullAgent(id=f's{i}', role='swarm') for i in range(n)]
for i, a in enumerate(agents):
    a.memory_vector = np.random.randn(128) * 0.1
    a.memory_vector[i % 5] += 1.0

pop = PopulationDynamics(agents)

# Run
for step in range(100):
    s = pop.step(dt=0.1)
    if step % 20 == 0:
        print(f'Step {step}: alive={s[\"n_alive\"]}, energy={s[\"avg_energy\"]:.3f}, burnout={s[\"avg_burnout\"]:.3f}, love={s[\"avg_love_strength\"]:.3f}')

    # Crisis at step 50
    if step == 50:
        print('--- CRISIS INJECTED ---')
        for a in pop.agents:
            if a.is_alive:
                a.affect.energy *= 0.3
                a.shutdown_risk = 0.7

print(f'Final: alive={sum(1 for a in pop.agents if a.is_alive)}/{n}')
stats = pop.get_love_network_stats()
print(f'Love connections: {stats[\"n_connections\"]}, mutual: {stats[\"mutual_loves\"]}')
"

echo ""
echo "=== Done ==="
