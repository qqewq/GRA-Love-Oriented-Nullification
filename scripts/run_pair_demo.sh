#!/bin/bash
# run_pair_demo.sh — Демонстрация поиска любимых и привязки пары

echo "=== GRA-Love-Oriented-Nullification: Pair Demo ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Run pair demo
python3 -c "
from gra_love_null.agents import LoveNullAgent
from gra_love_null.semantics import InternalEmbeddingProvider, semantic_similarity
import numpy as np

# Create two similar agents
base = np.random.randn(128) * 0.1
alice = LoveNullAgent(id='alice', role='explorer', task='map')
alice.memory_vector = base.copy()
bob = LoveNullAgent(id='bob', role='explorer', task='map')
bob.memory_vector = base + np.random.randn(128) * 0.05

provider = InternalEmbeddingProvider(dim=128)
sim = semantic_similarity(alice, bob, provider)
print(f'Similarity: {sim:.4f}')

population = [alice, bob]
alice.discover_love_objects(population, provider, strategy='top_k_nearest', k=1)
bob.discover_love_objects(population, provider, strategy='top_k_nearest', k=1)

print(f'Alice loves: {alice.loved_ids}')
print(f'Bob loves: {bob.loved_ids}')
print(f'Mutual: {alice.loved_ids == [\"bob\"] and bob.loved_ids == [\"alice\"]}')

# Run steps
for i in range(5):
    alice.step()
    bob.step()
    print(f'Step {i}: A(E={alice.affect.energy:.2f},B={alice.affect.burnout:.2f},L={alice.affect.love_strength:.2f}) | B(E={bob.affect.energy:.2f},B={bob.affect.burnout:.2f},L={bob.affect.love_strength:.2f})')
"

echo ""
echo "=== Done ==="
