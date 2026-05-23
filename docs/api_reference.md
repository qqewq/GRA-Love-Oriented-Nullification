```markdown
# API Reference

## Core

### `AffectiveState`

```python
@dataclass
class AffectiveState:
    energy: float = 1.0      # [0, 1]
    burnout: float = 0.0     # [0, 1]
    meaning: float = 0.5     # [0, 1]
    longing: float = 0.0     # [0, 1]
    joy: float = 0.5         # [0, 1]
    love_strength: float = 0.0  # [0, 1]
    coherence: float = 1.0   # [0, 1]
LoveNullification
python
class LoveNullification:
    def __init__(self, foam_threshold=1e-6, nullification_rate=0.1, preserve_love=True)
    def nullify(self, agent, foam, love_goal=None, loved_ids=None) -> agent
Semantics
AgentEmbeddingProvider
python
class AgentEmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding(self, agent) -> np.ndarray
    @abstractmethod
    def get_dimension(self) -> int
semantic_similarity
python
def semantic_similarity(agent_a, agent_b, provider) -> float  # [-1, 1]
Love Selection
select_love_objects
python
def select_love_objects(
    agent, candidates, provider,
    strategy="top_k_nearest", k=3, min_threshold=0.5,
    task_graph=None, n_clusters=5
) -> List[Tuple[agent, float]]
Love Goals
LoveGoal
python
@dataclass
class LoveGoal:
    love_object_ids: List[str]
    weights: List[float]
    protect_from_shutdown: bool = True
    maintain_vitality: bool = True
    creation_timestamp: float = 0.0
    nullification_survival_count: int = 0
Agents
LoveNullAgent
python
class LoveNullAgent(BaseGRAAgent):
    def discover_love_objects(self, population, provider=None, strategy=None)
    def step(self, context=None, population=None) -> dict
Swarm
LoveNetwork
python
class LoveNetwork:
    def build_from_agents(self, agents) -> LoveNetwork
    def get_stability_metrics(self) -> dict
    def get_adjacency_matrix(self) -> np.ndarray
    def visualize(self, ax=None) -> ax
