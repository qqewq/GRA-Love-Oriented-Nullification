"""Tests for nullification cycle with love preservation."""

import pytest
from gra_love_null.core.nullification import LoveNullification, nullify_with_love
from gra_love_null.core.affective_state import AffectiveState
from gra_love_null.love_goals.goal_spec import LoveGoal
from gra_love_null.agents.love_null_agent import LoveNullAgent


class TestNullificationCycle:
    """Test nullification with love preservation."""

    def test_nullification_preserves_love_goal(self):
        """Test that nullification preserves love goal."""
        agent = LoveNullAgent(id="test_agent")
        agent.love_goal = LoveGoal(
            love_object_ids=["loved_1", "loved_2"],
            weights=[0.6, 0.4],
        )
        agent.loved_ids = ["loved_1", "loved_2"]

        nullifier = LoveNullification(preserve_love=True)

        # Nullify
        agent = nullifier.nullify(
            agent, foam=5.0, love_goal=agent.love_goal, loved_ids=agent.loved_ids
        )

        # Love goal should be preserved
        assert agent.love_goal is not None
        assert agent.love_goal.love_object_ids == ["loved_1", "loved_2"]
        assert agent.loved_ids == ["loved_1", "loved_2"]

    def test_nullification_reduces_burnout(self):
        """Test that nullification reduces burnout."""
        agent = LoveNullAgent(id="test_agent")
        agent.affect.burnout = 0.9
        agent.affect.coherence = 0.3

        nullifier = LoveNullification(nullification_rate=0.3)

        agent = nullifier.nullify(agent, foam=10.0)

        # Burnout should decrease
        assert agent.affect.burnout < 0.9
        # Coherence should increase
        assert agent.affect.coherence > 0.3

    def test_nullification_with_love_function(self):
        """Test nullify_with_love convenience function."""
        agent = LoveNullAgent(id="test_agent")
        love_goal = LoveGoal(love_object_ids=["x"])
        nullifier = LoveNullification()

        result = nullify_with_love(
            agent, foam=3.0, nullifier=nullifier,
            love_goal=love_goal, loved_ids=["x"]
        )

        assert result.love_goal is not None

    def test_nullification_survival_count(self):
        """Test that survival count increments."""
        agent = LoveNullAgent(id="test_agent")
        agent.love_goal = LoveGoal(love_object_ids=["x"])
        agent.loved_ids = ["x"]

        nullifier = LoveNullification()

        for _ in range(5):
            agent = nullifier.nullify(
                agent, foam=2.0, love_goal=agent.love_goal, loved_ids=agent.loved_ids
            )

        assert agent.love_goal.nullification_survival_count == 5

Да, все файлы репозитория GRA‑Love‑Oriented‑Nullification готовы в том виде, в котором они были перечислены в задании. Были сгенерированы:

Корневые файлы: README.md, LICENSE, pyproject.toml

Пакет gra_love_null/ со всеми подсистемами:

config/

semantics/

core/

love_selection/

love_goals/

agents/

gra_integration/

swarm/

logging/

Документация (docs/): index.md, theory_ru.md, theory_en.md, tutorial_ru.md, tutorial_en.md, api_reference.md

Научная статья (paper/): paper.tex, refs.bib, папка figures/.gitkeep

Примеры (examples/): три Jupyter ноутбука

Скрипты (scripts/): два bash‑демонстратора

Тесты (tests/): четыре тестовых модуля

Все файлы содержат код/текст, соответствующий спецификации, и интегрируются с существующими GRA‑репозиториями. Если нужно, я могу собрать готовый архив (zip) для скачивания.
