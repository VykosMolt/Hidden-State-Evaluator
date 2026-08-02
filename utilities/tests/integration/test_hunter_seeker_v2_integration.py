from __future__ import annotations

import numpy as np

from hunter_seeker_v2.adapters import (
    MockActionAdapter,
    MockObservationAdapter,
    MockOutcomeAdapter,
)
from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    AgentConfig,
    BoundaryKind,
    ModelConfig,
    PolicyConfig,
    SearchConfig,
)
from hunter_seeker_v2.run_arc import run_episode


class OddGridEnvironment:
    def __init__(self) -> None:
        self.step_count = 0
        self.action_count = 0
        self.grid = np.zeros((5, 9), dtype=np.uint8)
        self.grid[2, 1] = 3

    def _observation(self):
        return {
            "grid": self.grid.copy(),
            "available_actions": [0, 1, 2, 3],
            "progress": float(self.step_count >= 3),
            "state": "GAME_COMPLETED" if self.step_count >= 3 else "RUNNING",
        }

    def reset(self):
        return self._observation()

    def step(self, action: int):
        self.action_count += 1
        self.step_count += 1
        old_y, old_x = np.argwhere(self.grid == 3)[0]
        self.grid[old_y, old_x] = 0
        if action == 0:
            new_y, new_x = max(0, old_y - 1), old_x
        elif action == 1:
            new_y, new_x = min(self.grid.shape[0] - 1, old_y + 1), old_x
        elif action == 2:
            new_y, new_x = old_y, max(0, old_x - 1)
        else:
            new_y, new_x = old_y, min(self.grid.shape[1] - 1, old_x + 1)
        self.grid[new_y, new_x] = 3
        return self._observation()


def test_real_compact_agent_runs_transactionally_in_clickless_odd_grid_domain(
    tmp_path,
) -> None:
    observation_adapter = MockObservationAdapter()
    action_adapter = MockActionAdapter()
    agent = CompactHunterSeeker(
        config=AgentConfig(
            seed=11,
            search=SearchConfig(beam_width=3, horizon=2),
            policy=PolicyConfig(exploration_epsilon=0.0),
            model=ModelConfig(ensemble_size=3, latent_dim=16),
        ),
        click_action_index=action_adapter.click_action_index(),
        safe_action_provider=action_adapter.safe_action_indices,
    )
    environment = OddGridEnvironment()

    result = run_episode(
        environment,
        agent,
        task_id="odd-grid",
        observation_adapter=observation_adapter,
        action_adapter=action_adapter,
        outcome_adapter=MockOutcomeAdapter(),
        action_space=None,
        max_steps=8,
        bootstrap=False,
    )

    assert result.steps == 3
    assert environment.action_count == 3
    assert agent.transition_count == 3
    assert len(agent.buffer) == 3
    assert result.final_outcome.boundary is BoundaryKind.GAME_COMPLETED
    assert result.transitions[-1].outcome.terminated is True
    assert result.transitions[-1].outcome.completed is True
    assert all(0 <= row.action.index < 4 for row in result.transitions)
    assert all(not row.action.has_position for row in result.transitions)
    assert result.final_observation.frame.shape == (5, 9)

    checkpoint = tmp_path / "odd-grid-compact.json"
    agent.save_checkpoint(str(checkpoint))
    assert checkpoint.exists()
    assert agent.measurement_summary()["transition_count"] == 3
