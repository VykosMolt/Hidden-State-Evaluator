from __future__ import annotations

import copy

import numpy as np
import pytest

from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    ExogenousConfig,
    ModelConfig,
    Observation,
    Outcome,
    PolicyConfig,
    SearchConfig,
)
from hunter_seeker_v2.exogenous import (
    EXOGENOUS_STATE_VERSION,
    ExogenousChangeFilter,
)


def _config(*, seed: int = 3, exogenous_enabled: bool = True) -> AgentConfig:
    return AgentConfig(
        seed=seed,
        search=SearchConfig(beam_width=3, horizon=1, max_click_candidates=8),
        policy=PolicyConfig(exploration_epsilon=0.0),
        model=ModelConfig(ensemble_size=3, latent_dim=16),
        exogenous=ExogenousConfig(enabled=exogenous_enabled, min_common_horizon=4),
    )


def _frame(avatar_x: int, tick: int) -> np.ndarray:
    """5x9 frame: avatar cell (endogenous) plus a one-cell step counter."""

    frame = np.zeros((5, 9), dtype=np.uint8)
    frame[2, int(avatar_x)] = 3
    frame[4, 0] = tick % 97 + 1
    return frame


def _observation(avatar_x: int, tick: int, *, task: str = "task") -> Observation:
    return Observation(
        frame=_frame(avatar_x, tick),
        available_actions=(0, 1),
        task_id=task,
    )


def _run_episode(
    filter_: ExogenousChangeFilter,
    actions: list[int],
    *,
    task: str = "task",
) -> None:
    filter_.begin_episode(task)
    avatar_x = 3
    for step, action_index in enumerate(actions):
        next_x = int(np.clip(avatar_x + (1 if action_index else -1), 0, 8))
        filter_.observe_transition(
            task_id=task,
            before_frame=_frame(avatar_x, step),
            after_frame=_frame(next_x, step + 1),
            action=Action(action_index),
        )
        avatar_x = next_x


def test_mask_forms_only_under_differing_action_histories() -> None:
    ticker_cell = (4, 0)

    intervened = ExogenousChangeFilter(ExogenousConfig(min_common_horizon=4))
    _run_episode(intervened, [1, 1, 1, 1, 0, 0, 0, 0])
    _run_episode(intervened, [0, 0, 0, 0, 1, 1, 1, 1])
    intervened.begin_episode("task")

    mask = intervened.mask_cells("task", (5, 9))
    assert ticker_cell in mask
    # The avatar's cells are action-coupled and must never be masked.
    assert all(cell[0] != 2 for cell in mask)

    replayed = ExogenousChangeFilter(ExogenousConfig(min_common_horizon=4))
    _run_episode(replayed, [1, 1, 1, 1, 0, 0, 0, 0])
    _run_episode(replayed, [1, 1, 1, 1, 0, 0, 0, 0])
    replayed.begin_episode("task")

    # Identical action histories are no intervention: no evidence, no mask.
    assert replayed.mask_cells("task", (5, 9)) == frozenset()


def test_masked_state_id_merges_tick_variants_only() -> None:
    filter_ = ExogenousChangeFilter(ExogenousConfig(min_common_horizon=4))
    _run_episode(filter_, [1, 1, 1, 1, 0, 0, 0, 0])
    _run_episode(filter_, [0, 0, 0, 0, 1, 1, 1, 1])
    filter_.begin_episode("task")

    same_avatar_tick_2 = _observation(5, 2)
    same_avatar_tick_6 = _observation(5, 6)
    other_avatar_tick_2 = _observation(6, 2)

    merged = filter_.masked_state_id(same_avatar_tick_2)
    assert merged
    assert merged == filter_.masked_state_id(same_avatar_tick_6)
    assert merged != filter_.masked_state_id(other_avatar_tick_2)
    # Exact transactional identity still distinguishes tick variants.
    assert same_avatar_tick_2.state_id != same_avatar_tick_6.state_id

    disabled = ExogenousChangeFilter(ExogenousConfig(enabled=False))
    assert disabled.masked_state_id(same_avatar_tick_2) == ""


def test_contradiction_unmasks_conservatively() -> None:
    filter_ = ExogenousChangeFilter(ExogenousConfig(min_common_horizon=4))
    _run_episode(filter_, [1, 1, 1, 1, 0, 0, 0, 0])
    _run_episode(filter_, [0, 0, 0, 0, 1, 1, 1, 1])
    filter_.begin_episode("task")
    assert (4, 0) in filter_.mask_cells("task", (5, 9))

    # An episode where the ticker behaves differently contradicts the mask.
    filter_.begin_episode("task")
    avatar_x = 3
    for step, action_index in enumerate([0, 1, 0, 1, 1, 0, 0, 1]):
        next_x = int(np.clip(avatar_x + (1 if action_index else -1), 0, 8))
        before = _frame(avatar_x, step)
        after = _frame(next_x, step + 1)
        if step % 2:
            after = after.copy()
            after[4, 0] = 0  # ticker stalls: different event trajectory
        filter_.observe_transition(
            task_id="task",
            before_frame=before,
            after_frame=after,
            action=Action(action_index),
        )
        avatar_x = next_x
    filter_.begin_episode("task")

    assert (4, 0) not in filter_.mask_cells("task", (5, 9))


def test_agent_graph_reuses_states_across_tick_variants() -> None:
    # The scripted environment ignores the agent's action, so the avatar path
    # is exact by construction; random exploration (epsilon=1) makes the two
    # recorded action histories differ, which is the required intervention.
    def _drive_episode(agent: CompactHunterSeeker, xs: list[int]) -> None:
        observation = _observation(xs[0], 0)
        agent.begin_run("task", observation)
        for tick, avatar_x in enumerate(xs[1:], start=1):
            decision = agent.act(observation)
            next_observation = _observation(avatar_x, tick)
            agent.observe(decision, next_observation, Outcome())
            observation = next_observation
        agent.end_run()

    # No (avatar, tick) pair repeats anywhere, so exact identity never
    # collides; avatar positions repeat at different ticks, so masked
    # identity does.
    paths = [
        [3, 4, 5, 6, 5, 6, 5, 6, 5],
        [1, 0, 1, 0, 1, 0, 1, 0, 1],
        [6, 5, 6, 5, 4, 5, 4, 5, 4],
    ]
    results = {}
    for enabled in (True, False):
        config = AgentConfig(
            seed=3,
            search=SearchConfig(beam_width=3, horizon=1, max_click_candidates=8),
            policy=PolicyConfig(exploration_epsilon=1.0),
            model=ModelConfig(ensemble_size=3, latent_dim=16),
            exogenous=ExogenousConfig(
                enabled=enabled,
                min_common_horizon=4,
            ),
        )
        agent = CompactHunterSeeker(
            config=config,
            safe_action_provider=lambda _obs: (0, 1),
        )
        for path in paths:
            _drive_episode(agent, path)
        results[enabled] = agent

    # With the filter, episode-3 states merge with earlier visits despite the
    # ticker; without it every ticked state is brand new forever.
    assert results[True].graph.revisit_count > 0
    assert results[False].graph.revisit_count == 0
    assert len(results[True].graph) < len(results[False].graph)

    summary = results[True].measurement_summary()
    assert summary["exogenous"]["masked_cells"] >= 1
    assert summary["graph_revisits"] == results[True].graph.revisit_count


def test_fuse_estimate_and_time_pressure_brake_exploration() -> None:
    from hunter_seeker_v2.contracts import CompetenceState
    from hunter_seeker_v2.policy import exploration_scale

    filter_ = ExogenousChangeFilter(ExogenousConfig(min_common_horizon=4))
    _run_episode(filter_, [1, 1, 1, 1, 0, 0, 0, 0])
    _run_episode(filter_, [0, 0, 0, 0, 1, 1, 1, 1])
    filter_.begin_episode("task")

    assert (4, 0) in filter_.mask_cells("task", (5, 9))
    # The ticker changed on every one of the 8 recorded steps.
    assert filter_.fuse_estimate("task", (5, 9)) == 8
    assert filter_.time_pressure("task", (5, 9), 0) == 0.0
    assert filter_.time_pressure("task", (5, 9), 4) == pytest.approx(0.5)
    assert filter_.time_pressure("task", (5, 9), 12) == 1.0
    # No mask, no pressure.
    assert filter_.time_pressure("other-task", (5, 9), 4) == 0.0

    relaxed = exploration_scale(
        CompetenceState(expected_learning_gain=1.0, remaining_risk_budget=1.0)
    )
    urgent = exploration_scale(
        CompetenceState(
            expected_learning_gain=1.0,
            remaining_risk_budget=1.0,
            time_pressure=1.0,
        )
    )
    assert urgent == pytest.approx(relaxed * 0.4)


def test_filter_state_survives_checkpoint_roundtrip(tmp_path) -> None:
    agent = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _obs: (0, 1),
    )
    seeded = ExogenousChangeFilter(ExogenousConfig(min_common_horizon=4))
    _run_episode(seeded, [1, 1, 1, 1, 0, 0, 0, 0])
    _run_episode(seeded, [0, 0, 0, 0, 1, 1, 1, 1])
    seeded.begin_episode("task")
    agent.exogenous = ExogenousChangeFilter.from_state(
        seeded.state_dict(),
        config=agent.config.exogenous,
    )

    observation = _observation(3, 0)
    agent.begin_run("task", observation)
    decision = agent.act(observation)
    agent.observe(decision, _observation(4, 1), Outcome())

    path = tmp_path / "exogenous-roundtrip.json"
    agent.save_checkpoint(str(path))
    restored = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _obs: (0, 1),
    )
    restored.load_checkpoint(str(path))

    assert restored.exogenous.state_dict() == agent.exogenous.state_dict()
    assert restored.exogenous.mask_cells("task", (5, 9)) == agent.exogenous.mask_cells(
        "task",
        (5, 9),
    )
    resumed = restored.act(_observation(4, 1))
    assert resumed.action == restored._chosen_candidate(resumed).action


def _run_stage_clock(
    filter_: ExogenousChangeFilter,
    *,
    stage: int,
    actions: tuple[int, ...],
) -> None:
    filter_.begin_episode("task", stage=stage)
    for step, action_index in enumerate(actions):
        before = np.zeros((2, 3), dtype=np.int64)
        after = np.zeros((2, 3), dtype=np.int64)
        before[0, 1] = step
        after[0, 1] = step + 1
        filter_.observe_transition(
            task_id="task",
            stage=stage,
            after_stage=stage,
            before_frame=before,
            after_frame=after,
            action=Action(action_index),
        )
    assert filter_.end_episode() is True


def test_masks_are_isolated_by_task_stage_and_shape() -> None:
    filter_ = ExogenousChangeFilter(
        ExogenousConfig(min_common_horizon=2, min_confirmations=1)
    )
    _run_stage_clock(filter_, stage=1, actions=(0, 0))
    _run_stage_clock(filter_, stage=1, actions=(1, 1))

    assert filter_.mask_cells("task", (2, 3), stage=1) == frozenset({(0, 1)})
    assert filter_.mask_cells("task", (2, 3), stage=2) == frozenset()

    stage_one_a = Observation(
        np.asarray([[0, 2, 0], [0, 0, 0]]),
        (0,),
        "task",
        stage=1,
    )
    stage_one_b = Observation(
        np.asarray([[0, 9, 0], [0, 0, 0]]),
        (0,),
        "task",
        stage=1,
    )
    assert filter_.masked_state_id(stage_one_a)
    assert (
        filter_.masked_state_id(stage_one_a)
        == filter_.masked_state_id(stage_one_b)
    )

    stage_two_a = Observation(stage_one_a.frame, (0,), "task", stage=2)
    stage_two_b = Observation(stage_one_b.frame, (0,), "task", stage=2)
    assert filter_.masked_state_id(stage_two_a) == ""
    assert filter_.masked_state_id(stage_two_b) == ""
    assert stage_two_a.state_id != stage_two_b.state_id


def test_cross_stage_transition_finalizes_old_scope_and_is_not_recorded() -> None:
    filter_ = ExogenousChangeFilter(
        ExogenousConfig(min_common_horizon=1, min_confirmations=1)
    )
    filter_.begin_episode("task", stage=1)
    filter_.observe_transition(
        task_id="task",
        stage=1,
        before_frame=np.zeros((2, 3), dtype=np.int64),
        after_frame=np.asarray([[0, 1, 0], [0, 0, 0]], dtype=np.int64),
        action=Action(0),
    )
    # The boundary image differs everywhere, but none of it belongs to a
    # comparable within-stage trajectory.
    assert (
        filter_.observe_transition(
            task_id="task",
            stage=1,
            after_stage=2,
            before_frame=np.zeros((2, 3), dtype=np.int64),
            after_frame=np.full((2, 3), 9, dtype=np.int64),
            action=Action(1),
        )
        == 0
    )
    boundary_state = filter_.state_dict()
    assert boundary_state["finalized_episodes"] == 1
    assert boundary_state["active"] == {
        "task_id": "task",
        "stage": 2,
        "shape": None,
        "steps": 0,
        "actions": [],
        "events": [],
    }

    filter_.observe_transition(
        task_id="task",
        stage=2,
        before_frame=np.zeros((2, 3), dtype=np.int64),
        after_frame=np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.int64),
        action=Action(0),
    )
    assert filter_.end_episode() is True
    scopes = {
        row["stage"]: row for row in filter_.state_dict()["scopes"]
    }
    assert set(scopes) == {1, 2}
    assert scopes[2]["previous_events"] == [
        {"cell": [1, 1], "events": [[0, 1]]}
    ]


def test_v2_state_is_structured_and_v1_masks_are_quarantined() -> None:
    filter_ = ExogenousChangeFilter(
        ExogenousConfig(min_common_horizon=2, min_confirmations=1)
    )
    _run_stage_clock(filter_, stage=3, actions=(0, 0))
    _run_stage_clock(filter_, stage=3, actions=(1, 1))
    filter_.begin_episode("task", stage=4)
    filter_.observe_transition(
        task_id="task",
        stage=4,
        before_frame=np.zeros((2, 3), dtype=np.int64),
        after_frame=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
        action=Action(0),
    )

    state = filter_.state_dict()
    assert state["version"] == EXOGENOUS_STATE_VERSION
    assert isinstance(state["scopes"], list)
    assert state["scopes"][0]["task_id"] == "task"
    assert state["scopes"][0]["stage"] == 3
    assert state["scopes"][0]["shape"] == [2, 3]
    assert state["active"]["stage"] == 4
    restored = ExogenousChangeFilter.from_state(
        state,
        config=filter_.config,
    )
    assert restored.state_dict() == state
    assert restored.mask_cells("task", (2, 3), stage=3) == frozenset({(0, 1)})
    assert restored.mask_cells("task", (2, 3), stage=4) == frozenset()

    legacy = {
        "stats": {
            "task␟2x3": {
                "masked": ["0,1"],
                "confirmations": {"0,1": 5},
            }
        },
        "active": {"task_id": "task", "shape_key": "2x3", "steps": 1},
        "finalized_episodes": 7,
    }
    migrated = ExogenousChangeFilter.from_state(
        legacy,
        config=filter_.config,
    )
    assert migrated.mask_cells("task", (2, 3), stage=1) == frozenset()
    assert migrated.mask_cells("task", (2, 3), stage=3) == frozenset()
    summary = migrated.summary()
    assert summary["quarantined_legacy_scopes"] == 1
    assert summary["quarantined_legacy_active"] is True
    assert migrated.state_dict()["active"] is None


def test_v2_import_rejects_coerced_and_structurally_invalid_evidence() -> None:
    config = ExogenousConfig(min_common_horizon=2, min_confirmations=1)
    filter_ = ExogenousChangeFilter(config)
    _run_stage_clock(filter_, stage=3, actions=(0, 0))
    _run_stage_clock(filter_, stage=3, actions=(1, 1))
    valid = filter_.state_dict()
    cases: list[tuple[str, dict]] = []

    malformed = copy.deepcopy(valid)
    malformed["version"] = 2.0
    cases.append(("float version", malformed))

    malformed = copy.deepcopy(valid)
    malformed["version"] = None
    cases.append(("null version", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["stage"] = 0
    cases.append(("clamped stage", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["shape"][0] = 2.0
    cases.append(("float shape", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["task_id"] = 3
    cases.append(("coerced task id", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["previous_steps"] = -1
    cases.append(("negative previous steps", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["previous_actions"][0][0] = 0.5
    cases.append(("fractional action", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["has_previous"] = "true"
    cases.append(("string boolean", malformed))

    malformed = copy.deepcopy(valid)
    scope = malformed["scopes"][0]
    scope["previous_events"][0]["events"][0][0] = scope["previous_steps"]
    cases.append(("out-of-horizon event", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["previous_events"][0]["events"][0][1] = float(
        "nan"
    )
    cases.append(("nonfinite event value", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["confirmations"][0]["count"] = -1
    cases.append(("negative confirmation", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["episodes"] = 0
    cases.append(("zero episodes", malformed))

    malformed = copy.deepcopy(valid)
    malformed["scopes"][0]["masked"].append(
        list(malformed["scopes"][0]["masked"][0])
    )
    cases.append(("duplicate masked cell", malformed))

    malformed = copy.deepcopy(valid)
    malformed["finalized_episodes"] = 1
    cases.append(("inconsistent finalized count", malformed))

    malformed = copy.deepcopy(valid)
    malformed["legacy_quarantine"]["active"] = 0
    cases.append(("integer boolean", malformed))

    malformed = copy.deepcopy(valid)
    malformed["active"] = {
        "task_id": "task",
        "stage": 3,
        "shape": None,
        "steps": 1,
        "actions": [[0, -1, -1]],
        "events": [],
    }
    cases.append(("nonempty shapeless active state", malformed))

    for label, malformed in cases:
        try:
            ExogenousChangeFilter.from_state(
                malformed,
                config=config,
            )
        except ValueError:
            continue
        pytest.fail(f"malformed exogenous state was accepted: {label}")


def test_legacy_import_rejects_negative_counters_and_bad_containers() -> None:
    with pytest.raises(ValueError, match="finalized_episodes"):
        ExogenousChangeFilter.from_state(
            {
                "stats": {},
                "active": None,
                "finalized_episodes": -1,
            }
        )
    with pytest.raises(ValueError, match="stats"):
        ExogenousChangeFilter.from_state(
            {
                "stats": [],
                "active": None,
                "finalized_episodes": 0,
            }
        )
