from __future__ import annotations

import numpy as np
import pytest

from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    HypothesisConfig,
    ModelConfig,
    ObjectState,
    Observation,
    PolicyConfig,
    SearchConfig,
)
from hunter_seeker_v2.hypotheses import (
    Hypothesis,
    RelationalHypothesisEngine,
    propose_completion_motion_reach,
    propose_goal_hypotheses,
    propose_hypotheses,
)
from hunter_seeker_v2.perception import PerceptionSystem


def _config(*, seed: int = 3, hypotheses_enabled: bool = True) -> AgentConfig:
    return AgentConfig(
        seed=seed,
        search=SearchConfig(beam_width=3, horizon=1, max_click_candidates=8),
        policy=PolicyConfig(exploration_epsilon=0.0),
        model=ModelConfig(ensemble_size=3, latent_dim=16),
        hypotheses=HypothesisConfig(enabled=hypotheses_enabled),
    )


def _puzzle_frame(*, fixed: bool) -> np.ndarray:
    """Template block (marked center) and workspace block of equal shape."""

    frame = np.zeros((12, 20), dtype=np.uint8)
    frame[2:5, 2:5] = 4
    frame[3, 3] = 2
    frame[2:5, 12:15] = 4
    if fixed:
        frame[3, 13] = 2
    return frame


def _observation(frame: np.ndarray, *, progress: float = 0.0) -> Observation:
    return Observation(
        frame=frame,
        available_actions=(0, 1),
        task_id="task",
        progress=progress,
    )


def _objects(frame: np.ndarray):
    return PerceptionSystem().observe(frame).objects


def test_proposals_detect_equal_regions_and_respect_masked_cells() -> None:
    frame = _puzzle_frame(fixed=False)
    proposals = propose_hypotheses(frame, _objects(frame), HypothesisConfig())

    equal = [
        h
        for h in proposals
        if h.kind == "equal"
        and {h.region_a, h.region_b} == {(2, 2, 4, 4), (12, 2, 14, 4)}
    ]
    assert equal
    hypothesis = equal[0]
    assert hypothesis.initial_potential == pytest.approx(1.0 / 9.0)
    assert hypothesis.potential(_puzzle_frame(fixed=True)) == 0.0

    # A masked (exogenous) cell inside a region is excluded from mismatch.
    noisy = _puzzle_frame(fixed=True)
    noisy[4, 14] = 9
    assert hypothesis.potential(noisy) == pytest.approx(1.0 / 9.0)
    assert hypothesis.potential(noisy, frozenset({(4, 14)})) == 0.0


def test_scaled_canonical_template_sees_resized_repaletted_copies() -> None:
    # An 8x8 key (2px tiles) in one palette and a 16x16 workspace (4px
    # tiles) in another palette, tile-for-tile the same structure except one
    # wrong tile.
    tiles = np.array(
        [
            [1, 1, 2, 2],
            [1, 2, 2, 1],
            [2, 2, 1, 1],
            [2, 1, 1, 2],
        ],
        dtype=np.uint8,
    )
    workspace_tiles = np.where(tiles == 1, 7, 9).astype(np.uint8)
    workspace_tiles[3, 3] = 7  # wrong: should map from 2 -> 9
    frame = np.zeros((24, 32), dtype=np.uint8)
    frame[2:10, 2:10] = np.kron(tiles, np.ones((2, 2), np.uint8))
    frame[2:18, 14:30] = np.kron(workspace_tiles, np.ones((4, 4), np.uint8))

    hypothesis = Hypothesis(
        hypothesis_id="scaled",
        kind="equal_canonical",
        region_a=(2, 2, 9, 9),
        region_b=(14, 2, 29, 17),
        lattice=(4, 4),
    )
    assert hypothesis.potential(frame) == pytest.approx(1.0 / 16.0)

    fixed = frame.copy()
    fixed[2:18, 14:30] = np.kron(
        np.where(tiles == 1, 7, 9).astype(np.uint8),
        np.ones((4, 4), np.uint8),
    )
    assert hypothesis.potential(fixed) == 0.0

    proposals = propose_hypotheses(
        frame,
        _objects(frame),
        HypothesisConfig(min_region_area=16),
    )
    scaled = [h for h in proposals if h.kind == "equal_canonical"]
    assert scaled
    assert any(h.lattice == (4, 4) for h in scaled)


def test_realized_delta_learning_produces_directional_candidate_signal() -> None:
    engine = RelationalHypothesisEngine(HypothesisConfig())
    broken = _puzzle_frame(fixed=False)
    solved = _puzzle_frame(fixed=True)
    objects = _objects(broken)

    for _ in range(2):
        assert engine.observe_transition(
            before_observation=_observation(broken),
            after_observation=_observation(solved),
            objects=objects,
            action=Action(1),
            progressed=False,
            before_memory_id="",
            after_memory_id="",
        )
        assert engine.observe_transition(
            before_observation=_observation(solved),
            after_observation=_observation(broken),
            objects=objects,
            action=Action(0),
            progressed=False,
            before_memory_id="",
            after_memory_id="",
        )

    agent = CompactHunterSeeker(config=_config(), safe_action_provider=lambda _o: (0, 1))
    agent.hypotheses = RelationalHypothesisEngine.from_state(
        engine.state_dict(),
        config=agent.config.hypotheses,
    )
    snapshot = agent._commit_initial_snapshot(_observation(broken))

    improving = agent.hypotheses.candidate_signal(snapshot, Action(1))
    worsening = agent.hypotheses.candidate_signal(snapshot, Action(0))
    assert improving > 0.0
    assert worsening < 0.0
    assert engine.candidate_signal(snapshot, Action(1)) >= improving  # same state


def test_hypothesis_term_flips_the_selected_action() -> None:
    broken = _puzzle_frame(fixed=False)
    solved = _puzzle_frame(fixed=True)
    observation = _observation(broken)

    decisions = {}
    for enabled in (True, False):
        agent = CompactHunterSeeker(
            config=_config(hypotheses_enabled=enabled),
            safe_action_provider=lambda _o: (0, 1),
        )
        agent.prior.observe_label(task_id="task", action_index=0, weight=4.0)
        seed_engine = RelationalHypothesisEngine(HypothesisConfig())
        objects = _objects(broken)
        for _ in range(3):
            seed_engine.observe_transition(
                before_observation=_observation(broken),
                after_observation=_observation(solved),
                objects=objects,
                action=Action(1),
                progressed=False,
                before_memory_id="",
                after_memory_id="",
            )
        for hypothesis in seed_engine._by_scope[("task", 1)]:
            hypothesis.verified = True
        agent.hypotheses = RelationalHypothesisEngine.from_state(
            seed_engine.state_dict(),
            config=agent.config.hypotheses,
        )
        agent.search_engine = agent._new_search_engine()
        agent.begin_run("task", observation)
        decisions[enabled] = agent.act(observation)

    enabled_terms = {
        candidate.action.index: {term.name: term.value for term in candidate.terms}
        for candidate in decisions[True].candidates
    }
    disabled_terms = {
        candidate.action.index: {term.name: term.value for term in candidate.terms}
        for candidate in decisions[False].candidates
    }
    assert enabled_terms[1]["hypothesis_potential"] > 0.0
    assert "hypothesis_potential" not in disabled_terms[1]
    # Prior pressure wins without the goal signal; the goal signal wins with it.
    assert decisions[False].action.index == 0
    assert decisions[True].action.index == 1


def test_promotion_requires_satisfied_potential_and_contradiction_refutes() -> None:
    engine = RelationalHypothesisEngine(HypothesisConfig())
    scope = ("task", 1)
    satisfied = Hypothesis(
        hypothesis_id="satisfied",
        kind="equal",
        region_a=(2, 2, 4, 4),
        region_b=(12, 2, 14, 4),
    )
    contradicted = Hypothesis(
        hypothesis_id="contradicted",
        kind="uniform",
        region_a=(2, 7, 4, 9),
    )
    engine._by_scope[scope] = [satisfied, contradicted]

    # 'satisfied' reads the solved template/workspace pair (potential 0);
    # 'contradicted' reads a separate scattered block whose modal value
    # covers only 4 of 9 cells, keeping it above the refutation floor.
    frame = _puzzle_frame(fixed=True)
    frame[7:10, 2:5] = 4
    frame[8, 3] = 2
    for y, x in ((7, 2), (7, 4), (9, 2), (9, 4)):
        frame[y, x] = 3
    assert contradicted.potential(frame) >= 0.5
    assert satisfied.potential(frame) == 0.0

    engine.observe_transition(
        before_observation=_observation(frame),
        after_observation=_observation(np.zeros_like(frame), progress=1.0),
        objects=_objects(frame),
        action=Action(1),
        progressed=True,
        before_memory_id="",
        after_memory_id="",
    )

    assert satisfied.verified is True
    assert contradicted.refuted is True
    assert engine.refutations == 1
    active_ids = {h.hypothesis_id for h in engine.active("task", 1)}
    assert "contradicted" not in active_ids


def test_disabled_engine_is_exact_noop_and_state_roundtrips() -> None:
    broken = _puzzle_frame(fixed=False)
    observation = _observation(broken)

    disabled = RelationalHypothesisEngine(HypothesisConfig(enabled=False))
    assert disabled.ensure_proposals(observation, _objects(broken)) == 0
    assert (
        disabled.observe_transition(
            before_observation=observation,
            after_observation=observation,
            objects=_objects(broken),
            action=Action(1),
            progressed=False,
            before_memory_id="",
            after_memory_id="",
        )
        == 0
    )

    engine = RelationalHypothesisEngine(HypothesisConfig())
    engine.ensure_proposals(observation, _objects(broken))
    restored = RelationalHypothesisEngine.from_state(engine.state_dict())
    assert restored.state_dict() == engine.state_dict()
    assert {h.hypothesis_id for h in restored.active("task", 1)} == {
        h.hypothesis_id for h in engine.active("task", 1)
    }


def test_rollout_signal_uses_cached_exact_potentials_along_graph_chains() -> None:
    engine = RelationalHypothesisEngine(HypothesisConfig())
    broken = _puzzle_frame(fixed=False)
    solved = _puzzle_frame(fixed=True)
    objects = _objects(broken)
    engine.observe_transition(
        before_observation=_observation(broken),
        after_observation=_observation(solved),
        objects=objects,
        action=Action(1),
        progressed=False,
        before_memory_id="state-broken",
        after_memory_id="state-solved",
    )

    agent = CompactHunterSeeker(config=_config(), safe_action_provider=lambda _o: (0, 1))
    agent.hypotheses = RelationalHypothesisEngine.from_state(
        engine.state_dict(),
        config=agent.config.hypotheses,
    )
    # from_state drops the volatile cache; rebuild it through one observe.
    agent.hypotheses.observe_transition(
        before_observation=_observation(broken),
        after_observation=_observation(solved),
        objects=objects,
        action=Action(1),
        progressed=False,
        before_memory_id="state-broken",
        after_memory_id="state-solved",
    )
    snapshot = agent._commit_initial_snapshot(_observation(broken))

    toward = agent.hypotheses.rollout_signal(
        snapshot,
        Action(1),
        current_state_id="state-broken",
        successor_state_id="state-solved",
    )
    away = agent.hypotheses.rollout_signal(
        snapshot,
        Action(1),
        current_state_id="state-solved",
        successor_state_id="state-broken",
    )
    assert toward > 0.0
    assert away < 0.0
    # Unknown chain falls back to the learned per-action delta prior.
    assert agent.hypotheses.rollout_signal(
        snapshot,
        Action(1),
        current_state_id=None,
        successor_state_id=None,
    ) >= 0.0


def test_mismatch_points_target_violating_cells_and_lead_click_proposals() -> None:
    from hunter_seeker_v2.search import CandidateGenerator

    broken = _puzzle_frame(fixed=False)
    observation = Observation(
        frame=broken,
        available_actions=(0, 6),
        task_id="task",
    )
    agent = CompactHunterSeeker(
        config=_config(),
        click_action_index=6,
        safe_action_provider=lambda _o: (0,),
    )
    snapshot = agent._commit_initial_snapshot(observation)

    points = agent.hypotheses.mismatch_points(snapshot)
    # The template/workspace pair differs exactly at the marked centers.
    assert (13, 3) in points
    assert (3, 3) in points

    generator = CandidateGenerator(click_action_index=6)
    with_extra = generator.generate(snapshot, extra_click_points=points)
    clicks = [a for a in with_extra if a.has_position]
    # Injected mismatch points lead the click list in their given order.
    assert (clicks[0].x, clicks[0].y) == points[0]
    assert {(a.x, a.y) for a in clicks[: len(points)]} == set(points)

    without_extra = generator.generate(snapshot)
    assert any(a.has_position for a in without_extra)


def test_agent_checkpoint_roundtrips_hypothesis_state(tmp_path) -> None:
    broken = _puzzle_frame(fixed=False)
    observation = _observation(broken)
    agent = CompactHunterSeeker(config=_config(), safe_action_provider=lambda _o: (0, 1))
    agent.begin_run("task", observation)
    assert agent.hypotheses.summary()["hypotheses"] > 0

    path = tmp_path / "hypotheses-roundtrip.json"
    agent.save_checkpoint(str(path))
    restored = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _o: (0, 1),
    )
    restored.load_checkpoint(str(path))

    assert restored.hypotheses.state_dict() == agent.hypotheses.state_dict()
    assert restored.measurement_summary()["hypotheses"] == agent.measurement_summary()[
        "hypotheses"
    ]


def _blob(value: int, top: int, left: int, size: int = 2) -> ObjectState:
    return ObjectState(
        object_id=value,
        track_id=value,
        value=value,
        area=size * size,
        centroid_x=float(left) + (size - 1) / 2.0,
        centroid_y=float(top) + (size - 1) / 2.0,
        bbox=(left, top, left + size - 1, top + size - 1),
    )


def _place(frame: np.ndarray, value: int, top: int, left: int, size: int = 2) -> None:
    frame[top : top + size, left : left + size] = value


def test_reach_potential_measures_object_adjacency() -> None:
    frame = np.zeros((20, 20), dtype=np.int64)
    _place(frame, 5, 2, 2)
    _place(frame, 7, 16, 16)
    far = Hypothesis(hypothesis_id="reach|5|7", kind="reach", region_a=(0, 0, 0, 0), value_a=5, value_b=7)
    assert far.potential(frame) >= 0.3  # distant objects: violated

    frame_adjacent = np.zeros((20, 20), dtype=np.int64)
    _place(frame_adjacent, 5, 2, 2)
    _place(frame_adjacent, 7, 2, 4)  # touches the value-5 block
    assert far.potential(frame_adjacent) <= 0.1  # adjacency: satisfied

    frame_absent = np.zeros((20, 20), dtype=np.int64)
    _place(frame_absent, 5, 2, 2)  # no value-7 cells at all
    assert far.potential(frame_absent) == 1.0


def test_goal_contrast_surfaces_reach_goal_violated_then_satisfied() -> None:
    initial = np.zeros((20, 20), dtype=np.int64)
    _place(initial, 5, 2, 2)
    _place(initial, 7, 16, 16)
    completion = np.zeros((20, 20), dtype=np.int64)
    _place(completion, 5, 2, 2)
    _place(completion, 7, 2, 4)
    objects = (_blob(5, 2, 2), _blob(7, 2, 4))
    config = HypothesisConfig()

    goals = propose_goal_hypotheses(initial, completion, objects, config)
    reach_goals = [g for g in goals if g.kind == "reach"]
    assert reach_goals, "expected a reach goal from violated->satisfied contrast"
    goal = reach_goals[0]
    assert {goal.value_a, goal.value_b} == {5, 7}
    assert goal.verified is True
    assert goal.origin == "goal_contrast"
    assert goal.initial_potential >= config.goal_contrast_floor

    # A relation already satisfied at the start is not a goal.
    assert propose_goal_hypotheses(completion, completion, objects, config) == ()

    # enable_goal_contrast=False is an exact no-op.
    disabled = HypothesisConfig(enable_goal_contrast=False)
    assert propose_goal_hypotheses(initial, completion, objects, disabled) == ()


def test_engine_promotes_reach_goal_on_completion() -> None:
    initial = np.zeros((20, 20), dtype=np.int64)
    _place(initial, 5, 2, 2)
    _place(initial, 7, 16, 16)
    completion = np.zeros((20, 20), dtype=np.int64)
    _place(completion, 5, 2, 2)
    _place(completion, 7, 2, 4)
    objects = (_blob(5, 2, 2), _blob(7, 2, 4))

    engine = RelationalHypothesisEngine(HypothesisConfig())
    before = _observation(initial)  # caches the level's initial frame
    engine.ensure_proposals(before, objects)
    engine.observe_transition(
        before_observation=_observation(completion),
        after_observation=_observation(completion),
        objects=objects,
        action=Action(1),
        progressed=True,
        before_memory_id="s0",
        after_memory_id="s1",
    )
    summary = engine.summary()
    assert summary["reach_goals"] >= 1
    assert summary["goals"] >= 1

    # enabled=False is an exact no-op: no goals, no proposals.
    off = RelationalHypothesisEngine(HypothesisConfig(enabled=False))
    off.ensure_proposals(_observation(initial), objects)
    assert (
        off.observe_transition(
            before_observation=_observation(completion),
            after_observation=_observation(completion),
            objects=objects,
            action=Action(1),
            progressed=True,
            before_memory_id="s0",
            after_memory_id="s1",
        )
        == 0
    )
    assert off.summary()["goals"] == 0


def test_satisfied_reach_has_no_mismatch_and_cap_uses_visible_cells() -> None:
    frame = np.zeros((4, 8), dtype=np.int64)
    frame[1, 1:5] = 5
    frame[1, 5] = 7
    hypothesis = Hypothesis(
        hypothesis_id="reach|5|7",
        kind="reach",
        region_a=(0, 0, 0, 0),
        value_a=5,
        value_b=7,
        reach_cell_cap=1,
    )
    # Three value-5 cells are exogenous.  The cap applies to the one visible
    # value-5 cell, which is adjacent to value 7.
    mask = frozenset({(1, 1), (1, 2), (1, 3)})
    assert hypothesis.potential(frame, mask) == 0.0
    assert hypothesis.mismatch_cells(frame, mask) == []

    far = frame.copy()
    far[1, 5] = 0
    far[3, 7] = 7
    assert hypothesis.potential(far, mask) > 0.0
    assert set(hypothesis.mismatch_cells(far, mask)) == {(4, 1), (7, 3)}


@pytest.mark.parametrize(
    ("kind", "region_b", "expected"),
    [
        ("mirror_h", np.asarray([[9, 1], [4, 3]]), {(1, 0), (4, 0)}),
        ("mirror_v", np.asarray([[9, 4], [1, 2]]), {(0, 1), (4, 0)}),
    ],
)
def test_cross_region_mirror_mismatch_includes_counterpart(
    kind: str,
    region_b: np.ndarray,
    expected: set[tuple[int, int]],
) -> None:
    frame = np.zeros((3, 7), dtype=np.int64)
    frame[0:2, 0:2] = np.asarray([[1, 2], [3, 4]])
    frame[0:2, 4:6] = region_b
    hypothesis = Hypothesis(
        hypothesis_id=kind,
        kind=kind,
        region_a=(0, 0, 1, 1),
        region_b=(4, 0, 5, 1),
    )
    assert set(hypothesis.mismatch_cells(frame)) == expected


def test_fully_masked_relation_is_unknown_and_cannot_be_promoted() -> None:
    frame = _puzzle_frame(fixed=True)
    hypothesis = Hypothesis(
        hypothesis_id="masked-equality",
        kind="equal",
        region_a=(2, 2, 4, 4),
        region_b=(12, 2, 14, 4),
    )
    mask = frozenset(
        {
            (y, x)
            for y in range(2, 5)
            for x in (*range(2, 5), *range(12, 15))
        }
    )
    assert hypothesis.potential(frame, mask) == 0.0
    assert hypothesis.observable(frame, mask) is False

    engine = RelationalHypothesisEngine(HypothesisConfig())
    engine._by_scope[("task", 1)] = [hypothesis]
    engine.observe_transition(
        before_observation=_observation(frame),
        after_observation=_observation(frame, progress=1.0),
        completion_observation=_observation(frame, progress=1.0),
        objects=_objects(frame),
        action=Action(1),
        progressed=True,
        before_memory_id="masked-before",
        after_memory_id="masked-after",
        mask_cells=mask,
    )
    assert hypothesis.verified is False
    assert hypothesis.refuted is False
    assert engine.promotions == 0
    assert engine.refutations == 0


def test_direct_helpers_respect_disabled_and_engine_enforces_scope_cap() -> None:
    initial = np.zeros((20, 20), dtype=np.int64)
    _place(initial, 5, 2, 2)
    _place(initial, 7, 16, 16)
    completion = np.zeros((20, 20), dtype=np.int64)
    _place(completion, 5, 2, 2)
    _place(completion, 7, 2, 4)
    objects = (_blob(5, 2, 2), _blob(7, 2, 4))

    disabled = HypothesisConfig(enabled=False)
    assert propose_hypotheses(initial, objects, disabled) == ()
    assert (
        propose_goal_hypotheses(initial, completion, objects, disabled) == ()
    )

    config = HypothesisConfig(max_hypotheses=1)
    engine = RelationalHypothesisEngine(config)
    engine._initial_frames[("task", 1)] = initial
    engine._by_scope[("task", 1)] = [
        Hypothesis(
            hypothesis_id="weak-invariant",
            kind="uniform",
            region_a=(0, 0, 2, 2),
            initial_potential=0.1,
        )
    ]
    engine.observe_transition(
        before_observation=_observation(completion),
        after_observation=_observation(completion, progress=1.0),
        completion_observation=_observation(completion, progress=1.0),
        objects=objects,
        action=Action(1),
        progressed=True,
        before_memory_id="complete-before",
        after_memory_id="complete-after",
    )
    retained = engine.active("task", 1)
    assert len(retained) == config.max_hypotheses
    assert retained[0].kind == "reach"
    assert retained[0].origin == "goal_contrast"


def _count_frame(value: int, count: int) -> np.ndarray:
    frame = np.zeros((6, 6), dtype=np.int64)
    frame.reshape(-1)[: int(count)] = int(value)
    return frame


@pytest.mark.parametrize(
    ("initial_count", "middle_count", "target_count", "kind"),
    [
        (8, 5, 2, "count_at_most"),
        (2, 5, 8, "count_at_least"),
    ],
)
def test_monotonic_count_goal_is_one_sided(
    initial_count: int,
    middle_count: int,
    target_count: int,
    kind: str,
) -> None:
    value = 11
    initial = _count_frame(value, initial_count)
    middle = _count_frame(value, middle_count)
    completion = _count_frame(value, target_count)
    trace = [
        {0: 36 - initial_count, value: initial_count},
        {0: 36 - middle_count, value: middle_count},
        {0: 36 - target_count, value: target_count},
    ]
    goals = propose_goal_hypotheses(
        initial,
        completion,
        (),
        HypothesisConfig(),
        count_trace=trace,
    )
    count_goals = [goal for goal in goals if goal.kind == kind]
    assert len(count_goals) == 1
    goal = count_goals[0]
    assert goal.initial_count == initial_count
    assert goal.target_count == target_count
    assert goal.potential(initial) == 1.0
    assert goal.potential(completion) == 0.0

    overshoot_count = target_count - 1 if kind == "count_at_most" else target_count + 1
    overshoot = _count_frame(value, overshoot_count)
    assert goal.potential(overshoot) == 0.0
    if kind == "count_at_most":
        assert goal.mismatch_cells(completion) == []
        assert goal.mismatch_cells(overshoot) == []


def test_nonmonotonic_count_trace_is_rejected() -> None:
    value = 11
    initial = _count_frame(value, 8)
    completion = _count_frame(value, 2)
    trace = [
        {0: 28, value: 8},
        {0: 32, value: 4},
        {0: 30, value: 6},
        {0: 34, value: 2},
    ]
    goals = propose_goal_hypotheses(
        initial,
        completion,
        (),
        HypothesisConfig(),
        count_trace=trace,
    )
    assert all(
        goal.kind not in {"count_at_most", "count_at_least"}
        for goal in goals
    )


def test_count_goal_and_trace_persist_exactly() -> None:
    value = 11
    initial = _count_frame(value, 8)
    middle = _count_frame(value, 5)
    completion = _count_frame(value, 2)
    engine = RelationalHypothesisEngine(HypothesisConfig())
    engine.ensure_proposals(_observation(initial), ())
    engine.observe_transition(
        before_observation=_observation(initial),
        after_observation=_observation(middle),
        objects=(),
        action=Action(0),
        progressed=False,
        before_memory_id="count-8",
        after_memory_id="count-5",
    )
    engine.observe_transition(
        before_observation=_observation(middle),
        after_observation=_observation(completion, progress=1.0),
        completion_observation=_observation(completion, progress=1.0),
        objects=(),
        action=Action(0),
        progressed=True,
        before_memory_id="count-5",
        after_memory_id="count-2",
    )
    goals = [
        hypothesis
        for hypothesis in engine.active("task", 1)
        if hypothesis.kind == "count_at_most"
    ]
    assert len(goals) == 1
    assert (goals[0].value_a, goals[0].initial_count, goals[0].target_count) == (
        value,
        8,
        2,
    )

    state = engine.state_dict()
    restored = RelationalHypothesisEngine.from_state(state)
    assert restored.state_dict() == state
    restored_goals = [
        hypothesis
        for hypothesis in restored.active("task", 1)
        if hypothesis.kind == "count_at_most"
    ]
    assert len(restored_goals) == 1
    assert restored_goals[0].potential(_count_frame(value, 1)) == 0.0


def _immediate_reach_scene() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[ObjectState, ...],
]:
    initial = np.zeros((64, 64), dtype=np.uint8)
    initial[16, 17] = 4
    initial[45:48, 45:48] = 14
    predecessor = initial.copy()
    predecessor[16, 17] = 0
    predecessor[40, 47] = 4
    objects = (
        ObjectState(
            object_id=0,
            track_id=61,
            value=4,
            area=1,
            centroid_x=47.0,
            centroid_y=40.0,
            bbox=(47, 40, 47, 40),
            controllable=0.8,
            confidence=1.0,
            signature="avatar",
        ),
        ObjectState(
            object_id=1,
            track_id=64,
            value=14,
            area=9,
            centroid_x=46.0,
            centroid_y=46.0,
            bbox=(45, 45, 47, 47),
            confidence=1.0,
            signature="target",
        ),
    )
    return initial, predecessor, objects


def test_completion_motion_reconstructs_hidden_reach_only_on_direct_overlap() -> None:
    initial, predecessor, objects = _immediate_reach_scene()
    config = HypothesisConfig()

    goals = propose_completion_motion_reach(
        initial,
        predecessor,
        objects,
        config,
        controlled_motion_predictions={61: (0.0, 7.0, 1.0)},
    )
    assert len(goals) == 1
    assert goals[0].hypothesis_id == "reach|4|14"
    assert goals[0].origin == "completion_motion"
    assert goals[0].verified
    assert goals[0].initial_potential == pytest.approx(0.875)
    engine = RelationalHypothesisEngine(config)
    engine._by_scope[("task", 1)] = list(goals)
    restored = RelationalHypothesisEngine.from_state(
        engine.state_dict(),
        config=config,
    )
    assert restored.state_dict() == engine.state_dict()
    assert restored.active("task", 1)[0].origin == "completion_motion"

    # Projected adjacency is not strong enough when the actual solved frame
    # was hidden, and inconsistent motion evidence must also abstain.
    assert propose_completion_motion_reach(
        initial,
        predecessor,
        objects,
        config,
        controlled_motion_predictions={61: (0.0, 4.0, 1.0)},
    ) == ()
    assert propose_completion_motion_reach(
        initial,
        predecessor,
        objects,
        config,
        controlled_motion_predictions={61: (0.0, 7.0, 0.5)},
    ) == ()


def test_completion_motion_reach_abstains_on_moving_or_ambiguous_target() -> None:
    initial, predecessor, objects = _immediate_reach_scene()
    config = HypothesisConfig()

    moved_target = predecessor.copy()
    moved_target[45:48, 45:48] = 0
    moved_target[45:48, 46:49] = 14
    moved_objects = (
        objects[0],
        ObjectState(
            object_id=1,
            track_id=64,
            value=14,
            area=9,
            centroid_x=47.0,
            centroid_y=46.0,
            bbox=(46, 45, 48, 47),
            confidence=1.0,
            signature="target",
        ),
    )
    assert propose_completion_motion_reach(
        initial,
        moved_target,
        moved_objects,
        config,
        controlled_motion_predictions={61: (0.0, 7.0, 1.0)},
    ) == ()

    ambiguous_initial = initial.copy()
    ambiguous_initial[16, 18] = 3
    ambiguous_predecessor = predecessor.copy()
    ambiguous_predecessor[16, 18] = 0
    ambiguous_predecessor[40, 46] = 3
    ambiguous_objects = objects + (
        ObjectState(
            object_id=2,
            track_id=65,
            value=3,
            area=1,
            centroid_x=46.0,
            centroid_y=40.0,
            bbox=(46, 40, 46, 40),
            controllable=0.8,
            confidence=1.0,
            signature="other-avatar",
        ),
    )
    assert propose_completion_motion_reach(
        ambiguous_initial,
        ambiguous_predecessor,
        ambiguous_objects,
        config,
        controlled_motion_predictions={
            61: (0.0, 7.0, 1.0),
            65: (0.0, 7.0, 1.0),
        },
    ) == ()


def test_immediate_stage_swap_never_mines_generic_cross_stage_goals() -> None:
    initial, predecessor, objects = _immediate_reach_scene()
    successor = np.full_like(initial, 8)
    engine = RelationalHypothesisEngine(HypothesisConfig())
    engine.ensure_proposals(_observation(initial), objects)
    engine.observe_transition(
        before_observation=_observation(predecessor),
        after_observation=Observation(
            frame=successor,
            available_actions=(0, 1),
            task_id="task",
            stage=2,
            progress=1.0,
        ),
        objects=objects,
        action=Action(0),
        progressed=True,
        before_memory_id="immediate-before",
        after_memory_id="new-stage",
        completion_frame_available=False,
        controlled_motion_predictions={},
    )
    assert all(
        hypothesis.origin not in {"goal_contrast", "completion_motion"}
        for hypothesis in engine.active("task", 1)
    )
