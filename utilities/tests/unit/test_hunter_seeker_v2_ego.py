from __future__ import annotations

import copy

import numpy as np
import pytest

from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    EgoConfig,
    EventKind,
    ModelConfig,
    ObjectState,
    Observation,
    Outcome,
    PolicyConfig,
    SearchConfig,
    WorldEvent,
)
from hunter_seeker_v2.ego import ControlAttribution
from hunter_seeker_v2.perception import connected_components


def _config(*, seed: int = 3, ego_enabled: bool = True) -> AgentConfig:
    return AgentConfig(
        seed=seed,
        search=SearchConfig(beam_width=3, horizon=1, max_click_candidates=8),
        policy=PolicyConfig(
            exploration_epsilon=0.0,
            risk_limit=0.65,
            ego_hazard_weight=1.0,
        ),
        model=ModelConfig(ensemble_size=3, latent_dim=16),
        ego=EgoConfig(enabled=ego_enabled),
    )


def _object(track_id: int, *, signature: str, x: float = 3.0, y: float = 2.0) -> ObjectState:
    return ObjectState(
        object_id=0,
        track_id=track_id,
        value=3,
        area=1,
        centroid_x=x,
        centroid_y=y,
        bbox=(int(x), int(y), int(x), int(y)),
        signature=signature,
    )


def _moved(track_id: int, signature: str, dx: float, dy: float = 0.0) -> WorldEvent:
    return WorldEvent(
        EventKind.MOVED,
        subject_track_id=track_id,
        object_signature=signature,
        magnitude=abs(dx) + abs(dy),
        metadata={"dx": dx, "dy": dy},
    )


def _seed_directional_control(
    model: ControlAttribution,
    *,
    track_id: int,
    signature: str,
    repeats: int = 6,
) -> None:
    subject = _object(track_id, signature=signature)
    for _ in range(repeats):
        model.observe(
            action=Action(1),
            events=(_moved(track_id, signature, 2.0),),
            visible_objects=(subject,),
        )
        model.observe(
            action=Action(0),
            events=(_moved(track_id, signature, -2.0),),
            visible_objects=(subject,),
        )


def _hazard_frame() -> tuple[np.ndarray, str, str]:
    """Ego candidate at (2, 3); two-cell hazard object at (2, 6..7)."""

    frame = np.zeros((5, 9), dtype=np.uint8)
    frame[2, 3] = 3
    frame[2, 6:8] = 4
    _bg, components = connected_components(frame)
    by_value = {component.value: component for component in components}
    return frame, by_value[3].signature, by_value[4].signature


def test_control_attribution_learns_action_coupled_track_not_drifter_or_static() -> None:
    model = ControlAttribution(EgoConfig())
    controlled = _object(1, signature="sig-controlled")
    drifter = _object(2, signature="sig-drifter", x=6.0)
    static = _object(3, signature="sig-static", x=1.0)
    directions = {0: (-2.0, 0.0), 1: (2.0, 0.0), 2: (0.0, -2.0), 3: (0.0, 2.0)}
    for action_index, (dx, dy) in directions.items():
        for _ in range(3):
            model.observe(
                action=Action(action_index),
                events=(
                    _moved(1, "sig-controlled", dx, dy),
                    _moved(2, "sig-drifter", 2.0),
                ),
                visible_objects=(controlled, drifter, static),
            )

    assert model.update_count == 12
    assert model.influence(1, "sig-controlled") >= 0.35
    assert model.influence(2, "sig-drifter") == 0.0
    assert model.influence(3, "sig-static") == 0.0

    ranked = model.controlled_objects((controlled, drifter, static))
    assert tuple(obj.track_id for obj in ranked) == (1,)

    enriched = model.enrich((controlled, drifter))
    assert enriched[0].controllable >= 0.35
    assert enriched[1].controllable == 0.0

    # Positional actions are excluded from directional attribution.
    before = model.update_count
    assert (
        model.observe(
            action=Action(5, x=1, y=1),
            events=(_moved(1, "sig-controlled", 2.0),),
            visible_objects=(controlled,),
        )
        == 0
    )
    assert model.update_count == before


def test_ego_motion_hazard_term_changes_the_selected_action() -> None:
    frame, ego_signature, hazard_signature = _hazard_frame()
    observation = Observation(
        frame=frame,
        available_actions=(0, 1),
        task_id="task",
    )

    decisions = {}
    for enabled in (True, False):
        agent = CompactHunterSeeker(
            config=_config(ego_enabled=enabled),
            safe_action_provider=lambda _obs: (0, 1),
        )
        for _ in range(5):
            agent.affordances.observe_signature(
                hazard_signature,
                task_id="task",
                hazard=1.0,
                terminal=1.0,
            )
        agent.prior.observe_label(task_id="task", action_index=1, weight=4.0)
        seed_model = ControlAttribution(EgoConfig())
        _seed_directional_control(seed_model, track_id=999, signature=ego_signature)
        agent.ego = ControlAttribution.from_state(
            seed_model.state_dict(),
            config=agent.config.ego,
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

    # The risk schema is stable across candidates; only the hazard-ward action
    # in the enabled arm carries a nonzero value.
    assert enabled_terms[1]["ego_motion_hazard"] < 0.0
    assert enabled_terms[0]["ego_motion_hazard"] == 0.0
    assert disabled_terms[1]["ego_motion_hazard"] == 0.0

    # The component acts through the named term plus belief enrichment; the
    # enrichment route only perturbs shared terms negligibly, so the named
    # term is the decisive cause.
    shared_shift = 0.0
    for name, value in enabled_terms[1].items():
        if name in {"ego_motion_hazard", "risk:ego_motion_hazard"}:
            continue
        shared_shift += abs(value - disabled_terms[1][name])
    assert shared_shift < 0.02
    assert abs(enabled_terms[1]["ego_motion_hazard"]) > 10.0 * shared_shift

    # Controlled decision flip: prior pressure wins without the component.
    assert decisions[False].action.index == 1
    assert decisions[True].action.index == 0


def test_contact_hazard_attribution_blames_adjacent_object_and_protects_ego() -> None:
    frame = np.zeros((5, 9), dtype=np.uint8)
    frame[2, 5] = 3
    frame[2, 6] = 4
    _bg, components = connected_components(frame)
    by_value = {component.value: component for component in components}
    ego_signature = by_value[3].signature
    hazard_signature = by_value[4].signature

    agent = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _obs: (0, 1),
    )
    seed_model = ControlAttribution(EgoConfig())
    _seed_directional_control(seed_model, track_id=999, signature=ego_signature)
    agent.ego = ControlAttribution.from_state(
        seed_model.state_dict(),
        config=agent.config.ego,
    )

    observation = Observation(frame=frame, available_actions=(0, 1), task_id="task")
    after_frame = np.zeros((5, 9), dtype=np.uint8)
    after_frame[2, 6] = 4
    after = Observation(frame=after_frame, available_actions=(0, 1), task_id="task")

    agent.begin_run("task", observation)
    decision = agent.act(observation)
    agent.observe(
        decision,
        after,
        Outcome(terminated=True, boundary=BoundaryKind.DEATH, hazard=1.0),
    )

    _c, hazard_belief, _r, terminal_belief = agent.affordances.belief(
        hazard_signature,
        task_id="task",
    )
    ego_beliefs = agent.affordances.belief(ego_signature, task_id="task")
    assert hazard_belief > 0.0
    assert terminal_belief > 0.0
    assert ego_beliefs == (0.0, 0.0, 0.0, 0.0)


def test_disabled_ego_is_an_exact_noop_even_with_loaded_statistics() -> None:
    frame, ego_signature, hazard_signature = _hazard_frame()
    observation = Observation(frame=frame, available_actions=(0, 1), task_id="task")

    seed_model = ControlAttribution(EgoConfig())
    _seed_directional_control(seed_model, track_id=999, signature=ego_signature)

    disabled = CompactHunterSeeker(
        config=_config(ego_enabled=False),
        safe_action_provider=lambda _obs: (0, 1),
    )
    disabled.ego = ControlAttribution.from_state(
        seed_model.state_dict(),
        config=disabled.config.ego,
    )
    disabled.search_engine = disabled._new_search_engine()
    plain = CompactHunterSeeker(
        config=_config(ego_enabled=False),
        safe_action_provider=lambda _obs: (0, 1),
    )
    for agent in (disabled, plain):
        for _ in range(5):
            agent.affordances.observe_signature(
                hazard_signature,
                task_id="task",
                hazard=1.0,
                terminal=1.0,
            )
        agent.begin_run("task", observation)

    disabled_decision = disabled.act(observation)
    plain_decision = plain.act(observation)

    assert disabled_decision.action == plain_decision.action
    assert [
        tuple((term.name, term.value) for term in candidate.terms)
        for candidate in disabled_decision.candidates
    ] == [
        tuple((term.name, term.value) for term in candidate.terms)
        for candidate in plain_decision.candidates
    ]
    assert (
        disabled.ego.observe(
            action=Action(1),
            events=(_moved(999, ego_signature, 2.0),),
            visible_objects=(_object(999, signature=ego_signature),),
        )
        == 0
    )


def test_ego_statistics_survive_checkpoint_roundtrip(tmp_path) -> None:
    frame, ego_signature, _hazard_signature = _hazard_frame()
    observation = Observation(frame=frame, available_actions=(0, 1), task_id="task")
    after = Observation(
        frame=np.roll(frame, 1, axis=1),
        available_actions=(0, 1),
        task_id="task",
    )

    agent = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _obs: (0, 1),
    )
    seed_model = ControlAttribution(EgoConfig())
    _seed_directional_control(seed_model, track_id=999, signature=ego_signature)
    agent.ego = ControlAttribution.from_state(
        seed_model.state_dict(),
        config=agent.config.ego,
    )
    agent.search_engine = agent._new_search_engine()
    agent.begin_run("task", observation)
    decision = agent.act(observation)
    agent.observe(decision, after, Outcome())

    path = tmp_path / "ego-roundtrip.json"
    agent.save_checkpoint(str(path))

    restored = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _obs: (0, 1),
    )
    restored.load_checkpoint(str(path))

    assert restored.ego.state_dict() == agent.ego.state_dict()
    assert restored.ego.influence(999, ego_signature) == agent.ego.influence(
        999,
        ego_signature,
    )
    assert restored.measurement_summary()["ego"] == agent.measurement_summary()["ego"]


def test_state_import_rejects_nonfinite_negative_and_inconsistent_statistics() -> None:
    model = ControlAttribution(EgoConfig())
    _seed_directional_control(
        model,
        track_id=7,
        signature="controlled-signature",
    )
    valid = model.state_dict()
    track_key = next(iter(valid["track_buckets"]))
    action_key = next(iter(valid["track_buckets"][track_key]))
    signature_key = next(iter(valid["signature_buckets"]))
    signature_action = next(
        iter(valid["signature_buckets"][signature_key])
    )
    cases: list[tuple[str, dict]] = []

    malformed = copy.deepcopy(valid)
    malformed["track_buckets"][track_key][action_key][0] = float("nan")
    cases.append(("nonfinite bucket count", malformed))

    malformed = copy.deepcopy(valid)
    malformed["track_buckets"][track_key][action_key][0] = -1.0
    cases.append(("negative bucket count", malformed))

    malformed = copy.deepcopy(valid)
    malformed["track_buckets"][track_key][action_key][0] = 0.5
    cases.append(("fractional bucket count", malformed))

    malformed = copy.deepcopy(valid)
    malformed["track_motion"][track_key][action_key][0] = float("inf")
    cases.append(("nonfinite motion", malformed))

    malformed = copy.deepcopy(valid)
    malformed["track_motion"][track_key][action_key][2] += 1.0
    cases.append(("bucket/motion support mismatch", malformed))

    malformed = copy.deepcopy(valid)
    malformed["track_buckets"]["07"] = malformed["track_buckets"].pop(
        track_key
    )
    cases.append(("noncanonical track key", malformed))

    malformed = copy.deepcopy(valid)
    malformed["signature_motion"][signature_key][signature_action].pop()
    cases.append(("wrong motion row width", malformed))

    malformed = copy.deepcopy(valid)
    malformed["track_signatures"][track_key] = True
    cases.append(("boolean signature", malformed))

    malformed = copy.deepcopy(valid)
    malformed["update_count"] = -1
    cases.append(("negative update count", malformed))

    malformed = copy.deepcopy(valid)
    malformed["update_count"] = True
    cases.append(("boolean update count", malformed))

    malformed = copy.deepcopy(valid)
    malformed["active_task_id"] = 7
    cases.append(("non-string task id", malformed))

    for label, malformed in cases:
        try:
            ControlAttribution.from_state(malformed)
        except ValueError:
            continue
        pytest.fail(f"malformed ego state was accepted: {label}")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True])
def test_observe_rejects_nonfinite_or_boolean_motion(bad_value) -> None:
    model = ControlAttribution(EgoConfig())
    subject = _object(1, signature="subject")
    event = WorldEvent(
        EventKind.MOVED,
        subject_track_id=1,
        object_signature="subject",
        metadata={"dx": bad_value, "dy": 0.0},
    )

    with pytest.raises(ValueError):
        model.observe(
            action=Action(0),
            events=(event,),
            visible_objects=(subject,),
        )

    assert model.state_dict() == ControlAttribution(EgoConfig()).state_dict()
