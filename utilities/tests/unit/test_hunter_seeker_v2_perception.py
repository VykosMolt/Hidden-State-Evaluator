from __future__ import annotations

import copy

import numpy as np
import pytest

from hunter_seeker_v2.contracts import EventKind, Observation, PerceptionConfig
from hunter_seeker_v2.perception import (
    PerceptionSystem,
    cheap_topology,
    connected_components,
)


def _obs(frame: np.ndarray, *, task_id: str = "task") -> Observation:
    return Observation(
        frame=frame,
        available_actions=(0, 1, 2, 3),
        task_id=task_id,
    )


def _event_kinds(result) -> list[EventKind]:
    return [event.kind for event in result.events]


def test_connected_components_are_categorical_deterministic_and_translation_invariant() -> None:
    frame = np.zeros((7, 8), dtype=np.uint8)
    frame[1:3, 1:3] = 4
    frame[4:6, 5:7] = 4
    frame[0, 7] = 9
    config = PerceptionConfig(background_value=0, min_object_area=2)

    background, first = connected_components(frame, config)
    _, second = connected_components(frame.copy(), config)

    assert background == 0
    assert first == second
    assert len(first) == 2
    assert [component.object_id for component in first] == [0, 1]
    assert all(component.value == 4 for component in first)
    assert first[0].signature == first[1].signature
    assert first[0].bbox == (1, 1, 2, 2)
    assert first[1].bbox == (5, 4, 6, 5)
    assert not any(component.value == 9 for component in first)


def test_connected_components_reject_non_categorical_or_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        connected_components(np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="finite integers"):
        connected_components(np.array([[0.0, 1.5]], dtype=np.float32))


def test_tracking_preserves_identity_and_reports_current_move_velocity() -> None:
    system = PerceptionSystem(
        PerceptionConfig(background_value=0, track_match_distance=4.0)
    )
    first = np.zeros((8, 8), dtype=np.uint8)
    first[2:4, 2:4] = 3
    second = np.zeros_like(first)
    second[2:4, 3:5] = 3

    r1 = system.observe(_obs(first))
    r2 = system.observe(_obs(second))

    assert _event_kinds(r1) == [EventKind.APPEARED]
    assert _event_kinds(r2) == [EventKind.MOVED]
    assert r1.objects[0].track_id == r2.objects[0].track_id
    assert r2.objects[0].velocity_x == pytest.approx(1.0)
    assert r2.objects[0].velocity_y == pytest.approx(0.0)
    assert r1.objects[0].signature == r2.objects[0].signature
    moved = r2.events[0]
    assert moved.metadata["dx"] == pytest.approx(1.0)
    assert moved.magnitude == pytest.approx(1.0)


def test_global_matching_preserves_parallel_objects_with_distinct_shapes() -> None:
    """A cheap local edge must not force the remaining identities to swap."""

    system = PerceptionSystem(
        PerceptionConfig(background_value=0, track_match_distance=8.0)
    )
    before = np.zeros((6, 8), dtype=np.uint8)
    before[2, 2] = 3
    before[1:3, 5] = 3
    after = np.zeros_like(before)
    after[2, 0] = 3
    after[1:3, 3] = 3

    first = system.observe(before)
    second = system.observe(after)
    first_by_area = {obj.area: obj for obj in first.objects}
    second_by_area = {obj.area: obj for obj in second.objects}

    assert second_by_area[1].track_id == first_by_area[1].track_id
    assert second_by_area[2].track_id == first_by_area[2].track_id
    assert all(event.kind == EventKind.MOVED for event in second.events)
    assert {
        event.metadata["dx"] for event in second.events
    } == {-2.0}


def test_matching_prefers_same_value_continuity_over_local_cross_value_swaps() -> None:
    """A moving marker must not trade identities with its vacated floor tile."""

    system = PerceptionSystem(
        PerceptionConfig(background_value=5, track_match_distance=8.0)
    )
    before = np.full((14, 22), 5, dtype=np.uint8)
    after = np.full_like(before, 5)

    marker = np.full((3, 3), 9, dtype=np.uint8)
    marker[1, 2] = 4
    floor = np.zeros((3, 3), dtype=np.uint8)
    before[5:8, 4:7] = marker
    before[5:8, 10:13] = floor
    after[5:8, 4:7] = floor
    after[5:8, 10:13] = marker

    first = system.observe(before)
    second = system.observe(after)
    first_by_value = {obj.value: obj for obj in first.objects}
    second_by_value = {obj.value: obj for obj in second.objects}

    assert second_by_value[9].track_id == first_by_value[9].track_id
    assert second_by_value[0].track_id == first_by_value[0].track_id
    assert second_by_value[9].velocity_x == pytest.approx(6.0)
    assert second_by_value[0].velocity_x == pytest.approx(-6.0)
    assert EventKind.TRANSFORMED not in _event_kinds(second)


def test_matching_preserves_cardinality_before_same_value_continuity() -> None:
    """Continuity must not turn two local transformations into appear/disappear."""

    system = PerceptionSystem(
        PerceptionConfig(background_value=0, track_match_distance=8.0)
    )
    before = np.zeros((3, 16), dtype=np.uint8)
    before[1, 5] = 1
    before[1, 13] = 2
    after = np.zeros_like(before)
    after[1, 9] = 1
    after[1, 1] = 2

    first = system.observe(before)
    second = system.observe(after)
    first_by_value = {obj.value: obj for obj in first.objects}
    second_by_value = {obj.value: obj for obj in second.objects}

    assert second_by_value[2].track_id == first_by_value[1].track_id
    assert second_by_value[1].track_id == first_by_value[2].track_id
    assert _event_kinds(second).count(EventKind.TRANSFORMED) == 2
    assert EventKind.APPEARED not in _event_kinds(second)
    assert EventKind.DISAPPEARED not in _event_kinds(second)


def test_local_value_or_shape_change_is_a_transformation_on_same_track() -> None:
    system = PerceptionSystem(
        PerceptionConfig(background_value=0, track_match_distance=4.0)
    )
    before = np.zeros((8, 8), dtype=np.uint8)
    before[2:4, 2:4] = 3
    after = np.zeros_like(before)
    after[2:4, 2:4] = 7

    first = system.observe(before)
    second = system.observe(after)

    assert first.objects[0].track_id == second.objects[0].track_id
    assert _event_kinds(second) == [EventKind.TRANSFORMED]
    event = second.events[0]
    assert event.metadata["old_value"] == 3
    assert event.metadata["new_value"] == 7


def test_disappearance_fires_once_reappearance_keeps_track_and_misses_prune() -> None:
    system = PerceptionSystem(
        PerceptionConfig(
            background_value=0,
            track_match_distance=4.0,
            track_miss_tolerance=2,
        )
    )
    visible = np.zeros((8, 8), dtype=np.uint8)
    visible[2:4, 2:4] = 5
    empty = np.zeros_like(visible)

    first = system.observe(visible)
    track_id = first.objects[0].track_id
    disappeared = system.observe(empty)
    still_missing = system.observe(empty)
    reappeared = system.observe(visible)

    assert _event_kinds(disappeared) == [EventKind.DISAPPEARED]
    assert still_missing.events == ()
    assert _event_kinds(reappeared)[0] == EventKind.APPEARED
    assert reappeared.events[0].metadata["reappeared"] is True
    assert reappeared.objects[0].track_id == track_id

    system.observe(empty)
    system.observe(empty)
    assert system.track_count == 1
    system.observe(empty)
    assert system.track_count == 0


def test_contact_fires_when_adjacency_is_created_not_while_it_persists() -> None:
    system = PerceptionSystem(
        PerceptionConfig(background_value=0, track_match_distance=5.0)
    )
    separated = np.zeros((7, 9), dtype=np.uint8)
    separated[2:4, 1:3] = 2
    separated[2:4, 5:7] = 6
    touching = np.zeros_like(separated)
    touching[2:4, 2:4] = 2
    touching[2:4, 4:6] = 6

    system.observe(separated)
    made_contact = system.observe(touching)
    persisted = system.observe(touching)
    system.observe(separated)
    made_contact_again = system.observe(touching)

    contact_events = [
        event for event in made_contact.events if event.kind == EventKind.CONTACT
    ]
    assert len(contact_events) == 1
    assert contact_events[0].metadata["other_track_id"] >= 0
    assert EventKind.CONTACT not in _event_kinds(persisted)
    assert EventKind.CONTACT in _event_kinds(made_contact_again)


def test_cheap_topology_reports_free_regions_frontiers_and_object_adjacency() -> None:
    frame = np.zeros((6, 7), dtype=np.uint8)
    frame[:, 3] = 9
    frame[2:4, 1:3] = 2
    frame[2:4, 4:6] = 6
    # Make the two categorical objects touch across the central separator.
    frame[2:4, 3] = 9
    background, components = connected_components(
        frame,
        PerceptionConfig(background_value=0),
    )
    topology = cheap_topology(
        frame,
        background_value=background,
        components=components,
    )

    assert topology.component_count == 2
    assert 0.45 <= topology.largest_component_fraction <= 0.55
    assert topology.frontier_fraction > 0.0
    assert topology.reachable_object_ids
    assert topology.object_adjacencies


def test_export_import_roundtrip_preserves_next_transition_behavior() -> None:
    config = PerceptionConfig(
        background_value=0,
        track_match_distance=4.0,
        track_miss_tolerance=2,
    )
    original = PerceptionSystem(config)
    first = np.zeros((8, 8), dtype=np.uint8)
    first[1:3, 1:3] = 3
    second = np.zeros_like(first)
    second[1:3, 2:4] = 3
    third = np.zeros_like(first)
    third[1:3, 3:5] = 3

    original.observe(_obs(first))
    original.observe(_obs(second))
    exported = copy.deepcopy(original.export_state())
    restored = PerceptionSystem.from_state(exported)

    original_result = original.observe(_obs(third))
    restored_result = restored.observe(_obs(third))

    assert restored.export_state() == original.export_state()
    assert restored_result == original_result


def test_import_rejects_noncanonical_and_structurally_invalid_tracks() -> None:
    system = PerceptionSystem(
        PerceptionConfig(
            background_value=0,
            track_match_distance=4.0,
            track_miss_tolerance=2,
        )
    )
    frame = np.zeros((6, 6), dtype=np.uint8)
    frame[1:3, 1:3] = 4
    system.observe(_obs(frame))
    valid = system.export_state()

    cases: list[tuple[str, dict]] = []
    for label, field, value in (
        ("nonfinite centroid", "centroid_x", float("nan")),
        ("nonfinite velocity", "velocity_y", float("inf")),
        ("float track id", "track_id", 0.0),
        ("negative area", "area", -1),
        ("negative misses", "misses", -1),
        ("string boolean", "visible", "true"),
    ):
        malformed = copy.deepcopy(valid)
        malformed["tracks"][0][field] = value
        cases.append((label, malformed))

    malformed = copy.deepcopy(valid)
    malformed["tracks"][0]["area"] += 1
    cases.append(("area/pixel mismatch", malformed))

    malformed = copy.deepcopy(valid)
    malformed["tracks"][0]["bbox"][0] += 1
    cases.append(("bbox/pixel mismatch", malformed))

    malformed = copy.deepcopy(valid)
    malformed["next_track_id"] = 0
    cases.append(("reused next track id", malformed))

    malformed = copy.deepcopy(valid)
    malformed["stage"] = 0
    cases.append(("clamped stage", malformed))

    malformed = copy.deepcopy(valid)
    malformed["config"]["track_match_distance"] = float("nan")
    cases.append(("nonfinite config", malformed))

    malformed = copy.deepcopy(valid)
    malformed["contacts"] = [[0, 99]]
    cases.append(("dangling contact", malformed))

    for label, malformed in cases:
        try:
            PerceptionSystem.from_state(malformed)
        except ValueError:
            continue
        pytest.fail(f"malformed perception state was accepted: {label}")


def test_failed_import_is_atomic() -> None:
    system = PerceptionSystem(PerceptionConfig(background_value=0))
    frame = np.zeros((5, 5), dtype=np.uint8)
    frame[1:3, 1:3] = 7
    system.observe(_obs(frame))
    before = copy.deepcopy(system.export_state())
    malformed = copy.deepcopy(before)
    malformed["config"]["track_match_distance"] = 1.0
    malformed["tracks"][0]["centroid_x"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        system.import_state(malformed)

    assert system.export_state() == before


def test_fork_and_preview_cannot_mutate_durable_tracking_state() -> None:
    system = PerceptionSystem(PerceptionConfig(background_value=0))
    first = np.zeros((8, 8), dtype=np.uint8)
    first[1:3, 1:3] = 3
    hypothetical = np.zeros_like(first)
    hypothetical[5:7, 5:7] = 8
    system.observe(_obs(first))
    state_before = copy.deepcopy(system.export_state())

    fork = system.fork()
    fork.observe(_obs(hypothetical))
    preview = system.preview(_obs(hypothetical))

    assert system.export_state() == state_before
    assert fork.export_state() != state_before
    assert preview.objects
    assert preview.objects[0].value == 8


def test_task_switch_resets_tracking_identity_and_emits_fresh_appearance() -> None:
    system = PerceptionSystem(PerceptionConfig(background_value=0))
    frame = np.zeros((6, 6), dtype=np.uint8)
    frame[2:4, 2:4] = 4

    first = system.observe(_obs(frame, task_id="a"))
    second = system.observe(_obs(frame, task_id="b"))

    assert _event_kinds(first) == [EventKind.APPEARED]
    assert _event_kinds(second) == [EventKind.APPEARED]
    assert second.objects[0].track_id == 0
