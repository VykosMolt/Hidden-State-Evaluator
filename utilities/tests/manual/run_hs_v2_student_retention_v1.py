"""Gate-3 experiment: state-conditioned student route retention.

Two persistent student agents receive identical seeds, policies, budgets, and
episode schedules.  The only pre-episode difference is that one agent imports
the selected trusted trajectory through ``distill_teacher``.  That call trains
the prior, dynamics, affordance/replay, and state-conditioned head, so live arm
differences are attributed to the distilled stack rather than to the head
alone.  A realized-candidate counterfactual separately removes exactly the
head's score term.  Both agents are wired to an explicit
``TeacherAccessGuard``; even an *attempted* action-selection teacher access
fails the experiment.

The primary measures align live states with trusted states before comparing
actions and successors.  Positional overlap and action-frequency similarity
are retained only as diagnostics, since either can look strong without route
retention.

The current GridFeature backend supplies one representation per state.  This
is therefore state-conditioned student distillation, not RLTT.  The student
API is trajectory-aware so a later looped backend can provide multiple latent
representations, but this experiment makes no latent-thought-trajectory claim.

Full run (the output directory must not already exist)::

    venv/bin/python utilities/tests/manual/run_hs_v2_student_retention_v1.py \
        --games ls20 tr87 wa30 --seeds 0 1 --episodes 3 \
        --max-steps 500 --run-index 0
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from hunter_seeker_v2.adapters import (
    ArcActionAdapter,
    ArcObservationAdapter,
    ArcOutcomeAdapter,
)
from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    PolicyConfig,
    RuntimeMode,
)
from hunter_seeker_v2.run_arc import (
    ARC_ENVIRONMENTS_DIR,
    PROJECT_ROOT,
    make_arcade,
    run_episode,
)
from hunter_seeker_v2.student import (
    StateConditionedStudentPolicy,
    StudentPolicyConfig,
)
from hunter_seeker_v2.teacher import (
    TeacherAccessGuard,
    TeacherAccessPhase,
    TeacherQuery,
    TrajectoryTeacher,
)


EXPERIMENT_ID = "hs_v2_state_conditioned_student_retention_v2"
SCHEMA_VERSION = 3
DEFAULT_TRAJECTORY_ROOT = (
    PROJECT_ROOT / "data" / "trajectories" / "trusted_topology_trio_20260513"
)
DEFAULT_ARTIFACT_PARENT = (
    PROJECT_ROOT / "artifacts" / "reports" / "hunter_seeker_v2"
)
ARM_SPECS = (
    ("teacher_distilled_stack", True),
    ("undistilled_stack", False),
)
_GAME_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class TrustedTrajectory:
    """Validated trusted route plus its exact source identity."""

    game: str
    path: Path
    sha256: str
    frames: np.ndarray
    frames_after: np.ndarray
    actions: tuple[Action, ...]
    levels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def stages(self) -> tuple[int, ...]:
        return tuple(level + 1 for level in self.levels)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(row) for key, row in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(row) for row in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("experiment payload contains a nonfinite float")
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_digest(frame: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(frame))
    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        # Trusted recordings and live ARC observations may use different
        # integer storage widths for the same categorical grid. Route
        # identity is content-based, matching ``stable_frame_hash``.
        array = np.ascontiguousarray(array.astype(np.int64, copy=False))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_stable_json(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _observation_signature(observation: Any) -> dict[str, Any]:
    """Exact paired-reset identity independent of integer storage width."""

    payload = {
        "task_id": str(observation.task_id),
        "stage": int(observation.stage),
        "progress": float(observation.progress),
        "available_actions": [
            int(value) for value in observation.available_actions
        ],
        "frame_sha256": _frame_digest(observation.frame),
        "frame_shape": [int(value) for value in observation.frame.shape],
    }
    return {"sha256": _fingerprint(payload), **payload}


def _environment_manifest(games: Sequence[str]) -> dict[str, Any]:
    """Hash the exact repository-local ARC implementations in scope."""

    manifest: dict[str, Any] = {}
    for game in games:
        root = (ARC_ENVIRONMENTS_DIR / str(game)).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"ARC environment directory is absent: {root}")
        files = sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".json"}
        )
        if not files:
            raise FileNotFoundError(
                f"ARC environment {game!r} has no Python/metadata files"
            )
        rows = []
        aggregate = hashlib.sha256()
        for path in files:
            relative = str(path.relative_to(PROJECT_ROOT))
            if path.name == "metadata.json":
                raw_metadata = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw_metadata, Mapping):
                    raise ValueError(f"ARC metadata is not an object: {path}")
                # arc-agi rewrites these cache bookkeeping fields on every
                # fetch. They are not game inputs; all semantic metadata and
                # executable Python remain integrity-checked.
                semantic_metadata = {
                    str(key): value
                    for key, value in raw_metadata.items()
                    if key not in {"date_downloaded", "local_dir"}
                }
                sha256 = _fingerprint(semantic_metadata)
                size = len(_stable_json(semantic_metadata).encode("utf-8"))
                hash_mode = "canonical_json_without_cache_fields"
            else:
                sha256 = _sha256_file(path)
                size = path.stat().st_size
                hash_mode = "raw_file_sha256"
            rows.append(
                {
                    "path": relative,
                    "sha256": sha256,
                    "bytes": size,
                    "hash_mode": hash_mode,
                }
            )
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(sha256.encode("ascii"))
            aggregate.update(b"\0")
        manifest[str(game)] = {
            "aggregate_sha256": aggregate.hexdigest(),
            "files": rows,
        }
    return manifest


def _trajectory_path(root: Path, game: str, run_index: int) -> Path:
    if not _GAME_ID.fullmatch(str(game)):
        raise ValueError(f"unsafe or invalid game id: {game!r}")
    path = root / f"{game}_run{int(run_index)}_traj.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"trusted trajectory {path} does not exist; no fallback run is used"
        )
    return path.resolve()


def _load_trusted(root: Path, game: str, run_index: int) -> TrustedTrajectory:
    path = _trajectory_path(root, game, run_index)
    with np.load(path, allow_pickle=False) as data:
        required = {"frames", "frames_after", "actions", "levels"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        frames = np.array(data["frames"], copy=True)
        frames_after = np.array(data["frames_after"], copy=True)
        raw_actions = np.asarray(data["actions"])
        raw_levels = np.asarray(data["levels"]).reshape(-1)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError(f"{path} frames must have nonempty shape [N,H,W]")
    if frames_after.shape != frames.shape:
        raise ValueError(f"{path} frames_after must match frames")
    if raw_actions.shape != (len(frames), 3):
        raise ValueError(f"{path} actions must have shape [N,3]")
    if len(raw_levels) != len(frames):
        raise ValueError(f"{path} levels must align with frames")
    if not np.issubdtype(raw_actions.dtype, np.integer):
        raise ValueError(f"{path} actions must be integral")
    if not np.issubdtype(raw_levels.dtype, np.integer):
        raise ValueError(f"{path} levels must be integral")
    if np.any(raw_levels < 0):
        raise ValueError(f"{path} levels must be non-negative")
    actions = tuple(
        Action(int(index), int(x), int(y)) for index, x, y in raw_actions
    )
    frames.setflags(write=False)
    frames_after.setflags(write=False)
    return TrustedTrajectory(
        game=str(game),
        path=path,
        sha256=_sha256_file(path),
        frames=frames,
        frames_after=frames_after,
        actions=actions,
        levels=tuple(int(value) for value in raw_levels),
    )


def _trajectory_manifest(route: TrustedTrajectory) -> dict[str, Any]:
    try:
        relative_path: str | None = str(route.path.relative_to(PROJECT_ROOT))
    except ValueError:
        relative_path = None
    return {
        "game": route.game,
        "path": str(route.path),
        "path_relative_to_project": relative_path,
        "sha256": route.sha256,
        "bytes": route.path.stat().st_size,
        "rows": len(route),
        "arrays": {
            "frames": {
                "shape": list(route.frames.shape),
                "dtype": route.frames.dtype.str,
            },
            "frames_after": {
                "shape": list(route.frames_after.shape),
                "dtype": route.frames_after.dtype.str,
            },
            "actions": {"shape": [len(route), 3], "dtype": "integral"},
            "levels": {"shape": [len(route)], "dtype": "integral"},
        },
        "first_frame_sha256": _frame_digest(route.frames[0]),
        "last_after_frame_sha256": _frame_digest(route.frames_after[-1]),
        "action_index_counts": {
            str(index): count
            for index, count in sorted(
                Counter(action.index for action in route.actions).items()
            )
        },
    }


def _normalized_counter(counter: Counter[Any]) -> dict[Any, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {key: float(value / total) for key, value in counter.items()}


def _frequency_similarity(left: Counter[Any], right: Counter[Any]) -> float:
    left_probability = _normalized_counter(left)
    right_probability = _normalized_counter(right)
    if not left_probability or not right_probability:
        return 0.0
    keys = set(left_probability) | set(right_probability)
    distance = 0.5 * sum(
        abs(left_probability.get(key, 0.0) - right_probability.get(key, 0.0))
        for key in keys
    )
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def _expected_frequency_match(left: Counter[Any], right: Counter[Any]) -> float:
    left_probability = _normalized_counter(left)
    right_probability = _normalized_counter(right)
    return float(
        sum(
            left_probability.get(key, 0.0) * right_probability.get(key, 0.0)
            for key in set(left_probability) | set(right_probability)
        )
    )


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _counter_payload(counter: Counter[Any]) -> dict[str, int]:
    def label(key: Any) -> str:
        if isinstance(key, tuple):
            return ",".join(str(value) for value in key)
        return str(key)

    return {label(key): int(value) for key, value in sorted(counter.items())}


def _ordered_state_alignment(
    *,
    live_stages: Sequence[int],
    live_digests: Sequence[str],
    route_stages: Sequence[int],
    route_digests: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Longest state-only ordered alignment with deterministic early ties.

    This is the Hunt-Szymanski form of LCS. Candidate occurrence selection
    depends only on frame content and the documented one-stage boundary
    tolerance; the live action and successor never participate. Repeated
    trusted states therefore cannot be cherry-picked to make an action look
    correct after it was observed.
    """

    if len(live_stages) != len(live_digests):
        raise ValueError("live alignment stages and digests do not align")
    if len(route_stages) != len(route_digests):
        raise ValueError("route alignment stages and digests do not align")
    route_by_digest: dict[str, list[int]] = {}
    for route_index, digest in enumerate(route_digests):
        route_by_digest.setdefault(str(digest), []).append(route_index)

    # Each node is (live index, route index, predecessor node index).
    nodes: list[tuple[int, int, int | None]] = []
    tail_route_indexes: list[int] = []
    tail_nodes: list[int] = []
    for live_index, (stage, digest) in enumerate(
        zip(live_stages, live_digests, strict=True)
    ):
        candidates = [
            route_index
            for route_index in route_by_digest.get(str(digest), ())
            if abs(int(route_stages[route_index]) - int(stage)) <= 1
        ]
        # Descending route positions prevent one live row from extending an
        # alignment more than once when a trusted state is repeated.
        for route_index in reversed(candidates):
            length_index = bisect_left(tail_route_indexes, route_index)
            predecessor = (
                tail_nodes[length_index - 1] if length_index > 0 else None
            )
            node_index = len(nodes)
            nodes.append((live_index, route_index, predecessor))
            if length_index == len(tail_route_indexes):
                tail_route_indexes.append(route_index)
                tail_nodes.append(node_index)
            elif route_index < tail_route_indexes[length_index]:
                tail_route_indexes[length_index] = route_index
                tail_nodes[length_index] = node_index

    if not tail_nodes:
        return ()
    result: list[tuple[int, int]] = []
    cursor: int | None = tail_nodes[-1]
    while cursor is not None:
        live_index, route_index, cursor = nodes[cursor]
        result.append((live_index, route_index))
    result.reverse()
    return tuple(result)


def _student_term(candidate: Any) -> tuple[str, float, str, bool]:
    terms = tuple(
        term for term in candidate.terms if str(term[0]) == "student_policy"
    )
    if len(terms) != 1:
        raise AssertionError(
            f"candidate {candidate.action} has {len(terms)} student terms"
        )
    name, value, group, influences = terms[0]
    if not np.isfinite(float(value)):
        raise AssertionError("student policy score term is nonfinite")
    if str(group) != "learned_value" or not bool(influences):
        raise AssertionError(
            "student policy term must be one score-influencing learned_value term"
        )
    return str(name), float(value), str(group), bool(influences)


def _trace_effective_risk(candidate: Any) -> float:
    explicit = sum(
        max(0.0, float(term[1]))
        for term in candidate.terms
        if str(term[0]).startswith("risk:")
    )
    if explicit > 0.0:
        policy_risk = explicit
    else:
        policy_risk = sum(
            -float(term[1])
            for term in candidate.terms
            if str(term[0])
            in {"evidence_negative", "target_hazard_affordance"}
            and float(term[1]) < 0.0
        )
    return float(np.clip(float(candidate.risk) + policy_risk, 0.0, 2.0))


def _without_student_choice(decision: Any, *, risk_limit: float) -> tuple[int, int, int]:
    """Exact epsilon-zero arbiter counterfactual on the realized candidates."""

    rows = []
    for candidate in decision.candidates:
        _name, student_value, _group, _influences = _student_term(candidate)
        rows.append(
            (
                candidate,
                float(candidate.score) - student_value,
                _trace_effective_risk(candidate),
            )
        )
    safe = [row for row in rows if row[2] <= float(risk_limit)]
    # CandidateTrace stores Action.key as (index, x, y), whereas RiskArbiter's
    # deterministic tie-break is (index, y, x).  Reproduce the controller's
    # order explicitly instead of expanding the serialized tuple.
    def action_order(row: tuple[Any, float, float]) -> tuple[int, int, int]:
        index, x, y = row[0].action
        return int(index), int(y), int(x)

    if safe:
        chosen = min(
            safe,
            key=lambda row: (-row[1], *action_order(row)),
        )
    else:
        chosen = min(
            rows,
            key=lambda row: (row[2], -row[1], *action_order(row)),
        )
    return tuple(int(value) for value in chosen[0].action)


def _chosen_candidate_payload(
    decision: Any,
    *,
    risk_limit: float | None = None,
) -> dict[str, Any]:
    candidates = [
        row
        for row in decision.candidates
        if tuple(row.action) == tuple(decision.chosen_action)
    ]
    if not candidates:
        raise AssertionError(
            f"decision {decision.decision_id} has no chosen candidate trace"
        )
    candidate = candidates[0]
    terms = [
        {
            "name": str(name),
            "value": float(value),
            "group": str(group),
            "influences_score": bool(influences),
        }
        for name, value, group, influences in candidate.terms
    ]
    payload = {
        "decision_id": decision.decision_id,
        "chosen_score": float(decision.chosen_score),
        "selection_method": decision.method,
        "student_policy_terms": [
            row for row in terms if row["name"] == "student_policy"
        ],
        "chosen_candidate_terms": terms,
    }
    if risk_limit is not None:
        without_student = _without_student_choice(
            decision,
            risk_limit=float(risk_limit),
        )
        payload.update(
            {
                "without_student_action": list(without_student),
                "student_term_changed_choice": bool(
                    without_student != tuple(decision.chosen_action)
                ),
            }
        )
    return payload


def _assert_student_scoring(
    decisions: Sequence[Any],
    *,
    require_nonzero: bool = False,
) -> dict[str, Any]:
    """Audit that the integrated head is live and score-causal."""

    if not decisions:
        raise AssertionError("student scoring audit received no decisions")
    invalid: list[str] = []
    values: list[float] = []
    candidate_count = 0
    for decision in decisions:
        if not decision.candidates:
            invalid.append(f"{decision.decision_id}:no-candidates")
        for candidate in decision.candidates:
            candidate_count += 1
            try:
                _name, value, _group, _influences = _student_term(candidate)
                values.append(value)
            except AssertionError as exc:
                invalid.append(
                    f"{decision.decision_id}:{candidate.action}:{exc}"
                )
    if invalid:
        preview = ", ".join(invalid[:5])
        raise AssertionError(
            "student policy scoring contract failed: " + preview
        )
    nonzero = sum(int(abs(value) > 1e-12) for value in values)
    if require_nonzero and nonzero == 0:
        raise AssertionError("trained student policy emitted only zero score terms")
    return {
        "decisions": len(decisions),
        "candidates": candidate_count,
        "nonzero_terms": nonzero,
        "max_absolute_term": max((abs(value) for value in values), default=0.0),
    }


def _analyze_episode(
    result: Any,
    route: TrustedTrajectory,
    decisions: Sequence[Any] = (),
    *,
    risk_limit: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return independent frequency, positional, and state-aligned measures."""

    transitions = tuple(result.transitions)
    if decisions and len(decisions) != len(transitions):
        raise AssertionError("decision and transition traces do not align")
    decision_by_id = {row.decision_id: row for row in decisions}
    route_before = tuple(_frame_digest(frame) for frame in route.frames)
    route_after = tuple(_frame_digest(frame) for frame in route.frames_after)
    exact_index: dict[tuple[int, str], list[int]] = {}
    frame_index: dict[str, list[int]] = {}
    for index, (stage, digest) in enumerate(zip(route.stages, route_before, strict=True)):
        exact_index.setdefault((stage, digest), []).append(index)
        frame_index.setdefault(digest, []).append(index)

    live_actions = tuple(transition.action for transition in transitions)
    live_before = tuple(
        _frame_digest(transition.before.observation.frame)
        for transition in transitions
    )
    live_after = tuple(
        _frame_digest(transition.after_observation.frame)
        for transition in transitions
    )
    live_stages = tuple(
        int(transition.before.observation.stage) for transition in transitions
    )
    live_indices = tuple(action.index for action in live_actions)
    route_indices = tuple(action.index for action in route.actions)
    live_keys = tuple(action.key for action in live_actions)
    route_keys = tuple(action.key for action in route.actions)
    live_index_counts = Counter(live_indices)
    route_index_counts = Counter(route_indices)
    live_key_counts = Counter(live_keys)
    route_key_counts = Counter(route_keys)
    live_bigrams = Counter(zip(live_indices, live_indices[1:]))
    route_bigrams = Counter(zip(route_indices, route_indices[1:]))
    ordered_pairs = _ordered_state_alignment(
        live_stages=live_stages,
        live_digests=live_before,
        route_stages=route.stages,
        route_digests=route_before,
    )
    ordered_by_live = {
        live_index: route_index for live_index, route_index in ordered_pairs
    }
    ordered_action_matches = sum(
        int(live_keys[live_index] == route_keys[route_index])
        for live_index, route_index in ordered_pairs
    )
    ordered_successor_matches = sum(
        int(
            live_keys[live_index] == route_keys[route_index]
            and live_after[live_index] == route_after[route_index]
        )
        for live_index, route_index in ordered_pairs
    )

    aligned_steps = 0
    exact_stage_aligned_steps = 0
    frame_only_steps = 0
    aligned_index_matches = 0
    aligned_key_matches = 0
    aligned_successor_matches = 0
    reached_route_rows: set[int] = set()
    trace: list[dict[str, Any]] = []

    for position, transition in enumerate(transitions):
        before_digest = live_before[position]
        after_digest = live_after[position]
        stage = live_stages[position]
        exact_stage_candidates = tuple(
            exact_index.get((stage, before_digest), ())
        )
        frame_candidates = tuple(frame_index.get(before_digest, ()))
        # ARC recordings can label the same boundary frame with the level
        # before or after the transition. Admit only that one-level offset;
        # unrelated cross-stage frame matches remain excluded.
        candidates = tuple(
            index
            for index in frame_candidates
            if abs(int(route.stages[index]) - stage) <= 1
        )
        if frame_candidates:
            frame_only_steps += 1
        if exact_stage_candidates:
            exact_stage_aligned_steps += 1
        index_matches = tuple(
            index
            for index in candidates
            if route.actions[index].index == transition.action.index
        )
        key_matches = tuple(
            index
            for index in candidates
            if route.actions[index].key == transition.action.key
        )
        successor_matches = tuple(
            index for index in key_matches if route_after[index] == after_digest
        )
        if candidates:
            aligned_steps += 1
            aligned_index_matches += int(bool(index_matches))
            aligned_key_matches += int(bool(key_matches))
            aligned_successor_matches += int(bool(successor_matches))
            preferred = successor_matches or key_matches or index_matches or candidates
            reached_route_rows.add(int(preferred[0]))

        ordered_route_row = ordered_by_live.get(position)
        ordered_action_match = (
            None
            if ordered_route_row is None
            else transition.action.key == route.actions[ordered_route_row].key
        )
        ordered_successor_match = (
            None
            if ordered_route_row is None
            else bool(
                ordered_action_match
                and after_digest == route_after[ordered_route_row]
            )
        )

        decision_payload = (
            _chosen_candidate_payload(
                decision_by_id[transition.decision_id],
                risk_limit=risk_limit,
            )
            if transition.decision_id in decision_by_id
            else None
        )
        trace.append(
            {
                "position": position,
                "transition_id": transition.transition_id,
                "decision_id": transition.decision_id,
                "stage": stage,
                "progress": float(transition.before.observation.progress),
                "before_frame_sha256": before_digest,
                "after_frame_sha256": after_digest,
                "before_state_id": transition.before.state_id,
                "after_state_id": transition.after_state_id,
                "action": list(transition.action.key),
                "reward": float(transition.outcome.reward),
                "progress_delta": float(transition.outcome.progress_delta),
                "hazard": float(transition.outcome.hazard),
                "boundary": transition.outcome.boundary.value,
                "terminated": bool(transition.outcome.terminated),
                "truncated": bool(transition.outcome.truncated),
                "exact_stage_state_route_rows": list(exact_stage_candidates),
                "boundary_tolerant_state_route_rows": list(candidates),
                "frame_only_route_rows": list(frame_candidates),
                "state_aligned_action_index_match": bool(index_matches),
                "state_aligned_full_action_match": bool(key_matches),
                "state_action_successor_match": bool(successor_matches),
                "ordered_state_route_row": ordered_route_row,
                "ordered_state_action_match": ordered_action_match,
                "ordered_state_action_successor_match": (
                    ordered_successor_match
                ),
                "decision": decision_payload,
            }
        )

    shared_horizon = min(len(transitions), len(route))
    position_index_matches = sum(
        int(live_indices[index] == route_indices[index])
        for index in range(shared_horizon)
    )
    position_key_matches = sum(
        int(live_keys[index] == route_keys[index])
        for index in range(shared_horizon)
    )
    prefix_steps = 0
    exact_stage_prefix_steps = 0
    for index in range(shared_horizon):
        transition = transitions[index]
        frame_matches = (
            _frame_digest(transition.before.observation.frame) == route_before[index]
        )
        stage_delta = abs(
            int(transition.before.observation.stage) - int(route.stages[index])
        )
        exact_stage_step = (
            frame_matches
            and stage_delta == 0
            and transition.action.key == route.actions[index].key
            and _frame_digest(transition.after_observation.frame) == route_after[index]
        )
        tolerant_step = (
            frame_matches
            and stage_delta <= 1
            and transition.action.key == route.actions[index].key
            and _frame_digest(transition.after_observation.frame) == route_after[index]
        )
        if exact_stage_step and exact_stage_prefix_steps == index:
            exact_stage_prefix_steps += 1
        if not tolerant_step:
            break
        prefix_steps += 1

    route_majority_rate = (
        max(route_index_counts.values()) / len(route_indices) if route_indices else 0.0
    )
    live_majority_rate = (
        max(live_index_counts.values()) / len(live_indices) if live_indices else 0.0
    )
    counterfactual_changes = sum(
        int(
            _without_student_choice(decision, risk_limit=float(risk_limit))
            != tuple(decision.chosen_action)
        )
        for decision in decisions
    ) if risk_limit is not None else 0
    metrics = {
        "live_steps": len(transitions),
        "trusted_route_rows": len(route),
        "shared_positional_horizon": shared_horizon,
        "position_action_index_overlap": _ratio(
            position_index_matches, shared_horizon
        ),
        "position_full_action_overlap": _ratio(position_key_matches, shared_horizon),
        "strict_route_prefix_state_action_successor_steps": prefix_steps,
        "strict_route_prefix_fraction": _ratio(prefix_steps, len(route)),
        "strict_route_prefix_shared_horizon_fraction": _ratio(
            prefix_steps, shared_horizon
        ),
        "exact_stage_strict_route_prefix_state_action_successor_steps": (
            exact_stage_prefix_steps
        ),
        "exact_stage_strict_route_prefix_fraction": _ratio(
            exact_stage_prefix_steps, len(route)
        ),
        "state_aligned_steps": aligned_steps,
        "exact_stage_state_aligned_steps": exact_stage_aligned_steps,
        "frame_only_aligned_steps": frame_only_steps,
        "live_state_alignment_rate": _ratio(aligned_steps, len(transitions)),
        "exact_stage_live_state_alignment_rate": _ratio(
            exact_stage_aligned_steps, len(transitions)
        ),
        "state_aligned_action_index_accuracy": _ratio(
            aligned_index_matches, aligned_steps
        ),
        "state_aligned_full_action_accuracy": _ratio(
            aligned_key_matches, aligned_steps
        ),
        "state_action_successor_accuracy": _ratio(
            aligned_successor_matches, aligned_steps
        ),
        "unique_trusted_rows_reached": len(reached_route_rows),
        "trusted_row_coverage": _ratio(len(reached_route_rows), len(route)),
        "trusted_row_opportunity_coverage": _ratio(
            len(reached_route_rows), shared_horizon
        ),
        "ordered_state_rows_aligned": len(ordered_pairs),
        "ordered_state_alignment_shared_horizon_rate": _ratio(
            len(ordered_pairs), shared_horizon
        ),
        "ordered_state_route_coverage": _ratio(len(ordered_pairs), len(route)),
        "ordered_state_action_matches": ordered_action_matches,
        "ordered_state_action_accuracy": _ratio(
            ordered_action_matches, len(ordered_pairs)
        ),
        "ordered_state_action_successor_matches": ordered_successor_matches,
        "ordered_state_action_successor_accuracy": _ratio(
            ordered_successor_matches, len(ordered_pairs)
        ),
        "ordered_transition_shared_horizon_rate": _ratio(
            ordered_successor_matches, shared_horizon
        ),
        "ordered_transition_route_coverage": _ratio(
            ordered_successor_matches, len(route)
        ),
        "level_completion_transitions": sum(
            int(transition.outcome.boundary == BoundaryKind.LEVEL_COMPLETED)
            for transition in transitions
        ),
        "game_completion_transitions": sum(
            int(transition.outcome.boundary == BoundaryKind.GAME_COMPLETED)
            for transition in transitions
        ),
        "student_counterfactual_evaluable_decisions": (
            len(decisions) if risk_limit is not None else 0
        ),
        "student_counterfactual_choice_changes": counterfactual_changes,
        "student_counterfactual_choice_change_rate": _ratio(
            counterfactual_changes,
            len(decisions) if risk_limit is not None else 0,
        ),
        "action_index_frequency_similarity": _frequency_similarity(
            live_index_counts, route_index_counts
        ),
        "full_action_frequency_similarity": _frequency_similarity(
            live_key_counts, route_key_counts
        ),
        "action_bigram_frequency_similarity": _frequency_similarity(
            live_bigrams, route_bigrams
        ),
        "frequency_only_expected_index_match": _expected_frequency_match(
            live_index_counts, route_index_counts
        ),
        "trusted_majority_action_rate": float(route_majority_rate),
        "live_majority_action_rate": float(live_majority_rate),
        "trusted_action_index_counts": _counter_payload(route_index_counts),
        "live_action_index_counts": _counter_payload(live_index_counts),
        "trusted_full_action_counts": _counter_payload(route_key_counts),
        "live_full_action_counts": _counter_payload(live_key_counts),
    }
    return metrics, trace


def _teacher_audit_payload(guard: TeacherAccessGuard) -> dict[str, Any]:
    records = tuple(guard.audit.records)
    action_selection_attempts = sum(
        int(row.phase is TeacherAccessPhase.ACTION_SELECTION) for row in records
    )
    payload = {
        "teacher_id": guard.teacher_id,
        "mode": guard.mode.value,
        "summary": dict(guard.audit.summary()),
        "action_selection_attempts": action_selection_attempts,
        "records": [_jsonable(row) for row in records],
    }
    guard.assert_no_action_selection_reads()
    if action_selection_attempts != 0:
        raise AssertionError(
            f"teacher {guard.teacher_id!r} had an action-selection access attempt"
        )
    return payload


def _source_manifest(script_path: Path) -> dict[str, Any]:
    candidates = sorted((PROJECT_ROOT / "src" / "hunter_seeker_v2").rglob("*.py"))
    candidates.append(script_path.resolve())
    for name in ("pyproject.toml", "uv.lock", "requirements.txt"):
        path = PROJECT_ROOT / name
        if path.is_file():
            candidates.append(path)
    unique = sorted(set(path.resolve() for path in candidates))
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in unique:
        relative = str(path.relative_to(PROJECT_ROOT))
        sha256 = _sha256_file(path)
        files.append(
            {
                "path": relative,
                "sha256": sha256,
                "bytes": path.stat().st_size,
            }
        )
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(sha256.encode("ascii"))
        aggregate.update(b"\0")
    return {"aggregate_sha256": aggregate.hexdigest(), "files": files}


def _command_output(arguments: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(
        tuple(arguments),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": list(arguments),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _package_versions() -> dict[str, str | None]:
    names = ("numpy", "torch", "arc-agi", "arcengine", "pytest")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _provenance(
    *,
    args: argparse.Namespace,
    script_path: Path,
    routes: Mapping[str, TrustedTrajectory],
    run_id: str,
    design: Mapping[str, Any],
    environment_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "argv": [str(script_path), *sys.argv[1:]],
        "arguments": _jsonable(vars(args)),
        "working_directory": str(Path.cwd().resolve()),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "package_versions": _package_versions(),
        "git": {
            "head": _command_output(("git", "rev-parse", "HEAD")),
            "status": _command_output(("git", "status", "--porcelain=v1")),
            "diff_stat": _command_output(("git", "diff", "--stat")),
        },
        "design": _jsonable(design),
        "source_manifest": _jsonable(source_manifest),
        "environment_manifest": _jsonable(environment_manifest),
        "trajectory_manifest": {
            game: _trajectory_manifest(route) for game, route in routes.items()
        },
        "methodology": {
            "label": "state_conditioned_student_stack_distillation",
            "is_rltt_experiment": False,
            "reason": (
                "The integrated runtime supplies one connector output per state; "
                "the trajectory-aware head does not by itself constitute RLTT"
            ),
            "teacher_boundary": (
                "Every arm owns a student-mode TeacherAccessGuard; zero attempted "
                "action-selection accesses are required after every episode"
            ),
        },
    }


def _exclusive_output_directory(requested: Path | None, run_id: str) -> Path:
    target = (
        requested.expanduser().resolve()
        if requested is not None
        else (DEFAULT_ARTIFACT_PARENT / f"student_retention_{run_id}").resolve()
    )
    target.mkdir(parents=True, exist_ok=False)
    return target


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(_jsonable(payload), target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def _episode_seed(seed: int, episode: int) -> int:
    payload = f"{int(seed)}|{int(episode)}|hs-v2-retention".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _arm_execution_order(seed: int, episode: int) -> tuple[tuple[str, bool], ...]:
    """Alternate which arm runs first to expose process-order dependence."""

    if _episode_seed(seed, episode) % 2:
        return tuple(reversed(ARM_SPECS))
    return ARM_SPECS


def _set_episode_seed(seed: int, episode: int) -> int:
    value = _episode_seed(seed, episode)
    random.seed(value)
    np.random.seed(value)
    return value


def _run_seeded_arc_episode(
    game: str,
    agent: CompactHunterSeeker,
    *,
    arcade: Any,
    environment_seed: int,
    max_steps: int,
) -> Any:
    """Mirror ``run_arc_game`` while setting the ARC seed explicitly."""

    from arcengine import GameAction

    environment = arcade.make(
        str(game),
        seed=int(environment_seed),
        render_mode=None,
    )
    if environment is None:
        raise RuntimeError(f"ARC could not construct game {game!r}")
    try:
        return run_episode(
            environment,
            agent,
            task_id=str(game),
            observation_adapter=ArcObservationAdapter(),
            action_adapter=ArcActionAdapter(),
            outcome_adapter=ArcOutcomeAdapter(),
            action_space=GameAction,
            max_steps=max_steps,
            bootstrap=True,
        )
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def _paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["game"], int(row["seed"]), int(row["episode"]), row["arm"]): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    metrics = (
        "progress",
        "level_completion_transitions",
        "game_completion_transitions",
        "strict_route_prefix_shared_horizon_fraction",
        "ordered_state_alignment_shared_horizon_rate",
        "ordered_state_action_accuracy",
        "ordered_state_action_successor_accuracy",
        "ordered_transition_shared_horizon_rate",
        "student_counterfactual_choice_change_rate",
    )
    keys = sorted({(row["game"], int(row["seed"]), int(row["episode"])) for row in rows})
    for game, seed, episode in keys:
        distilled = by_key[(game, seed, episode, "teacher_distilled_stack")]
        baseline = by_key[(game, seed, episode, "undistilled_stack")]
        comparisons.append(
            {
                "game": game,
                "seed": seed,
                "episode": episode,
                "teacher_distilled_minus_undistilled": {
                    metric: float(distilled[metric]) - float(baseline[metric])
                    for metric in metrics
                },
            }
        )
    return comparisons


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=["ls20", "tr87", "wa30"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=DEFAULT_TRAJECTORY_ROOT,
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=(
            "exclusive output directory (must not exist); by default a UTC/UUID "
            "run directory is created under artifacts/reports/hunter_seeker_v2"
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.run_index < 0:
        raise ValueError("run-index must be non-negative")
    if len(args.games) != len(set(args.games)):
        raise ValueError("games must not contain duplicates")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must not contain duplicates")
    if any(int(seed) < 0 for seed in args.seeds):
        raise ValueError("seeds must be non-negative")
    for game in args.games:
        if not _GAME_ID.fullmatch(str(game)):
            raise ValueError(f"invalid game id: {game!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    script_path = Path(__file__).resolve()
    trajectory_root = Path(args.trajectory_root).expanduser().resolve()
    routes = {
        game: _load_trusted(trajectory_root, game, args.run_index)
        for game in args.games
    }
    common_config = AgentConfig(
        runtime_mode=RuntimeMode.STUDENT,
        seed=0,
        policy=PolicyConfig(exploration_epsilon=0.0),
    )
    student_config = StudentPolicyConfig()
    design = {
        "arms": [name for name, _distilled in ARM_SPECS],
        "only_pre_schedule_arm_difference": "offline_distill_teacher_call",
        "distill_teacher_updates": [
            "state_independent_action_prior",
            "action_conditioned_dynamics",
            "affordance_and_teacher_replay",
            "state_conditioned_student_policy",
        ],
        "rollout_attribution": (
            "paired arm differences belong to the distilled stack; the "
            "realized-candidate counterfactual removes exactly student_policy"
        ),
        "persistent_agent_across_episodes": True,
        "paired_episode_environment_seed": True,
        "paired_initial_observation_must_match": True,
        "alternating_arm_execution_order": True,
        "agent_config_template": _jsonable(common_config),
        "student_policy_config": _jsonable(student_config),
        "primary_route_metrics": [
            "strict_route_prefix_shared_horizon_fraction",
            "ordered_state_alignment_shared_horizon_rate",
            "ordered_state_action_accuracy",
            "ordered_state_action_successor_accuracy",
            "ordered_transition_shared_horizon_rate",
        ],
        "primary_mapping_rule": (
            "state-only longest ordered alignment; actions and successors are "
            "scored only after the occurrence mapping is fixed"
        ),
        "diagnostics_not_sufficient_for_route_retention": [
            "position_action_index_overlap",
            "state_aligned_full_action_accuracy",
            "state_action_successor_accuracy",
            "trusted_row_coverage",
            "action_index_frequency_similarity",
            "action_bigram_frequency_similarity",
            "frequency_only_expected_index_match",
        ],
        "head_causality_diagnostic": (
            "epsilon-zero realized-candidate choice recomputed with exactly the "
            "student_policy term removed"
        ),
    }
    source_manifest = _source_manifest(script_path)
    environment_manifest = _environment_manifest(args.games)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "_"
        + uuid4().hex[:12]
    )
    output = _exclusive_output_directory(args.out_root, run_id)
    traces_dir = output / "episode_traces"
    traces_dir.mkdir()
    provenance = _provenance(
        args=args,
        script_path=script_path,
        routes=routes,
        run_id=run_id,
        design=design,
        environment_manifest=environment_manifest,
        source_manifest=source_manifest,
    )
    _write_json_exclusive(output / "run_started.json", provenance)
    rows: list[dict[str, Any]] = []
    arcade = make_arcade()
    rows_path = output / "rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as rows_file:
        for seed in args.seeds:
            config = dataclasses.replace(common_config, seed=int(seed))
            config_fingerprint = _fingerprint(config)
            for game in args.games:
                route = routes[game]
                arms: dict[str, dict[str, Any]] = {}
                initial_head_fingerprints: set[str] = set()
                for arm_name, distilled in ARM_SPECS:
                    action_adapter = ArcActionAdapter()
                    if _sha256_file(route.path) != route.sha256:
                        raise RuntimeError(
                            f"trusted trajectory changed before {arm_name} construction"
                        )
                    provider = TrajectoryTeacher.from_npz(
                        str(route.path),
                        task_id=game,
                        teacher_id=f"trusted:{game}:{route.sha256[:12]}",
                    )
                    if _sha256_file(route.path) != route.sha256:
                        raise RuntimeError(
                            f"trusted trajectory changed while loading {arm_name}"
                        )
                    guard = TeacherAccessGuard(provider, mode=RuntimeMode.STUDENT)
                    head = StateConditionedStudentPolicy(student_config)
                    initial_head_fingerprints.add(_fingerprint(head.state_dict()))
                    agent = CompactHunterSeeker(
                        config=config,
                        click_action_index=action_adapter.click_action_index(),
                        safe_action_provider=action_adapter.safe_action_indices,
                        teacher=guard,
                        student_policy=head,
                    )
                    if agent.teacher is not guard:
                        raise AssertionError("agent did not retain the explicit teacher guard")
                    if agent.student_policy is not head:
                        raise AssertionError("agent did not retain the explicit student head")
                    distilled_examples = 0
                    if distilled:
                        distilled_examples = agent.distill_teacher(
                            TeacherQuery(task_id=game, limit=max(len(route), 1))
                        )
                        if distilled_examples != len(route):
                            raise AssertionError(
                                f"distilled {distilled_examples} examples, expected {len(route)}"
                            )
                        if agent.student_policy.teacher_updates <= 0:
                            raise AssertionError(
                                "distill_teacher did not update the state-conditioned head"
                            )
                    elif agent.student_policy.teacher_updates != 0:
                        raise AssertionError("undistilled head received teacher updates")
                    arms[arm_name] = {
                        "agent": agent,
                        "guard": guard,
                        "distilled_examples": distilled_examples,
                        "pre_schedule_student_summary": (
                            agent.student_policy.summary()
                        ),
                    }
                if len(initial_head_fingerprints) != 1:
                    raise AssertionError("arms did not begin with identical student heads")

                for episode in range(1, args.episodes + 1):
                    paired_seed = _episode_seed(seed, episode)
                    execution_order = _arm_execution_order(seed, episode)
                    paired_initial_signature: dict[str, Any] | None = None
                    for execution_position, (arm_name, distilled) in enumerate(
                        execution_order,
                        start=1,
                    ):
                        arm = arms[arm_name]
                        agent = arm["agent"]
                        guard = arm["guard"]
                        applied_seed = _set_episode_seed(seed, episode)
                        if applied_seed != paired_seed:
                            raise AssertionError("paired episode seed changed between arms")
                        pre_episode_student_summary = agent.student_policy.summary()
                        pre_episode_student_sha256 = _fingerprint(
                            agent.student_policy.state_dict()
                        )
                        decision_offset = len(agent.diagnostics.decisions)
                        started = time.monotonic()
                        result = _run_seeded_arc_episode(
                            game,
                            agent,
                            arcade=arcade,
                            environment_seed=paired_seed,
                            max_steps=args.max_steps,
                        )
                        elapsed = time.monotonic() - started
                        decisions = agent.diagnostics.decisions[decision_offset:]
                        scoring_audit = _assert_student_scoring(
                            decisions,
                            require_nonzero=distilled,
                        )
                        audit = _teacher_audit_payload(guard)
                        initial_signature = _observation_signature(
                            result.initial_observation
                        )
                        if paired_initial_signature is None:
                            paired_initial_signature = initial_signature
                        elif initial_signature != paired_initial_signature:
                            raise AssertionError(
                                "paired seeded environments produced different "
                                f"initial observations for {game} seed={seed} "
                                f"episode={episode}"
                            )
                        metrics, trace = _analyze_episode(
                            result,
                            route,
                            decisions,
                            risk_limit=float(config.policy.risk_limit),
                        )
                        post_episode_student_sha256 = _fingerprint(
                            agent.student_policy.state_dict()
                        )
                        trace_name = (
                            f"{game}_seed{seed}_{arm_name}_episode{episode}.json"
                        )
                        _write_json_exclusive(
                            traces_dir / trace_name,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "experiment": EXPERIMENT_ID,
                                "run_id": run_id,
                                "game": game,
                                "seed": seed,
                                "paired_episode_seed": paired_seed,
                                "arm": arm_name,
                                "episode": episode,
                                "arm_execution_order": [
                                    name for name, _value in execution_order
                                ],
                                "arm_execution_position": execution_position,
                                "initial_observation_signature": initial_signature,
                                "trajectory_sha256": route.sha256,
                                "steps": trace,
                            },
                        )
                        summary = agent.measurement_summary()
                        action_indices = sorted(
                            {action.index for action in route.actions}
                            | {
                                transition.action.index
                                for transition in result.transitions
                            }
                        )
                        row: dict[str, Any] = {
                            "game": game,
                            "seed": int(seed),
                            "paired_episode_seed": paired_seed,
                            "agent_config_sha256": config_fingerprint,
                            "arm": arm_name,
                            "arm_execution_order": [
                                name for name, _value in execution_order
                            ],
                            "arm_execution_position": execution_position,
                            "episode": episode,
                            "budget_steps": args.max_steps,
                            "steps": result.steps,
                            "boundary": result.final_outcome.boundary.value,
                            "progress": float(result.final_observation.progress),
                            "stage": int(result.final_observation.stage),
                            "elapsed_seconds": float(elapsed),
                            "distilled_examples": arm["distilled_examples"],
                            "trajectory_sha256": route.sha256,
                            "initial_observation_signature": initial_signature,
                            "trace_file": f"episode_traces/{trace_name}",
                            "replay_teacher_fraction": float(
                                summary.get("replay_teacher_fraction", 0.0)
                            ),
                            "teacher_audit": audit,
                            "student_scoring_audit": scoring_audit,
                            "pre_schedule_student_summary": arm[
                                "pre_schedule_student_summary"
                            ],
                            "pre_episode_student_summary": (
                                pre_episode_student_summary
                            ),
                            "pre_episode_student_state_sha256": (
                                pre_episode_student_sha256
                            ),
                            "post_episode_student_summary": (
                                agent.student_policy.summary()
                            ),
                            "post_episode_student_state_sha256": (
                                post_episode_student_sha256
                            ),
                            "prior_scores": {
                                str(index): float(agent.prior.score(game, index))
                                for index in action_indices
                            },
                            **metrics,
                        }
                        rows.append(row)
                        line = _stable_json(row)
                        rows_file.write(line + "\n")
                        rows_file.flush()
                        os.fsync(rows_file.fileno())
                        print(line, flush=True)

    completion_source_manifest = _source_manifest(script_path)
    if completion_source_manifest != source_manifest:
        raise RuntimeError("Hunter-Seeker source changed during the experiment")
    completion_environment_manifest = _environment_manifest(args.games)
    if completion_environment_manifest != environment_manifest:
        raise RuntimeError("ARC environment source changed during the experiment")
    completion_trajectory_hashes = {
        game: _sha256_file(route.path) for game, route in routes.items()
    }
    expected_trajectory_hashes = {
        game: route.sha256 for game, route in routes.items()
    }
    if completion_trajectory_hashes != expected_trajectory_hashes:
        raise RuntimeError("trusted trajectory source changed during the experiment")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run_id": run_id,
        "completed_at_utc": _utc_now(),
        "provenance_file": "run_started.json",
        "design": design,
        "completion_integrity": {
            "source_manifest_unchanged": True,
            "environment_manifest_unchanged": True,
            "trajectory_hashes_unchanged": True,
            "source_aggregate_sha256": source_manifest["aggregate_sha256"],
            "environment_aggregate_sha256": {
                game: row["aggregate_sha256"]
                for game, row in environment_manifest.items()
            },
            "trajectory_sha256": completion_trajectory_hashes,
        },
        "environment_manifest": environment_manifest,
        "trajectory_manifest": {
            game: _trajectory_manifest(route) for game, route in routes.items()
        },
        "rows": rows,
        "paired_comparisons": _paired_comparisons(rows),
    }
    target = output / "summary.json"
    _write_json_exclusive(target, payload)
    print(f"wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
