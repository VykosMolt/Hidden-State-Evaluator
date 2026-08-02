"""Strict, versioned JSON checkpoints for Hunter-Seeker v2."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    EgoConfig,
    EventKind,
    ExogenousConfig,
    HypothesisConfig,
    LearningConfig,
    MemoryConfig,
    ModelConfig,
    ObjectState,
    Observation,
    Outcome,
    PerceptionConfig,
    PolicyConfig,
    Representation,
    RuntimeMode,
    SearchConfig,
    Topology,
    WorldEvent,
    WorldSnapshot,
)
from .diagnostics import Diagnostics
from .ego import ControlAttribution
from .exogenous import ExogenousChangeFilter
from .hypotheses import RelationalHypothesisEngine
from .learning import CompactLearner, ReplayItem
from .memory import EvidenceStore, StateGraph
from .models import (
    ActionPrior,
    AffordanceModel,
    CompetenceMonitor,
    DynamicsEnsemble,
)
from .perception import PerceptionSystem
from .student import StateConditionedStudentPolicy


CHECKPOINT_VERSION = 2
CHECKPOINT_FORMAT = "compact_hunter_seeker"


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _digest_value(value: Any) -> str:
    """Return a deterministic digest for tensor/config/model state.

    Representation backends often expose only a small descriptive
    ``state_dict`` while holding the actual frozen encoder in a child object.
    This digester lets the checkpoint signature cover those child weights
    without embedding a second copy of them in the JSON checkpoint.
    """

    digest = hashlib.sha256()
    active: set[int] = set()

    def update(item: Any) -> None:
        if item is None:
            digest.update(b"none;")
            return
        if isinstance(item, bool):
            digest.update(b"bool:1;" if item else b"bool:0;")
            return
        if isinstance(item, (int, str, bytes)):
            digest.update(type(item).__name__.encode("utf-8"))
            digest.update(b":")
            payload = item if isinstance(item, bytes) else str(item).encode("utf-8")
            digest.update(str(len(payload)).encode("ascii"))
            digest.update(b":")
            digest.update(payload)
            digest.update(b";")
            return
        if isinstance(item, float):
            digest.update(f"float:{item.hex()};".encode("ascii"))
            return
        if isinstance(item, np.generic):
            update(item.item())
            return

        # Torch/JAX-like tensor objects are reduced to a CPU NumPy array when
        # available.  The import remains optional for the compact baseline.
        if not isinstance(item, np.ndarray):
            detach = getattr(item, "detach", None)
            cpu = getattr(item, "cpu", None)
            numpy_fn = getattr(item, "numpy", None)
            if callable(detach):
                try:
                    candidate = detach()
                    candidate_cpu = getattr(candidate, "cpu", None)
                    if callable(candidate_cpu):
                        candidate = candidate_cpu()
                    candidate_numpy = getattr(candidate, "numpy", None)
                    if callable(candidate_numpy):
                        try:
                            update(np.asarray(candidate_numpy()))
                            return
                        except (TypeError, ValueError, RuntimeError):
                            # Some torch dtypes (notably bfloat16) cannot be
                            # exposed through NumPy.  Hash their typed shape
                            # and raw CPU storage instead.
                            storage_fn = getattr(candidate, "untyped_storage", None)
                            if callable(storage_fn):
                                digest.update(
                                    f"tensor:{getattr(candidate, 'dtype', '')}:"
                                    f"{tuple(int(v) for v in candidate.shape)}:".encode(
                                        "ascii"
                                    )
                                )
                                digest.update(bytes(storage_fn()))
                                digest.update(b";")
                                return
                except (TypeError, ValueError, RuntimeError):
                    pass
            elif callable(cpu) and callable(numpy_fn):
                try:
                    update(np.asarray(cpu().numpy()))
                    return
                except (TypeError, ValueError, RuntimeError):
                    pass

        if isinstance(item, np.ndarray):
            array = np.asarray(item)
            digest.update(b"ndarray:")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(b":")
            digest.update(repr(tuple(int(v) for v in array.shape)).encode("ascii"))
            digest.update(b":")
            if array.dtype.hasobject:
                update(array.tolist())
            else:
                digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(b";")
            return

        identity = id(item)
        if identity in active:
            digest.update(f"cycle:{_qualified_type(item)};".encode("utf-8"))
            return
        active.add(identity)
        try:
            if isinstance(item, Mapping):
                digest.update(b"mapping{")
                ordered = sorted(
                    item.items(),
                    key=lambda row: _digest_value(row[0]),
                )
                for key, value in ordered:
                    update(key)
                    update(value)
                digest.update(b"}")
                return
            if isinstance(item, (list, tuple)):
                digest.update(type(item).__name__.encode("ascii") + b"[")
                for value in item:
                    update(value)
                digest.update(b"]")
                return
            if isinstance(item, (set, frozenset)):
                digest.update(type(item).__name__.encode("ascii") + b"{")
                for child_digest in sorted(_digest_value(value) for value in item):
                    digest.update(child_digest.encode("ascii"))
                digest.update(b"}")
                return
            if is_dataclass(item) and not isinstance(item, type):
                # Match the mapping produced by ``asdict`` without deep-copying
                # opaque leaves.  Tap metadata is deliberately exposed through
                # ``MappingProxyType``, which ``asdict`` cannot pickle, while the
                # fields themselves remain safe to inspect for a deterministic
                # compatibility signature.
                digest.update(b"mapping{")
                ordered_fields = sorted(
                    fields(item),
                    key=lambda row: _digest_value(row.name),
                )
                for row in ordered_fields:
                    update(row.name)
                    update(getattr(item, row.name))
                digest.update(b"}")
                return
            code = getattr(item, "__code__", None)
            if code is None:
                function = getattr(item, "__func__", None)
                code = getattr(function, "__code__", None)
            if code is not None:
                digest.update(
                    f"callable:{getattr(item, '__module__', '')}:"
                    f"{getattr(item, '__qualname__', '')}:".encode("utf-8")
                )
                digest.update(code.co_code)
                update(code.co_consts)
                update(code.co_names)
                update(code.co_varnames)
                update(getattr(item, "__defaults__", None))
                update(getattr(item, "__kwdefaults__", None))
                closure = getattr(item, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            update(cell.cell_contents)
                        except ValueError:
                            digest.update(b"empty-cell;")
                return
            attributes = getattr(item, "__dict__", None)
            if isinstance(attributes, Mapping):
                digest.update(f"object:{_qualified_type(item)}:".encode("utf-8"))
                update(attributes)
                return
            digest.update(f"stateless:{_qualified_type(item)};".encode("utf-8"))
        finally:
            active.remove(identity)

    update(value)
    return digest.hexdigest()


def _component_signature(component: Any) -> dict[str, Any]:
    signature: dict[str, Any] = {"type": _qualified_type(component)}
    callable_target = component if getattr(component, "__code__", None) else getattr(
        type(component),
        "__call__",
        None,
    )
    if callable(callable_target):
        signature["callable"] = {
            "module": str(getattr(callable_target, "__module__", "")),
            "qualname": str(getattr(callable_target, "__qualname__", "")),
            "sha256": _digest_value(callable_target),
        }
    state_fn = getattr(component, "state_dict", None)
    if callable(state_fn):
        signature["state_sha256"] = _digest_value(state_fn())
    elif is_dataclass(component) and not isinstance(component, type):
        signature["state_sha256"] = _digest_value(component)
    else:
        code = getattr(component, "__code__", None)
        if code is not None:
            signature["state_sha256"] = _digest_value(component)
        else:
            call = getattr(component, "__call__", None)
            attributes = getattr(component, "__dict__", None)
            signature["state_sha256"] = _digest_value(
                {
                    "call": call,
                    "attributes": dict(attributes) if isinstance(attributes, Mapping) else {},
                }
            )
    return signature


def _backend_signature(backend: Any) -> dict[str, Any]:
    state_fn = getattr(backend, "state_dict", None)
    declared_state = (
        state_fn()
        if callable(state_fn)
        else {"type": type(backend).__name__}
    )
    components = {
        name: _component_signature(getattr(backend, name))
        for name in ("connector", "extractor", "encoder", "ouro")
        if getattr(backend, name, None) is not None
    }
    return {
        "type": _qualified_type(backend),
        "declared_state_sha256": _digest_value(declared_state),
        "components": components,
    }


def _method_signatures(component: Any, names: tuple[str, ...]) -> dict[str, str]:
    """Hash behavior-bearing methods without invoking them."""

    return {
        name: _digest_value(method)
        for name in names
        if callable(method := getattr(component, name, None))
    }


def _tap_signature(tap: Any, *, methods: tuple[str, ...]) -> dict[str, Any]:
    return {
        "component": _component_signature(tap),
        "methods": _method_signatures(tap, methods),
    }


def _tap_bundle_signature(bundle: Any | None) -> dict[str, Any] | None:
    """Fingerprint every tap that can change candidate scores or retention."""

    if bundle is None:
        return None
    pointwise = tuple(getattr(bundle, "pointwise", ()))
    pairwise = tuple(getattr(bundle, "pairwise", ()))
    survival = getattr(bundle, "survival", None)
    survival_tap = getattr(survival, "tap", None) if survival is not None else None
    return {
        "component": _component_signature(bundle),
        "methods": _method_signatures(
            bundle,
            ("pointwise_values", "apply_pairwise", "retain"),
        ),
        "pointwise": [
            _tap_signature(tap, methods=("read",)) for tap in pointwise
        ],
        "pairwise": [
            _tap_signature(tap, methods=("compare",)) for tap in pairwise
        ],
        "survival": (
            {
                "retainer": _tap_signature(
                    survival,
                    methods=("rank", "retain"),
                ),
                "tap": _tap_signature(survival_tap, methods=("read",)),
            }
            if survival is not None and survival_tap is not None
            else None
        ),
    }


def _executable_registry_signature(registry: Any | None) -> dict[str, Any] | None:
    """Fingerprint verified eligibility plus each executable model's behavior."""

    if registry is None:
        return None
    entries = getattr(registry, "_entries", {})
    models: dict[str, Any] = {}
    if isinstance(entries, Mapping):
        for model_id, entry in sorted(entries.items(), key=lambda row: str(row[0])):
            model = getattr(entry, "model", None)
            verification = getattr(entry, "verification", None)
            models[str(model_id)] = {
                "component": _component_signature(model),
                "methods": _method_signatures(model, ("predict", "plan")),
                "verification_sha256": _digest_value(verification),
            }
    return {
        "component": _component_signature(registry),
        "methods": _method_signatures(
            registry,
            ("ranked_verifications", "predict", "plan"),
        ),
        "models": models,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return 0.0
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
        return _json_safe(value.value)
    return repr(value)


def _checkpoint_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _checkpoint_sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _checkpoint_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _checkpoint_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _checkpoint_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _checkpoint_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _checkpoint_dtype(value: Any, name: str) -> np.dtype[Any]:
    text = _checkpoint_string(value, name)
    try:
        dtype = np.dtype(text)
    except TypeError as exc:
        raise ValueError(f"{name} is not a valid dtype") from exc
    if dtype.hasobject or dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real numeric dtype")
    return dtype


def _checkpoint_array(
    value: Any,
    name: str,
    *,
    dtype: np.dtype[Any],
    ndim: int | None = None,
) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be an array")
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} contains invalid values") from exc
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _checkpoint_metadata(value: Any, name: str) -> dict[str, Any]:
    mapping = _checkpoint_mapping(value, name)
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} keys must be strings")
    return dict(mapping)


def _checkpoint_bbox(value: Any, name: str) -> tuple[int, int, int, int]:
    rows = _checkpoint_sequence(value, name)
    if len(rows) != 4:
        raise ValueError(f"{name} must contain four integers")
    return tuple(
        _checkpoint_int(item, f"{name}[{index}]")
        for index, item in enumerate(rows)
    )  # type: ignore[return-value]


def _checkpoint_action(value: Any, name: str) -> tuple[int, int, int]:
    rows = _checkpoint_sequence(value, name)
    if len(rows) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    action = tuple(
        _checkpoint_int(item, f"{name}[{index}]")
        for index, item in enumerate(rows)
    )
    if action[0] < 0:
        raise ValueError(f"{name} action index must be non-negative")
    return action  # type: ignore[return-value]


def _config_to_state(config: AgentConfig) -> dict[str, Any]:
    state = asdict(config)
    state["runtime_mode"] = config.runtime_mode.value
    return _json_safe(state)


def _config_from_state(state: Mapping[str, Any]) -> AgentConfig:
    state = _checkpoint_mapping(state, "checkpoint config")

    def section(name: str) -> dict[str, Any]:
        return dict(
            _checkpoint_mapping(
                state.get(name, {}),
                f"checkpoint config {name}",
            )
        )

    return AgentConfig(
        runtime_mode=RuntimeMode(state.get("runtime_mode", RuntimeMode.AUTONOMOUS.value)),
        seed=state.get("seed", 0),
        search=SearchConfig(**section("search")),
        policy=PolicyConfig(**section("policy")),
        model=ModelConfig(**section("model")),
        perception=PerceptionConfig(**section("perception")),
        memory=MemoryConfig(**section("memory")),
        learning=LearningConfig(**section("learning")),
        ego=EgoConfig(**section("ego")),
        exogenous=ExogenousConfig(**section("exogenous")),
        hypotheses=HypothesisConfig(**section("hypotheses")),
        enable_online_learning=state.get("enable_online_learning", True),
        enable_executable_models=state.get("enable_executable_models", True),
        strict_finite=state.get("strict_finite", True),
        boundary_bridge_change_fraction=state.get(
            "boundary_bridge_change_fraction",
            0.05,
        ),
    )


def _observation_to_state(observation: Observation) -> dict[str, Any]:
    return {
        "frame": observation.frame.tolist(),
        "dtype": str(observation.frame.dtype),
        "available_actions": list(observation.available_actions),
        "task_id": observation.task_id,
        "stage": observation.stage,
        "progress": observation.progress,
        "metadata": _json_safe(observation.metadata),
    }


def _observation_from_state(state: Mapping[str, Any]) -> Observation:
    state = _checkpoint_mapping(state, "checkpoint observation")
    dtype = _checkpoint_dtype(
        state.get("dtype", "int64"),
        "checkpoint observation dtype",
    )
    raw_actions = _checkpoint_sequence(
        state.get("available_actions", ()),
        "checkpoint observation available_actions",
    )
    actions = tuple(
        _checkpoint_int(
            value,
            f"checkpoint observation available_actions[{index}]",
            minimum=0,
        )
        for index, value in enumerate(raw_actions)
    )
    if len(actions) != len(set(actions)):
        raise ValueError(
            "checkpoint observation available_actions must be unique"
        )
    return Observation(
        frame=_checkpoint_array(
            state.get("frame"),
            "checkpoint observation frame",
            dtype=dtype,
            ndim=2,
        ),
        available_actions=actions,
        task_id=_checkpoint_string(
            state.get("task_id"),
            "checkpoint observation task_id",
        ),
        stage=_checkpoint_int(
            state.get("stage", 1),
            "checkpoint observation stage",
            minimum=1,
        ),
        progress=_checkpoint_float(
            state.get("progress", 0.0),
            "checkpoint observation progress",
        ),
        metadata=_checkpoint_metadata(
            state.get("metadata", {}),
            "checkpoint observation metadata",
        ),
    )


def _snapshot_to_state(snapshot: WorldSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "observation": _observation_to_state(snapshot.observation),
        "objects": [asdict(obj) for obj in snapshot.objects],
        "events": [
            {
                "kind": event.kind.value,
                "subject_track_id": event.subject_track_id,
                "object_signature": event.object_signature,
                "magnitude": event.magnitude,
                "metadata": _json_safe(event.metadata),
            }
            for event in snapshot.events
        ],
        "topology": asdict(snapshot.topology),
        "representation": {
            "global_vector": snapshot.representation.global_vector.tolist(),
            "spatial": snapshot.representation.spatial.tolist(),
            "taps": {
                name: value.tolist()
                for name, value in snapshot.representation.taps.items()
            },
        },
        "step": snapshot.step,
        "memory_state_id": snapshot.memory_state_id,
    }


def _snapshot_from_state(state: Mapping[str, Any] | None) -> WorldSnapshot | None:
    if state is None:
        return None
    state = _checkpoint_mapping(state, "checkpoint snapshot")
    topology_raw = _checkpoint_mapping(
        state.get("topology"),
        "checkpoint snapshot topology",
    )
    raw_adjacencies = _checkpoint_sequence(
        topology_raw.get("object_adjacencies", ()),
        "checkpoint snapshot topology object_adjacencies",
    )
    adjacencies: list[tuple[int, int]] = []
    for index, raw_pair in enumerate(raw_adjacencies):
        pair = _checkpoint_sequence(
            raw_pair,
            f"checkpoint snapshot topology object_adjacencies[{index}]",
        )
        if len(pair) != 2:
            raise ValueError(
                "checkpoint snapshot topology adjacency must contain two IDs"
            )
        adjacencies.append(
            (
                _checkpoint_int(
                    pair[0],
                    f"checkpoint snapshot topology adjacency[{index}][0]",
                    minimum=0,
                ),
                _checkpoint_int(
                    pair[1],
                    f"checkpoint snapshot topology adjacency[{index}][1]",
                    minimum=0,
                ),
            )
        )
    raw_reachable = _checkpoint_sequence(
        topology_raw.get("reachable_object_ids", ()),
        "checkpoint snapshot topology reachable_object_ids",
    )
    reachable = tuple(
        _checkpoint_int(
            value,
            f"checkpoint snapshot topology reachable_object_ids[{index}]",
            minimum=0,
        )
        for index, value in enumerate(raw_reachable)
    )
    if len(reachable) != len(set(reachable)):
        raise ValueError(
            "checkpoint snapshot topology reachable_object_ids must be unique"
        )
    topology = Topology(
        component_count=_checkpoint_int(
            topology_raw.get("component_count", 0),
            "checkpoint snapshot topology component_count",
            minimum=0,
        ),
        largest_component_fraction=_checkpoint_float(
            topology_raw.get("largest_component_fraction", 0.0),
            "checkpoint snapshot topology largest_component_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        frontier_fraction=_checkpoint_float(
            topology_raw.get("frontier_fraction", 0.0),
            "checkpoint snapshot topology frontier_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        object_adjacencies=tuple(adjacencies),
        reachable_object_ids=reachable,
    )

    raw_objects = _checkpoint_sequence(
        state.get("objects", ()),
        "checkpoint snapshot objects",
    )
    objects: list[ObjectState] = []
    for index, raw_object in enumerate(raw_objects):
        row = _checkpoint_mapping(
            raw_object,
            f"checkpoint snapshot objects[{index}]",
        )
        objects.append(
            ObjectState(
                object_id=_checkpoint_int(
                    row.get("object_id"),
                    f"checkpoint snapshot objects[{index}].object_id",
                    minimum=0,
                ),
                track_id=_checkpoint_int(
                    row.get("track_id"),
                    f"checkpoint snapshot objects[{index}].track_id",
                    minimum=0,
                ),
                value=_checkpoint_int(
                    row.get("value"),
                    f"checkpoint snapshot objects[{index}].value",
                ),
                area=_checkpoint_int(
                    row.get("area"),
                    f"checkpoint snapshot objects[{index}].area",
                    minimum=1,
                ),
                centroid_x=_checkpoint_float(
                    row.get("centroid_x"),
                    f"checkpoint snapshot objects[{index}].centroid_x",
                ),
                centroid_y=_checkpoint_float(
                    row.get("centroid_y"),
                    f"checkpoint snapshot objects[{index}].centroid_y",
                ),
                bbox=_checkpoint_bbox(
                    row.get("bbox"),
                    f"checkpoint snapshot objects[{index}].bbox",
                ),
                touches_border=_checkpoint_bool(
                    row.get("touches_border", False),
                    f"checkpoint snapshot objects[{index}].touches_border",
                ),
                velocity_x=_checkpoint_float(
                    row.get("velocity_x", 0.0),
                    f"checkpoint snapshot objects[{index}].velocity_x",
                ),
                velocity_y=_checkpoint_float(
                    row.get("velocity_y", 0.0),
                    f"checkpoint snapshot objects[{index}].velocity_y",
                ),
                controllable=_checkpoint_float(
                    row.get("controllable", 0.0),
                    f"checkpoint snapshot objects[{index}].controllable",
                ),
                hazard=_checkpoint_float(
                    row.get("hazard", 0.0),
                    f"checkpoint snapshot objects[{index}].hazard",
                ),
                rewarding=_checkpoint_float(
                    row.get("rewarding", 0.0),
                    f"checkpoint snapshot objects[{index}].rewarding",
                ),
                confidence=_checkpoint_float(
                    row.get("confidence", 1.0),
                    f"checkpoint snapshot objects[{index}].confidence",
                ),
                signature=_checkpoint_string(
                    row.get("signature", ""),
                    f"checkpoint snapshot objects[{index}].signature",
                ),
            )
        )

    raw_events = _checkpoint_sequence(
        state.get("events", ()),
        "checkpoint snapshot events",
    )
    events: list[WorldEvent] = []
    for index, raw_event in enumerate(raw_events):
        row = _checkpoint_mapping(
            raw_event,
            f"checkpoint snapshot events[{index}]",
        )
        kind_text = _checkpoint_string(
            row.get("kind"),
            f"checkpoint snapshot events[{index}].kind",
        )
        try:
            kind = EventKind(kind_text)
        except ValueError as exc:
            raise ValueError(
                f"checkpoint snapshot events[{index}].kind is invalid"
            ) from exc
        events.append(
            WorldEvent(
                kind=kind,
                subject_track_id=_checkpoint_int(
                    row.get("subject_track_id", -1),
                    (
                        f"checkpoint snapshot events[{index}]"
                        ".subject_track_id"
                    ),
                    minimum=-1,
                ),
                object_signature=_checkpoint_string(
                    row.get("object_signature", ""),
                    (
                        f"checkpoint snapshot events[{index}]"
                        ".object_signature"
                    ),
                ),
                magnitude=_checkpoint_float(
                    row.get("magnitude", 0.0),
                    f"checkpoint snapshot events[{index}].magnitude",
                ),
                metadata=_checkpoint_metadata(
                    row.get("metadata", {}),
                    f"checkpoint snapshot events[{index}].metadata",
                ),
            )
        )

    representation_raw = _checkpoint_mapping(
        state.get("representation"),
        "checkpoint snapshot representation",
    )
    representation_dtype = np.dtype("float32")
    raw_taps = _checkpoint_mapping(
        representation_raw.get("taps", {}),
        "checkpoint snapshot representation taps",
    )
    taps: dict[str, np.ndarray] = {}
    for name, value in raw_taps.items():
        tap_name = _checkpoint_string(
            name,
            "checkpoint snapshot representation tap name",
        )
        taps[tap_name] = _checkpoint_array(
            value,
            f"checkpoint snapshot representation taps[{tap_name!r}]",
            dtype=representation_dtype,
        )
    return WorldSnapshot(
        observation=_observation_from_state(
            _checkpoint_mapping(
                state.get("observation"),
                "checkpoint snapshot observation",
            )
        ),
        objects=tuple(objects),
        events=tuple(events),
        topology=topology,
        representation=Representation(
            global_vector=_checkpoint_array(
                representation_raw.get("global_vector"),
                "checkpoint snapshot representation global_vector",
                dtype=representation_dtype,
            ),
            spatial=_checkpoint_array(
                representation_raw.get("spatial"),
                "checkpoint snapshot representation spatial",
                dtype=representation_dtype,
            ),
            taps=taps,
        ),
        step=_checkpoint_int(
            state.get("step", 0),
            "checkpoint snapshot step",
            minimum=0,
        ),
        memory_state_id=_checkpoint_string(
            state.get("memory_state_id", ""),
            "checkpoint snapshot memory_state_id",
        ),
    )


def _replay_item_to_state(item: ReplayItem) -> dict[str, Any]:
    transition = item.transition
    return {
        "before": _snapshot_to_state(item.before),
        "after": _snapshot_to_state(item.after),
        "transition": {
            "transition_id": transition.transition_id,
            "decision_id": transition.decision_id,
            "task_id": transition.task_id,
            "stage": transition.stage,
            "step": transition.step,
            "action": list(transition.action.key),
            "outcome": {
                "reward": transition.outcome.reward,
                "progress_delta": transition.outcome.progress_delta,
                "terminated": transition.outcome.terminated,
                "truncated": transition.outcome.truncated,
                "boundary": transition.outcome.boundary.value,
                "hazard": transition.outcome.hazard,
                "metadata": _json_safe(transition.outcome.metadata),
            },
            "frame_changed": transition.frame_changed,
            "after_state_id": transition.after_state_id,
        },
        "target_object_signature": item.target_object_signature,
        "source": item.source,
        "teacher": item.teacher,
    }


def _replay_item_from_state(state: Mapping[str, Any]) -> ReplayItem:
    state = _checkpoint_mapping(state, "checkpoint replay item")
    before = _snapshot_from_state(state.get("before"))
    after = _snapshot_from_state(state.get("after"))
    if before is None or after is None:
        raise ValueError("replay item snapshots cannot be null")
    raw = _checkpoint_mapping(
        state.get("transition"),
        "checkpoint replay transition",
    )
    outcome_raw = _checkpoint_mapping(
        raw.get("outcome"),
        "checkpoint replay outcome",
    )
    action_values = _checkpoint_action(
        raw.get("action"),
        "checkpoint replay action",
    )
    action = Action(*action_values)
    boundary_text = _checkpoint_string(
        outcome_raw.get("boundary", BoundaryKind.NONE.value),
        "checkpoint replay outcome boundary",
    )
    try:
        boundary = BoundaryKind(boundary_text)
    except ValueError as exc:
        raise ValueError("checkpoint replay outcome boundary is invalid") from exc
    terminated = _checkpoint_bool(
        outcome_raw.get("terminated", False),
        "checkpoint replay outcome terminated",
    )
    truncated = _checkpoint_bool(
        outcome_raw.get("truncated", False),
        "checkpoint replay outcome truncated",
    )
    outcome = Outcome(
        reward=_checkpoint_float(
            outcome_raw.get("reward", 0.0),
            "checkpoint replay outcome reward",
        ),
        progress_delta=_checkpoint_float(
            outcome_raw.get("progress_delta", 0.0),
            "checkpoint replay outcome progress_delta",
        ),
        terminated=terminated,
        truncated=truncated,
        boundary=boundary,
        hazard=_checkpoint_float(
            outcome_raw.get("hazard", 0.0),
            "checkpoint replay outcome hazard",
            minimum=0.0,
            maximum=1.0,
        ),
        metadata=_checkpoint_metadata(
            outcome_raw.get("metadata", {}),
            "checkpoint replay outcome metadata",
        ),
    )
    if outcome.terminated != terminated or outcome.truncated != truncated:
        raise ValueError(
            "checkpoint replay outcome flags conflict with its boundary"
        )
    from .contracts import Transition

    transition = Transition(
        transition_id=_checkpoint_string(
            raw.get("transition_id"),
            "checkpoint replay transition_id",
        ),
        decision_id=_checkpoint_string(
            raw.get("decision_id"),
            "checkpoint replay decision_id",
        ),
        task_id=_checkpoint_string(
            raw.get("task_id"),
            "checkpoint replay task_id",
        ),
        stage=_checkpoint_int(
            raw.get("stage"),
            "checkpoint replay stage",
            minimum=1,
        ),
        step=_checkpoint_int(
            raw.get("step"),
            "checkpoint replay step",
            minimum=0,
        ),
        before=before,
        action=action,
        after_observation=after.observation,
        outcome=outcome,
        frame_changed=_checkpoint_bool(
            raw.get("frame_changed", False),
            "checkpoint replay frame_changed",
        ),
        after_state_id=_checkpoint_string(
            raw.get("after_state_id"),
            "checkpoint replay after_state_id",
        ),
    )
    if transition.task_id != before.observation.task_id:
        raise ValueError("checkpoint replay task does not match before snapshot")
    if transition.task_id != after.observation.task_id:
        raise ValueError("checkpoint replay task does not match after snapshot")
    if transition.stage != before.observation.stage:
        raise ValueError("checkpoint replay stage does not match before snapshot")
    if transition.step != before.step:
        raise ValueError("checkpoint replay step does not match before snapshot")
    if transition.after_state_id != after.state_id:
        raise ValueError(
            "checkpoint replay after_state_id does not match after snapshot"
        )
    actual_frame_changed = not np.array_equal(
        before.observation.frame,
        after.observation.frame,
    )
    if transition.frame_changed != actual_frame_changed:
        raise ValueError(
            "checkpoint replay frame_changed does not match its snapshots"
        )
    return ReplayItem(
        before=before,
        after=after,
        transition=transition,
        target_object_signature=_checkpoint_string(
            state.get("target_object_signature", ""),
            "checkpoint replay target_object_signature",
        ),
        source=_checkpoint_string(
            state.get("source", "online"),
            "checkpoint replay source",
        ),
        teacher=_checkpoint_bool(
            state.get("teacher", False),
            "checkpoint replay teacher",
        ),
    )


def agent_state_dict(agent: Any, *, include_diagnostics: bool = True) -> dict[str, Any]:
    if agent.pending_decision is not None:
        raise RuntimeError("cannot checkpoint an unobserved decision")
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "agent_id": agent.agent_id,
        "config": _config_to_state(agent.config),
        "signature": {
            "latent_dim": int(agent.dynamics.latent_dim),
            "click_action_index": agent._click_action_index,
            "backend": _backend_signature(agent.representation_backend),
            "tap_bundle": _tap_bundle_signature(agent._tap_bundle),
            "executable_registry": _executable_registry_signature(
                agent.executable_registry
                if agent.config.enable_executable_models
                else None
            ),
        },
        "models": {
            "dynamics": agent.dynamics.state_dict(),
            "prior": agent.prior.state_dict(),
            "affordances": agent.affordances.state_dict(),
            "competence": agent.competence.state_dict(),
            "ego": agent.ego.state_dict(),
            "student_policy": agent.student_policy.state_dict(),
        },
        "knowledge": {
            "graph": agent.graph.export_state(),
            "evidence": agent.evidence.export_state(),
            "perception": agent.perception.export_state(),
            "exogenous": agent.exogenous.state_dict(),
            "hypotheses": agent.hypotheses.state_dict(),
        },
        "learning": {
            "real_updates": agent.learner.real_updates,
            "replay_updates": agent.learner.replay_updates,
            "last_replay_loss": agent.learner.last_replay_loss,
            "items": [
                _replay_item_to_state(item)
                for item in agent.learner.replay.items
            ],
            "rng_state": _json_safe(agent.learner._rng.bit_generator.state),
        },
        "runtime": {
            "task_id": agent._task_id,
            "step": agent._step,
            "decision_counter": agent._decision_counter,
            "current_snapshot": _snapshot_to_state(agent.current_snapshot),
            "run_active": agent._run_active,
            "awaiting_visual_reset": bool(
                getattr(agent, "_awaiting_visual_reset", False)
            ),
            "run_transition_count": agent._run_transition_count,
            "total_transition_count": agent._total_transition_count,
            "committed_decision_ids": sorted(agent._committed_decision_ids),
            "rng_state": _json_safe(agent._rng.bit_generator.state),
        },
        "diagnostics": (
            agent.diagnostics.export_state() if include_diagnostics else None
        ),
    }


def _validate_signature(
    agent: Any,
    state: Mapping[str, Any],
    *,
    validate_runtime_dependencies: bool,
) -> None:
    signature = _checkpoint_mapping(
        state.get("signature"),
        "checkpoint signature",
    )
    latent_dim = _checkpoint_int(
        signature.get("latent_dim", -1),
        "checkpoint signature latent_dim",
        minimum=1,
    )
    if latent_dim != int(agent.config.model.latent_dim):
        raise ValueError(
            f"checkpoint latent_dim {latent_dim} does not match "
            f"agent latent_dim {agent.config.model.latent_dim}"
        )
    checkpoint_click = signature.get("click_action_index")
    if checkpoint_click is not None:
        checkpoint_click = _checkpoint_int(
            checkpoint_click,
            "checkpoint signature click_action_index",
            minimum=0,
        )
    if checkpoint_click != agent._click_action_index:
        raise ValueError(
            "checkpoint click-action signature does not match the agent adapter"
        )
    checkpoint_backend = signature.get("backend", {})
    current_backend = _backend_signature(agent.representation_backend)
    if not isinstance(checkpoint_backend, Mapping) or dict(checkpoint_backend) != current_backend:
        checkpoint_type = (
            str(checkpoint_backend.get("type", ""))
            if isinstance(checkpoint_backend, Mapping)
            else ""
        )
        raise ValueError(
            f"checkpoint representation backend {checkpoint_type!r} does not "
            f"match agent backend {current_backend['type']!r}"
        )
    if not validate_runtime_dependencies:
        return
    if "tap_bundle" not in signature:
        raise ValueError("checkpoint is missing the tap-bundle compatibility signature")
    checkpoint_taps = signature.get("tap_bundle")
    current_taps = _tap_bundle_signature(agent._tap_bundle)
    if checkpoint_taps != current_taps:
        raise ValueError("checkpoint tap bundle does not match the agent tap bundle")
    if "executable_registry" not in signature:
        raise ValueError(
            "checkpoint is missing the executable-registry compatibility signature"
        )
    checkpoint_registry = signature.get("executable_registry")
    current_registry = _executable_registry_signature(
        agent.executable_registry
        if agent.config.enable_executable_models
        else None
    )
    if checkpoint_registry != current_registry:
        raise ValueError(
            "checkpoint executable registry does not match the agent registry"
        )


def load_agent_state(
    agent: Any,
    state: Mapping[str, Any],
    *,
    reset_optimizer: bool = False,
    weights_only: bool = False,
) -> None:
    del reset_optimizer  # v2's compact online learner has no optimizer moments.
    if agent.pending_decision is not None:
        raise RuntimeError(
            "cannot load a checkpoint while an action decision is pending"
        )
    state = _checkpoint_mapping(state, "checkpoint")
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"not a {CHECKPOINT_FORMAT!r} checkpoint; legacy checkpoints require "
            "an explicit importer"
        )
    version = _checkpoint_int(
        state.get("version", -1),
        "checkpoint version",
        minimum=0,
    )
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {version}; expected {CHECKPOINT_VERSION}"
        )
    _validate_signature(
        agent,
        state,
        validate_runtime_dependencies=not weights_only,
    )
    checkpoint_config = _config_from_state(
        _checkpoint_mapping(state.get("config"), "checkpoint config")
    )
    if checkpoint_config != agent.config:
        raise ValueError(
            "checkpoint agent configuration does not match the destination agent"
        )

    raw_models = state.get("models")
    if not isinstance(raw_models, Mapping):
        raise ValueError("checkpoint models must be a mapping")
    models = dict(raw_models)

    def model_state(name: str) -> dict[str, Any]:
        raw = models.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"checkpoint model {name!r} must be a mapping")
        return dict(raw)

    # Construct and validate every replacement before touching the live agent.
    new_dynamics = DynamicsEnsemble.from_state(
        model_state("dynamics"),
        seed=agent.config.seed,
    )
    if new_dynamics.config != agent.config.model:
        raise ValueError("checkpoint dynamics configuration does not match the agent")
    new_prior = ActionPrior.from_state(model_state("prior"))
    new_affordances = AffordanceModel.from_state(
        model_state("affordances")
    )
    new_competence = CompetenceMonitor.from_state(
        model_state("competence")
    )
    new_ego = ControlAttribution.from_state(
        model_state("ego"),
        config=agent.config.ego,
    )
    raw_student = models.get("student_policy")
    if "student_policy" not in models:
        # Same-version checkpoints written before the student policy was
        # introduced have no student payload.
        new_student_policy = StateConditionedStudentPolicy(
            agent.student_policy.config
        )
    elif isinstance(raw_student, Mapping):
        new_student_policy = StateConditionedStudentPolicy.from_state(
            dict(raw_student),
            config=agent.student_policy.config,
        )
    else:
        raise ValueError(
            "checkpoint model 'student_policy' must be a mapping"
        )

    if weights_only:
        # Loading weights into a live destination must preserve that
        # destination's graph, evidence, tracking, replay, diagnostics, and
        # transaction/runtime state.  Rewire the existing learner so future
        # updates reach the newly loaded model objects.
        original = {
            "dynamics": agent.dynamics,
            "prior": agent.prior,
            "affordances": agent.affordances,
            "competence": agent.competence,
            "ego": agent.ego,
            "student_policy": agent.student_policy,
            "search_engine": agent.search_engine,
            "learner_dynamics": agent.learner.dynamics,
            "learner_prior": agent.learner.prior,
            "learner_affordances": agent.learner.affordances,
        }
        try:
            agent.dynamics = new_dynamics
            agent.prior = new_prior
            agent.affordances = new_affordances
            agent.competence = new_competence
            agent.ego = new_ego
            agent.student_policy = new_student_policy
            agent.learner.dynamics = new_dynamics
            agent.learner.prior = new_prior
            agent.learner.affordances = new_affordances
            agent.search_engine = agent._new_search_engine()
        except Exception:
            agent.dynamics = original["dynamics"]
            agent.prior = original["prior"]
            agent.affordances = original["affordances"]
            agent.competence = original["competence"]
            agent.ego = original["ego"]
            agent.student_policy = original["student_policy"]
            agent.search_engine = original["search_engine"]
            agent.learner.dynamics = original["learner_dynamics"]
            agent.learner.prior = original["learner_prior"]
            agent.learner.affordances = original["learner_affordances"]
            raise
        return

    raw_knowledge = state.get("knowledge")
    if not isinstance(raw_knowledge, Mapping):
        raise ValueError("checkpoint knowledge must be a mapping")
    knowledge = dict(raw_knowledge)

    def knowledge_state(name: str) -> dict[str, Any]:
        raw = knowledge.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"checkpoint knowledge {name!r} must be a mapping")
        return dict(raw)

    new_graph = StateGraph.from_state(knowledge_state("graph"))
    new_evidence = EvidenceStore.from_state(knowledge_state("evidence"))
    if new_evidence.config != agent.config.memory:
        raise ValueError("checkpoint evidence configuration does not match the agent")
    new_perception = PerceptionSystem.from_state(knowledge_state("perception"))
    if new_perception.config != agent.config.perception:
        raise ValueError("checkpoint perception configuration does not match the agent")
    new_exogenous = ExogenousChangeFilter.from_state(
        knowledge_state("exogenous"),
        config=agent.config.exogenous,
    )
    new_hypotheses = RelationalHypothesisEngine.from_state(
        knowledge_state("hypotheses"),
        config=agent.config.hypotheses,
    )

    raw_learning = state.get("learning")
    if not isinstance(raw_learning, Mapping):
        raise ValueError("checkpoint learning state must be a mapping")
    learning_state = dict(raw_learning)
    raw_items = learning_state.get("items", ())
    if not isinstance(raw_items, (list, tuple)):
        raise ValueError("checkpoint replay items must be a sequence")
    if len(raw_items) > int(agent.config.learning.replay_capacity):
        raise ValueError("checkpoint replay exceeds the configured capacity")
    replay_items = tuple(_replay_item_from_state(row) for row in raw_items)
    new_learner = CompactLearner(
        dynamics=new_dynamics,
        prior=new_prior,
        affordances=new_affordances,
        config=agent.config.learning,
        seed=agent.config.seed + 1,
    )
    for item in replay_items:
        new_learner.replay.push(item)
    new_learner.real_updates = _checkpoint_int(
        learning_state.get("real_updates", 0),
        "checkpoint learner real_updates",
        minimum=0,
    )
    new_learner.replay_updates = _checkpoint_int(
        learning_state.get("replay_updates", 0),
        "checkpoint learner replay_updates",
        minimum=0,
    )
    new_learner.last_replay_loss = _checkpoint_float(
        learning_state.get("last_replay_loss", 0.0),
        "checkpoint learner last_replay_loss",
    )
    learner_rng = learning_state.get("rng_state")
    if not isinstance(learner_rng, Mapping):
        raise ValueError("checkpoint learner RNG state is missing")
    new_learner._rng.bit_generator.state = dict(learner_rng)

    raw_runtime = state.get("runtime")
    if not isinstance(raw_runtime, Mapping):
        raise ValueError("checkpoint runtime state must be a mapping")
    runtime = dict(raw_runtime)
    new_snapshot = _snapshot_from_state(runtime.get("current_snapshot"))
    task_id = runtime.get("task_id")
    if task_id is not None and not isinstance(task_id, str):
        raise ValueError("checkpoint runtime task_id must be a string or null")
    new_task_id = task_id
    new_step = _checkpoint_int(
        runtime.get("step", 0),
        "checkpoint runtime step",
        minimum=0,
    )
    new_decision_counter = _checkpoint_int(
        runtime.get("decision_counter", 0),
        "checkpoint runtime decision_counter",
        minimum=0,
    )
    new_run_transition_count = _checkpoint_int(
        runtime.get("run_transition_count", 0),
        "checkpoint runtime run_transition_count",
        minimum=0,
    )
    new_total_transition_count = _checkpoint_int(
        runtime.get("total_transition_count", 0),
        "checkpoint runtime total_transition_count",
        minimum=0,
    )
    if new_run_transition_count > new_total_transition_count:
        raise ValueError("checkpoint run transition count exceeds total count")
    raw_run_active = runtime.get("run_active", False)
    new_run_active = _checkpoint_bool(
        raw_run_active,
        "checkpoint runtime run_active",
    )
    raw_awaiting_visual_reset = runtime.get("awaiting_visual_reset", False)
    new_awaiting_visual_reset = _checkpoint_bool(
        raw_awaiting_visual_reset,
        "checkpoint runtime awaiting_visual_reset",
    )
    if new_snapshot is not None:
        snapshot_task = new_snapshot.observation.task_id
        if new_task_id is None or snapshot_task != new_task_id:
            raise ValueError("checkpoint snapshot task does not match runtime task")
        if int(new_snapshot.step) != new_step:
            raise ValueError("checkpoint snapshot step does not match runtime step")
    if new_run_active and new_task_id is None:
        raise ValueError("active checkpoint run requires a runtime task_id")
    if new_snapshot is None and (
        new_step != 0 or new_run_transition_count != 0
    ):
        raise ValueError(
            "checkpoint without a current snapshot cannot contain in-run progress"
        )
    if new_awaiting_visual_reset and new_snapshot is None:
        raise ValueError(
            "checkpoint awaiting a visual reset requires a current snapshot"
        )
    raw_committed_ids = runtime.get("committed_decision_ids", ())
    if (
        isinstance(raw_committed_ids, (str, bytes))
        or not isinstance(raw_committed_ids, (list, tuple, set, frozenset))
    ):
        raise ValueError(
            "checkpoint committed decision IDs must be a sequence"
        )
    committed_rows = tuple(
        _checkpoint_string(
            value,
            f"checkpoint committed decision IDs[{index}]",
        )
        for index, value in enumerate(raw_committed_ids)
    )
    if len(set(committed_rows)) != len(committed_rows):
        raise ValueError("checkpoint committed decision IDs must be unique")
    new_committed_ids = set(committed_rows)
    if new_task_id is None and (
        new_snapshot is not None
        or new_run_active
        or new_step != 0
        or new_decision_counter != 0
        or new_run_transition_count != 0
        or new_total_transition_count != 0
        or new_committed_ids
    ):
        raise ValueError(
            "checkpoint without a runtime task_id cannot contain runtime progress"
        )
    rng_state = runtime.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise ValueError("checkpoint runtime RNG state is missing")
    new_rng = np.random.default_rng(int(agent.config.seed))
    new_rng.bit_generator.state = dict(rng_state)
    diagnostics_state = state.get("diagnostics")
    if diagnostics_state is None:
        new_diagnostics = Diagnostics()
    elif isinstance(diagnostics_state, Mapping):
        new_diagnostics = Diagnostics.from_state(diagnostics_state)
    else:
        raise ValueError("checkpoint diagnostics must be a mapping or null")

    agent_id = _checkpoint_string(
        state.get("agent_id", agent.agent_id),
        "checkpoint agent_id",
    )

    replacements = {
        "agent_id": agent_id,
        "dynamics": new_dynamics,
        "prior": new_prior,
        "affordances": new_affordances,
        "competence": new_competence,
        "ego": new_ego,
        "student_policy": new_student_policy,
        "graph": new_graph,
        "evidence": new_evidence,
        "perception": new_perception,
        "exogenous": new_exogenous,
        "hypotheses": new_hypotheses,
        "learner": new_learner,
        "buffer": new_learner.replay,
        "diagnostics": new_diagnostics,
        "_rng": new_rng,
        "_task_id": new_task_id,
        "_step": new_step,
        "_decision_counter": new_decision_counter,
        "_current_snapshot": new_snapshot,
        "_pending": None,
        "_last_transition": None,
        "_run_active": new_run_active,
        "_awaiting_visual_reset": new_awaiting_visual_reset,
        "_run_transition_count": new_run_transition_count,
        "_total_transition_count": new_total_transition_count,
        "_committed_decision_ids": new_committed_ids,
        "_resume_ready": new_snapshot is not None,
    }
    originals = {
        name: getattr(agent, name)
        for name in (*replacements, "search_engine")
    }
    try:
        for name, value in replacements.items():
            setattr(agent, name, value)
        agent.search_engine = agent._new_search_engine()
    except Exception:
        for name, value in originals.items():
            setattr(agent, name, value)
        raise


def save_checkpoint(
    agent: Any,
    path: str | Path,
    *,
    include_diagnostics: bool = True,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = agent_state_dict(agent, include_diagnostics=include_diagnostics)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(state), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_checkpoint(
    agent: Any,
    path: str | Path,
    *,
    reset_optimizer: bool = False,
    weights_only: bool = False,
) -> None:
    source = Path(path)
    state = json.loads(source.read_text(encoding="utf-8"))
    load_agent_state(
        agent,
        state,
        reset_optimizer=reset_optimizer,
        weights_only=weights_only,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "agent_state_dict",
    "load_agent_state",
    "load_checkpoint",
    "save_checkpoint",
]
