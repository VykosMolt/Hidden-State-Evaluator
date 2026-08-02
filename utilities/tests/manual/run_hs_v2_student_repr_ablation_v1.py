"""Gate-4 named experiment: GridFeature vs frozen-Ouro student features.

The four-arm attribution run established that the trained student head is
the entire retention mechanism, which reduces both open retention failures
to representation questions: trusted ``tr87`` states collapse to latent
distances of ~0.026 in the default 96-value GridFeature encoding, and
``wa30``'s 1,556 route states exceed what five linear rows over those 96
values can separate (offline fit ~42% against a 98.7% memoryless Bayes
ceiling).  This experiment asks whether the frozen Ouro loop-state backend
(`OuroLoopRepresentationBackend`: GridEncoder tokens -> frozen
Ouro-2.6B-Thinking -> early/middle/late loop taps -> the same 32+8x8
Representation shape) supplies more separable state features to the
identical student head.

Offline protocol, per game and backend: build a student-mode agent whose
only difference is ``representation_backend``, distill the trusted route
through the ordinary ``distill_teacher`` path, then score every recorded
state through the same encode path and report argmax agreement (C1) with
the demonstrated action, margins, the fitted per-task bandwidth, and the
latent geometry that drives it.  Both backends produce 96 state values, so
the head's random-Fourier map and row structure are identical; the ablation
varies feature content only.  Compute is intentionally unmatched (a 2.6B
frozen forward versus a hand-rolled feature map) and reported per encode;
a live retention comparison is a separate follow-up gated on this result.

The backbone stays frozen (no gradients, no training); action selection is
never involved (offline scoring only).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

import run_hs_v2_student_retention_v1 as retention

from hunter_seeker_v2.adapters import ArcActionAdapter
from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    Observation,
    PolicyConfig,
    RuntimeMode,
    WorldSnapshot,
)
from hunter_seeker_v2.perception import PerceptionSystem
from hunter_seeker_v2.representation import OuroLoopRepresentationBackend
from hunter_seeker_v2.teacher import TeacherAccessGuard, TeacherQuery, TrajectoryTeacher

EXPERIMENT_ID = "hs_v2_student_repr_ablation_v1"
SCHEMA_VERSION = 1
BACKENDS = ("gridfeature", "ouro_loop")
ENCODER_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "running" / "sprint4_encoder_reverted.pt"
)
OURO_MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"


class _CastingEncoder:
    """Cast encoder tokens to the backbone dtype; freeze semantics unchanged."""

    def __init__(self, encoder: Any, dtype: Any) -> None:
        self.encoder = encoder
        self.dtype = dtype

    def encode_for_ouro(self, grid: Any) -> tuple[Any, Any, Any]:
        tokens, mask, patch_grid = self.encoder.encode_for_ouro(grid)
        return tokens.to(self.dtype), mask, patch_grid

    def eval(self) -> None:
        self.encoder.eval()

    def parameters(self):
        return self.encoder.parameters()


def _load_ouro_backend(device: str) -> tuple[OuroLoopRepresentationBackend, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    from hunter_seeker_core.grid_encoder import GridEncoder

    encoder = GridEncoder(n_values=16, patch_size=4, d_model=2048).to(device)
    checkpoint = torch.load(ENCODER_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("encoder", checkpoint)
    load_result = encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    model = AutoModelForCausalLM.from_pretrained(
        OURO_MODEL_ID,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    backend = OuroLoopRepresentationBackend(
        _CastingEncoder(encoder, torch.bfloat16),
        model,
        device=device,
    )
    manifest = {
        "model_id": OURO_MODEL_ID,
        "encoder_checkpoint": str(ENCODER_CHECKPOINT.relative_to(PROJECT_ROOT)),
        "encoder_checkpoint_sha256": retention._sha256_file(ENCODER_CHECKPOINT),
        "encoder_missing_keys": list(load_result.missing_keys),
        "encoder_unexpected_keys": list(load_result.unexpected_keys),
        "backend_state": backend.state_dict(),
        "dtype": "bfloat16",
        "device": device,
        "frozen": True,
    }
    return backend, manifest


def _evaluate_backend(
    backend_name: str,
    *,
    game: str,
    route: retention.TrustedTrajectory,
    backend: Any,
) -> dict[str, Any]:
    adapter = ArcActionAdapter()
    teacher = TrajectoryTeacher.from_npz(
        str(route.path),
        task_id=game,
        teacher_id=f"trusted:{game}:{route.sha256[:12]}",
    )
    agent = CompactHunterSeeker(
        config=AgentConfig(
            runtime_mode=RuntimeMode.STUDENT,
            seed=0,
            policy=PolicyConfig(exploration_epsilon=0.0),
        ),
        representation_backend=backend,
        click_action_index=adapter.click_action_index(),
        safe_action_provider=adapter.safe_action_indices,
        teacher=TeacherAccessGuard(teacher, mode=RuntimeMode.STUDENT),
    )
    distill_started = time.monotonic()
    distilled = agent.distill_teacher(
        TeacherQuery(task_id=game, limit=max(len(route), 1))
    )
    distill_elapsed = time.monotonic() - distill_started
    if distilled != len(route):
        raise AssertionError(
            f"{backend_name}/{game}: distilled {distilled}, expected {len(route)}"
        )
    legal_indices = tuple(
        sorted({int(action.index) for action in route.actions})
    )
    legal = tuple(Action(index) for index in legal_indices)
    agree = 0
    margins: list[float] = []
    encode_seconds: list[float] = []
    state_values: list[np.ndarray] = []
    for position in range(len(route)):
        observation = Observation(
            frame=np.asarray(route.frames[position]),
            available_actions=legal_indices,
            task_id=game,
            stage=int(route.levels[position]) + 1,
            progress=float(route.levels[position]),
        )
        perception = PerceptionSystem(agent.config.perception)
        perceived = perception.observe(observation, step=0)
        objects = agent.affordances.enrich(perceived.objects, task_id=game)
        encode_started = time.monotonic()
        representation = agent.representation_backend.encode(observation, objects)
        encode_seconds.append(time.monotonic() - encode_started)
        snapshot = WorldSnapshot(
            observation=observation,
            objects=objects,
            events=perceived.events,
            topology=perceived.topology,
            representation=representation,
            step=0,
        )
        state_values.append(agent.student_policy._state_values(representation))
        scores = {
            action.index: agent.student_policy.score(snapshot, action).value
            for action in legal
        }
        best = max(scores, key=scores.get)
        recorded = int(route.actions[position].index)
        agree += int(best == recorded)
        ordered = sorted(scores.values(), reverse=True)
        margins.append(scores[recorded] - ordered[1])
    stacked = np.stack(state_values)
    distances = np.linalg.norm(stacked[:, None, :] - stacked[None, :, :], axis=2)
    upper = np.triu_indices(len(stacked), 1)
    adjacent = np.asarray(
        [distances[i, i + 1] for i in range(len(stacked) - 1)]
    )
    return {
        "backend": backend_name,
        "game": game,
        "distilled_examples": distilled,
        "distill_seconds": distill_elapsed,
        "fitted_task_scale": agent.student_policy.summary()["task_scales"].get(game),
        "c1_agree": agree,
        "c1_total": len(route),
        "c1_rate": agree / max(len(route), 1),
        "mean_margin_recorded_minus_runner_up": float(np.mean(margins)),
        "encode_seconds_mean": float(np.mean(encode_seconds)),
        "state_value_size": int(stacked.shape[1]),
        "latent_geometry": {
            "median_pairwise_distance": float(np.median(distances[upper])),
            "median_adjacent_distance": float(np.median(adjacent)),
            "min_pairwise_distance": float(np.min(distances[upper])),
        },
        "teacher_audit": retention._teacher_audit_payload(agent.teacher),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=["ls20", "tr87", "wa30"])
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=retention.DEFAULT_TRAJECTORY_ROOT,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)
    routes = {
        game: retention._load_trusted(args.trajectory_root, game, args.run_index)
        for game in args.games
    }
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "_"
        + uuid4().hex[:12]
    )
    output = (
        args.out_root
        if args.out_root is not None
        else retention.DEFAULT_ARTIFACT_PARENT / f"student_repr_ablation_{run_id}"
    )
    output.mkdir(parents=True, exist_ok=False)
    ouro_backend, ouro_manifest = _load_ouro_backend(args.device)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "started_at_utc": retention._utc_now(),
        "args": {
            "games": list(args.games),
            "run_index": int(args.run_index),
            "device": str(args.device),
        },
        "backends": list(BACKENDS),
        "ouro_manifest": ouro_manifest,
        "design": {
            "identical_head": (
                "same StudentPolicyConfig, same 96-value state input, same "
                "RFF map; only representation_backend differs"
            ),
            "compute_unmatched": (
                "frozen 2.6B forward vs hand-rolled features; per-encode cost "
                "reported so the tradeoff is explicit"
            ),
            "primary_metric": "C1 recorded-state argmax agreement per game",
            "no_action_selection": "offline distillation and scoring only",
            "backbone_frozen": True,
        },
        "attribution_script_sha256": retention._sha256_file(Path(__file__).resolve()),
        "source_manifest": retention._source_manifest(Path(__file__).resolve()),
        "trajectories": {
            game: retention._trajectory_manifest(route)
            for game, route in routes.items()
        },
        "package_versions": retention._package_versions(),
    }
    retention._write_json_exclusive(output / "run_started.json", provenance)
    rows: list[dict[str, Any]] = []
    for game in args.games:
        route = routes[game]
        for backend_name in BACKENDS:
            row = _evaluate_backend(
                backend_name,
                game=game,
                route=route,
                # None selects the agent's default GridFeatureBackend.
                backend=ouro_backend if backend_name == "ouro_loop" else None,
            )
            rows.append(row)
            print(retention._stable_json({k: row[k] for k in (
                "backend", "game", "c1_agree", "c1_total",
                "fitted_task_scale", "encode_seconds_mean",
            )}))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run_id": run_id,
        "completed_at_utc": retention._utc_now(),
        "provenance_file": "run_started.json",
        "rows": rows,
    }
    retention._write_json_exclusive(output / "summary.json", summary)
    print(f"artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
