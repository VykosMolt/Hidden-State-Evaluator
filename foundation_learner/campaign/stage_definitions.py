"""The declarative campaign stage table (contract §7, §10, §11).

Every stage is a :class:`StageDefinition` record carrying

* its **priority** in the frozen §11 session order (BENCH, FL0, core matched
  comparison, FL4, FL5, FL6, FL7, FL8, additional predeclared seeds, and last
  the single sealed opening);
* an **entry condition** expressed as a dotted path to a promotion predicate
  (``campaign.promotion`` is the only source of promotion truth);
* a **work function** expressed as a DOTTED PATH resolved lazily at runtime, so
  a stage that depends on a module still under construction (``mechanisms/``)
  neither imports it at package-import time nor pins its API here;
* a **projection formula** consuming the BENCH measurements;
* the **outputs** it must produce, and
* its **predeclared fallback work** (contract §10) — no objective is ever
  invented live.

The work functions themselves live at the bottom of this module.  They are the
real integration points into W1 (data), W2 (training) and W4 (evaluation); the
mechanism rungs FL4–FL8 feature-detect ``foundation_learner.mechanisms.*`` and
record ``SKIPPED_MECHANISMS`` when that package is absent.  A mechanism module
that IS importable but lacks its declared entry point is an integration ERROR,
never a silent skip.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from . import o1_isolation, promotion, sealed_gate
from .affordability import (BenchMeasurement, PEFT_MODE, ThroughputTable,
                            grid_updates, projected_arm_seconds)

__all__ = [
    "StageError",
    "StageDefinition",
    "StageContext",
    "STAGE_TABLE",
    "STAGES_BY_ID",
    "CORE_ARM_STAGES",
    "EVAL_EPISODES_PER_STAGE",
    "BENCH_PROBE_UPDATES",
    "BENCH_PROBE_EPISODES",
    "resolve_dotted",
    "module_available",
    "resolve_optional",
    "project_stage_seconds",
    "stages_in_priority_order",
]


class StageError(RuntimeError):
    """A stage definition, entry condition, or work function refused."""


# --------------------------------------------------------------------------
# lazy dotted-path resolution
# --------------------------------------------------------------------------

def resolve_dotted(dotted: str) -> Any:
    """Resolve ``package.module:attribute`` at call time.

    Import happens HERE, not at module import, which is what lets the stage
    table name modules that are still being built by another worker.
    """
    if ":" not in dotted:
        raise StageError(
            f"dotted path {dotted!r} must be 'package.module:attribute'")
    module_name, attribute = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise StageError(
            f"module {module_name!r} has no attribute {attribute!r}; the stage "
            "table names an entry point that the module does not provide"
        ) from exc


def module_available(module_name: str) -> bool:
    """True when ``module_name`` can be imported (feature detection)."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def resolve_optional(dotted: str) -> Any | None:
    """Resolve a dotted path, or ``None`` when its MODULE is absent.

    A missing module is a legitimate "not built yet / not shipped" condition
    and yields ``None``.  A present module that lacks the named attribute is an
    integration error and raises: silently skipping that case would hide a real
    break.
    """
    module_name = dotted.split(":", 1)[0]
    if not module_available(module_name):
        return None
    return resolve_dotted(dotted)


# --------------------------------------------------------------------------
# frozen per-stage evaluation maxima (preregistration §12 item 5)
# --------------------------------------------------------------------------

#: Maximum DEVELOPMENT episodes evaluated per stage.  Frozen here so that the
#: only B200-derived quantity left open is the eval BATCH SIZE (post
#: equivalence gate), not the amount of evidence collected.
EVAL_EPISODES_PER_STAGE: dict[str, int] = {
    "FL0": 300,
    "DEV_GRID": 150,
    "FL1": 300,
    "FL2": 300,
    "FL3": 300,
    "FL4": 200,
    "FL5": 200,
    "FL6": 200,
    "FL7": 150,
    "FL8": 150,
    "SEALED_EVAL": 300,
}

#: BENCH probe size: small, measured, and never a scientific result.
BENCH_PROBE_UPDATES = 12
BENCH_PROBE_EPISODES = 2


# --------------------------------------------------------------------------
# stage definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    priority: int
    kind: str                      # BENCH | EVAL | TRAIN | GRID | MECHANISM | SEALED
    work: str                      # dotted path, resolved lazily
    projection: str                # projection formula key
    outputs: tuple[str, ...]
    entry: str | None = None       # dotted path to a promotion predicate
    fallback_work: tuple[str, ...] = ()
    requires_modules: tuple[str, ...] = ()
    arm_id: str | None = None
    eval_episodes: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "priority": self.priority,
            "kind": self.kind,
            "work": self.work,
            "projection": self.projection,
            "outputs": list(self.outputs),
            "entry": self.entry,
            "fallback_work": list(self.fallback_work),
            "requires_modules": list(self.requires_modules),
            "arm_id": self.arm_id,
            "eval_episodes": int(self.eval_episodes),
            "notes": self.notes,
        }


_W = "foundation_learner.campaign.stage_definitions"

STAGE_TABLE: tuple[StageDefinition, ...] = (
    StageDefinition(
        stage_id="BENCH", priority=1, kind="BENCH",
        work=f"{_W}:bench_work", projection="BENCH",
        outputs=("bench_measurement.json",),
        notes=("runtime/throughput validation; measured on a NON-EVALUATION "
               "TRAIN shard, never on DEV or SEALED data"),
        fallback_work=("refuse the whole FL ladder: without measured "
                       "throughput no projection may be made",),
    ),
    StageDefinition(
        stage_id="FL0", priority=2, kind="EVAL",
        work=f"{_W}:fl0_work", projection="EVAL",
        outputs=("fl0_base_report.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL0"],
        notes="base model, no training; not a treatment",
        fallback_work=("record the FL0 records that completed",),
    ),
    StageDefinition(
        stage_id="DEV_GRID", priority=3, kind="GRID",
        work=f"{_W}:dev_grid_work", projection="GRID",
        outputs=("DEV_DECISIONS_FROZEN.json", "dev_grid_report.json"),
        arm_id="FL3",
        eval_episodes=EVAL_EPISODES_PER_STAGE["DEV_GRID"],
        notes=("the frozen 2-LR development grid on FL3 at 25 % U; it is part "
               "of the core-comparison allocation (contract §8/§11) and must "
               "complete before any core arm starts"),
        fallback_work=("refuse the core arms: their learning rate is only "
                       "defined by this grid",),
    ),
    StageDefinition(
        stage_id="FL1", priority=4, kind="TRAIN",
        work=f"{_W}:core_arm_work", projection="TRAIN_ARM", arm_id="FL1",
        outputs=("arm_result_FL1.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL1"],
        notes="static baseline (isolated TASK -> ANSWER pairs)",
        fallback_work=("record the partial arm and its compute ledger",),
    ),
    StageDefinition(
        stage_id="FL2", priority=4, kind="TRAIN",
        work=f"{_W}:core_arm_work", projection="TRAIN_ARM", arm_id="FL2",
        outputs=("arm_result_FL2.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL2"],
        notes="successful-history imitation",
        fallback_work=("record the partial arm and its compute ledger",),
    ),
    StageDefinition(
        stage_id="FL3", priority=4, kind="TRAIN",
        work=f"{_W}:core_arm_work", projection="TRAIN_ARM", arm_id="FL3",
        outputs=("arm_result_FL3.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL3"],
        notes="ordered-feedback meta-training (CORE TREATMENT)",
        fallback_work=promotion.FALLBACKS["FL3_TO_EXTENSIONS"],
    ),
    StageDefinition(
        stage_id="FL4", priority=5, kind="MECHANISM",
        work=f"{_W}:fl4_work", projection="MECHANISM",
        entry=f"{_W}:fl4_entry",
        outputs=("fl4_value_head_report.json",),
        requires_modules=("foundation_learner.mechanisms.value_head",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL4"],
        fallback_work=promotion.FALLBACKS["FL4"],
    ),
    StageDefinition(
        stage_id="FL5", priority=6, kind="MECHANISM",
        work=f"{_W}:fl5_work", projection="MECHANISM",
        entry=f"{_W}:fl5_entry",
        outputs=("fl5_fast_state_report.json",),
        requires_modules=("foundation_learner.mechanisms.fast_state",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL5"],
        fallback_work=promotion.FALLBACKS["FL5"],
    ),
    StageDefinition(
        stage_id="FL6", priority=7, kind="MECHANISM",
        work=f"{_W}:fl6_work", projection="MECHANISM",
        entry=f"{_W}:fl6_entry",
        outputs=("fl6_value_gating_report.json",),
        requires_modules=("foundation_learner.mechanisms.value_gating",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL6"],
        fallback_work=promotion.FALLBACKS["FL6"],
    ),
    StageDefinition(
        stage_id="FL7", priority=8, kind="MECHANISM",
        work=f"{_W}:fl7_work", projection="MECHANISM",
        entry=f"{_W}:fl7_entry",
        outputs=("fl7_fast_adapter_report.json",),
        requires_modules=("foundation_learner.mechanisms.fast_adapter",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL7"],
        fallback_work=promotion.FALLBACKS["FL7"],
    ),
    StageDefinition(
        stage_id="FL8", priority=9, kind="MECHANISM",
        work=f"{_W}:fl8_work", projection="MECHANISM",
        entry=f"{_W}:fl8_entry",
        outputs=("fl8_consolidation_report.json",),
        requires_modules=("foundation_learner.mechanisms.consolidation",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL8"],
        fallback_work=promotion.FALLBACKS["FL8"],
    ),
    StageDefinition(
        stage_id="SECOND_SEED", priority=10, kind="TRAIN",
        work=f"{_W}:second_seed_work", projection="CORE_COMPARISON",
        outputs=("second_seed_report.json",),
        notes=("additional predeclared seed 20260810; predeclared work only, "
               "runs last among training work"),
        fallback_work=("skip; the single-seed result stands as reported",),
    ),
    StageDefinition(
        stage_id="SEALED_EVAL", priority=11, kind="SEALED",
        work=f"{_W}:sealed_eval_work", projection="EVAL",
        outputs=("SEALED_OPENING_LEDGER.jsonl", "sealed_eval_report.json"),
        eval_episodes=EVAL_EPISODES_PER_STAGE["SEALED_EVAL"],
        notes=("the single sealed opening, LAST: it can never inform a "
               "development decision because every such decision is already "
               "frozen in DEV_DECISIONS_FROZEN.json"),
        fallback_work=("leave the sealed set unopened",),
    ),
)

STAGES_BY_ID: dict[str, StageDefinition] = {s.stage_id: s for s in STAGE_TABLE}
CORE_ARM_STAGES: tuple[str, ...] = ("FL1", "FL2", "FL3")


def stages_in_priority_order() -> list[StageDefinition]:
    """Frozen §11 session priority (stable within a priority band)."""
    return sorted(STAGE_TABLE, key=lambda s: (s.priority, s.stage_id))


# --------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------

def project_stage_seconds(stage: StageDefinition, *, bench: BenchMeasurement,
                          updates: int | None,
                          eval_episodes: int | None = None) -> dict:
    """Projected seconds for one stage from the measured BENCH numbers."""
    episodes = int(stage.eval_episodes if eval_episodes is None else eval_episodes)
    detail: dict[str, Any] = {"stage": stage.stage_id,
                              "projection": stage.projection,
                              "scope": bench.scope}
    if stage.projection == "BENCH":
        seconds = projected_arm_seconds(bench, BENCH_PROBE_UPDATES)
        detail["probe_updates"] = BENCH_PROBE_UPDATES
    elif stage.projection == "EVAL":
        if bench.eval_seconds_per_episode is None:
            raise StageError(
                f"{stage.stage_id}: BENCH measured no evaluation cost; "
                "evaluation seconds may not be guessed")
        seconds = float(bench.eval_seconds_per_episode) * episodes
        detail["eval_episodes"] = episodes
    elif stage.projection in ("TRAIN_ARM", "MECHANISM"):
        if updates is None:
            raise StageError(
                f"{stage.stage_id}: no U selected; a training stage cannot be "
                "projected before the affordability rule chooses U")
        train = projected_arm_seconds(bench, int(updates))
        eval_seconds = (0.0 if bench.eval_seconds_per_episode is None
                        else float(bench.eval_seconds_per_episode) * episodes)
        seconds = train + eval_seconds
        detail.update({"updates": int(updates), "train_seconds": train,
                       "eval_seconds": eval_seconds, "eval_episodes": episodes})
    elif stage.projection == "GRID":
        if updates is None:
            raise StageError(f"{stage.stage_id}: no U selected")
        per_config = projected_arm_seconds(bench, grid_updates(int(updates)))
        eval_seconds = (0.0 if bench.eval_seconds_per_episode is None
                        else float(bench.eval_seconds_per_episode) * episodes * 2)
        seconds = 2.0 * per_config + eval_seconds
        detail.update({"configs": 2, "updates_per_config": grid_updates(int(updates)),
                       "eval_seconds": eval_seconds})
    elif stage.projection == "CORE_COMPARISON":
        if updates is None:
            raise StageError(f"{stage.stage_id}: no U selected")
        train = 3.0 * projected_arm_seconds(bench, int(updates))
        eval_seconds = (0.0 if bench.eval_seconds_per_episode is None
                        else float(bench.eval_seconds_per_episode) * episodes * 3)
        seconds = train + eval_seconds
        detail.update({"arms": 3, "updates": int(updates)})
    else:  # pragma: no cover - the table is closed
        raise StageError(f"unknown projection formula {stage.projection!r}")
    detail["projected_seconds"] = float(seconds)
    return detail


# --------------------------------------------------------------------------
# the stage context
# --------------------------------------------------------------------------

@dataclass
class StageContext:
    """Everything a work function may touch.

    ``bundle_factory`` returns a FRESH model bundle: contract §7 requires every
    independent arm to start from a fresh load of the identical frozen
    checkpoint, so no work function is given a shared, already-trained model.
    """

    out_dir: str
    pregen_root: str
    bundle_factory: Callable[[], Any]
    guard: o1_isolation.IsolationGuard
    throughput: ThroughputTable = field(default_factory=ThroughputTable)
    scope: str = PEFT_MODE
    updates: int | None = None
    learning_rate: float | None = None
    root_seed: int = 20260809
    max_tokens_per_batch: int = 2048
    eval_episode_cap: int | None = None
    train_example_cap: int | None = None
    rehearsal: bool = False
    label: str = "FL_CAMPAIGN"
    results: dict[str, Any] = field(default_factory=dict)
    scheduler: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------

    def stage_dir(self, stage_id: str) -> str:
        return self.guard.makedirs(os.path.join(self.out_dir, stage_id.lower()))

    def eval_cap(self, stage: StageDefinition) -> int:
        cap = stage.eval_episodes or 1
        if self.eval_episode_cap is not None:
            cap = min(cap, int(self.eval_episode_cap))
        return max(1, cap)

    def write(self, stage_id: str, name: str, payload: Any) -> str:
        return self.guard.write_json(
            os.path.join(self.stage_dir(stage_id), name), payload)


# --------------------------------------------------------------------------
# data access (always through the loader guard)
# --------------------------------------------------------------------------

def _episode_shards(pregen_root: str, split: str, mode: str,
                    guard: o1_isolation.IsolationGuard) -> list[str]:
    root = guard.guard(os.path.join(pregen_root, "episodes", split),
                       o1_isolation.MODE_READ)
    if not os.path.isdir(root):
        raise StageError(f"missing pre-generated episodes under {root!r}")
    return [os.path.join(root, name) for name in sorted(os.listdir(root))
            if name.endswith(f".{mode}.jsonl")]


def load_episodes(pregen_root: str, split: str, mode: str, *,
                  guard: o1_isolation.IsolationGuard,
                  limit_per_family: int | None = None) -> list[Any]:
    """Load plain (non-sealed) episodes through the loader guard."""
    from ..data.shards import read_shard
    from ..episodes.schema import Episode

    episodes: list[Any] = []
    for path in _episode_shards(pregen_root, split, mode, guard):
        resolved = sealed_gate.loader_guard(path, unlock=None, guard=guard,
                                            context="stage_definitions")
        rows = read_shard(resolved)
        if limit_per_family is not None:
            rows = rows[:int(limit_per_family)]
        episodes.extend(Episode.from_dict(row) for row in rows)
    return episodes


def load_fl1_items(pregen_root: str, *, guard: o1_isolation.IsolationGuard,
                   limit_per_family: int | None = None) -> list[dict]:
    from ..data.shards import read_shard

    root = guard.guard(os.path.join(pregen_root, "fl1_static", "TRAIN"),
                       o1_isolation.MODE_READ)
    if not os.path.isdir(root):
        raise StageError(f"missing FL1 static pool under {root!r}")
    items: list[dict] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl"):
            continue
        rows = read_shard(os.path.join(root, name))
        if limit_per_family is not None:
            rows = rows[:int(limit_per_family)]
        items.extend(rows)
    return items


def env_factory(episode: Any):
    """Exact-verifier environment for one episode (W4 API)."""
    from ..ecology.families import get_family
    from ..evaluation.learning_curve import environment_from_episode

    return environment_from_episode(episode, get_family(episode.family_id))


def _render_fl1_pair(prompt_text: str, answer_payload: str):
    """FL1 pool rendering with the SAME surface as the episode renderer.

    Contract §8/§20 forbid hidden asymmetries between the core arms; a
    different textual surface for FL1 would be exactly such an asymmetry, so
    the instruction header and the ``TASK:``/``ATTEMPT:`` markers are
    reproduced from ``episodes/render.py`` rather than reinvented.
    """
    from ..episodes.render import INSTRUCTION_HEADER_V0, MARKERS
    from ..episodes.schema import Role

    prefix = (f"{INSTRUCTION_HEADER_V0}{MARKERS[Role.TASK]} "
              f"{prompt_text.rstrip()}\n{MARKERS[Role.MODEL_ATTEMPT]} ")
    answer_line = f"ANSWER: {answer_payload.strip()}"
    return prefix + answer_line + "\n", (len(prefix), len(prefix) + len(answer_line))


def build_arm_examples(ctx: StageContext, arm_id: str, tokenizer) -> list[dict]:
    """Tokenised training examples for one arm from the pre-generated data."""
    from ..training.tokenization import (isolated_pair_to_training_example,
                                         iter_examples)

    cap = ctx.train_example_cap
    if arm_id == "FL1":
        items = load_fl1_items(ctx.pregen_root, guard=ctx.guard,
                               limit_per_family=cap)
        examples = [
            isolated_pair_to_training_example(
                item["prompt_text"], item["answer_canonical"], tokenizer,
                render_pair_fn=_render_fl1_pair,
                meta={"item_id": item.get("item_id"),
                      "episode_id": item.get("source_episode_id"),
                      "family_id": item.get("family_id")})
            for item in items]
    else:
        mode = "successful" if arm_id == "FL2" else "scripted"
        episodes = load_episodes(ctx.pregen_root, "TRAIN", mode,
                                 guard=ctx.guard, limit_per_family=cap)
        examples = list(iter_examples(episodes, tokenizer, arm_id))
    if not examples:
        raise StageError(f"{arm_id}: no training examples were built")
    return examples


# --------------------------------------------------------------------------
# work functions
# --------------------------------------------------------------------------

def _arm_config(ctx: StageContext, arm_id: str, *, updates: int, stage: str,
                learning_rate: float):
    """Build the arm configuration, with the frozen checkpoint cadence.

    A DRESS REHEARSAL runs the miniature ladder, whose step counts are (by
    design) not on the frozen ladder; such a run is tagged ``STAGE_SMOKE``,
    which ``training/arms.py`` defines as "local mechanics only; never a
    scientific result".  A real campaign keeps the CORE/GRID stage tags and is
    therefore fully bound by the frozen ladder and learning-rate grid.
    """
    from ..training.arms import STAGE_SMOKE, make_arm_config

    stage_tag = STAGE_SMOKE if ctx.rehearsal else stage
    return make_arm_config(
        arm_id, learning_rate=float(learning_rate), updates=int(updates),
        max_tokens_per_batch=int(ctx.max_tokens_per_batch), peft_mode=ctx.scope,
        seed=int(ctx.root_seed), stage=stage_tag,
        checkpoint_every_steps=200, checkpoint_every_seconds=600.0,
        max_seq_len=min(2048, int(ctx.max_tokens_per_batch)))


def bench_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """Measure real throughput on a NON-EVALUATION TRAIN shard (contract §11).

    Two kinds of number are measured, both by really running the code the
    campaign will run:

    * seconds per optimizer update, per trainable-parameter scope (W2 trainer
      on TRAIN episodes).  Both scopes are measured when both are requested,
      because the §11 FULL-vs-PEFT rule needs a FULL projection to decide, and
      an unmeasured scope is treated as INELIGIBLE rather than estimated;
    * seconds per evaluation episode (W4 online walker on TRAIN episodes).  The
      evaluation cost does not depend on the trainable-parameter scope — the
      walker only runs greedy inference — so it is measured once and attached
      to every measurement.

    The benchmark never touches DEVELOPMENT or SEALED_TEST data.
    """
    from ..training.arms import STAGE_SMOKE
    from ..training.trainer import run_training_arm

    out_dir = ctx.stage_dir(stage.stage_id)
    scopes = list(ctx.extra.get("bench_scopes") or [ctx.scope])
    updates = int(ctx.extra.get("bench_probe_updates", BENCH_PROBE_UPDATES))
    original_scope = ctx.scope

    # evaluation cost, measured on TRAIN episodes (never on evaluation data)
    n_probe = int(ctx.extra.get("bench_probe_episodes", BENCH_PROBE_EPISODES))
    probe_episodes = load_episodes(ctx.pregen_root, "TRAIN", "scripted",
                                   guard=ctx.guard, limit_per_family=1)[:n_probe]
    eval_seconds_per_episode = None
    eval_bundle = ctx.bundle_factory()
    if probe_episodes:
        from ..evaluation.learning_curve import LearningCurveConfig, run_episodes

        t0 = time.monotonic()
        records = run_episodes(eval_bundle, probe_episodes, env_factory,
                               cfg=LearningCurveConfig(
                                   arm_tag="BENCH_PROBE",
                                   generation=_generation_config(ctx)))
        elapsed = time.monotonic() - t0
        eval_seconds_per_episode = elapsed / max(1, len(records))

    measurements: dict[str, dict] = {}
    ledgers: dict[str, dict] = {}
    try:
        for scope in scopes:
            ctx.scope = scope
            bundle = ctx.bundle_factory()
            examples = build_arm_examples(ctx, "FL3", bundle.tokenizer)
            cfg = _arm_config(ctx, "FL3", updates=updates, stage=STAGE_SMOKE,
                              learning_rate=float(ctx.extra.get("bench_lr", 1e-4)))
            if ctx.scheduler is not None:
                ctx.scheduler.assert_checkpoint_cadence(cfg)
            result = run_training_arm(cfg, bundle, examples,
                                      os.path.join(out_dir, f"bench_arm_{scope}"),
                                      _trainer_hooks(ctx, stage.stage_id),
                                      allow_retry=False)
            measurement = BenchMeasurement.from_ledger(
                result.ledger, scope,
                eval_seconds_per_episode=eval_seconds_per_episode,
                notes=("measured on a non-evaluation TRAIN shard",
                       "NOT a scientific result: throughput only"))
            ctx.throughput.add(measurement)
            measurements[scope] = measurement.to_dict()
            ledgers[scope] = result.ledger
    finally:
        ctx.scope = original_scope

    payload = {
        "schema": "flb200.bench_report.v1",
        "stage": stage.stage_id,
        "scopes_measured": scopes,
        "measurements": measurements,
        "compute_ledgers": ledgers,
        "eval_seconds_per_episode": eval_seconds_per_episode,
        "probe_episodes": len(probe_episodes),
        "note": ("BENCH measures runtime only; it never produces a scientific "
                 "result and never reads DEVELOPMENT or SEALED_TEST data"),
    }
    ctx.write(stage.stage_id, "bench_measurement.json", payload)
    ctx.results["BENCH"] = payload
    return payload


#: Contract §6/§7 frozen evaluation decode budget.
FROZEN_MAX_NEW_TOKENS = 64


def _generation_config(ctx: StageContext):
    """Greedy decode configuration.

    ``max_new_tokens`` is FROZEN at 64 for the campaign.  A dress rehearsal may
    shorten it (``ctx.extra["max_new_tokens"]``) purely to keep the offline
    mechanics walk under its wall-clock limit; the override is only honoured
    for a run explicitly marked ``rehearsal`` and is recorded in the report.
    """
    from ..evaluation.generation import GenerationConfig

    tokens = FROZEN_MAX_NEW_TOKENS
    override = ctx.extra.get("max_new_tokens")
    if override is not None:
        if not ctx.rehearsal:
            raise StageError(
                "max_new_tokens is frozen at 64 for the campaign; only a "
                "labelled dress rehearsal may shorten it (contract §6)")
        tokens = int(override)
    return GenerationConfig(max_new_tokens=tokens)


def _dev_episodes(ctx: StageContext, stage: StageDefinition) -> list[Any]:
    cap = ctx.eval_cap(stage)
    episodes = load_episodes(ctx.pregen_root, "DEVELOPMENT", "scripted",
                             guard=ctx.guard)
    return episodes[:cap]


def _dev_metrics(records: Sequence[Mapping[str, Any]], stage: str,
                 stability_events: Sequence[str] = ()) -> promotion.DevMetrics:
    return promotion.DevMetrics.from_records(
        records, stage=stage, stability_events=stability_events)


def _run_dev_eval(ctx: StageContext, bundle: Any, episodes: Sequence[Any],
                  arm_tag: str) -> list[dict]:
    from ..evaluation.learning_curve import (LearningCurveConfig,
                                             annotate_records, run_episodes)

    records = run_episodes(bundle, episodes, env_factory,
                           cfg=LearningCurveConfig(
                               arm_tag=arm_tag,
                               generation=_generation_config(ctx)))
    return annotate_records(records, split="DEVELOPMENT")


def fl0_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """FL0: base model, no training; the four frozen cells on DEVELOPMENT."""
    from ..evaluation.fl0_base import run_fl0_base

    episodes = _dev_episodes(ctx, stage)
    bundle = ctx.bundle_factory()
    outcome = run_fl0_base(bundle, episodes, env_factory,
                           out_dir=ctx.stage_dir(stage.stage_id),
                           arm_tag="FL0_BASE",
                           generation=_generation_config(ctx),
                           run_gate=bool(ctx.extra.get("run_equivalence_gate", True)),
                           trained_family_ids=[])
    records = outcome["records_by_cell"].get("structured__history", [])
    records = [dict(r, split="DEVELOPMENT") for r in records]
    ctx.results["FL0"] = {"report": outcome["report"],
                          "dev": _dev_metrics(records, "FL0").to_dict()}
    ctx.results.setdefault("_records", {})["FL0"] = records
    return ctx.results["FL0"]


def _trainer_hooks(ctx: StageContext, stage_id: str):
    """Scheduler heartbeat + checkpoint notification, when a scheduler exists."""
    from ..training.trainer import TrainerHooks

    if ctx.scheduler is None:
        return None
    return TrainerHooks(
        on_step=ctx.scheduler.heartbeat_hook(stage_id),
        on_checkpoint=lambda tag, manifest: ctx.scheduler.journal(
            "CHECKPOINT_WRITTEN",
            {"stage_id": stage_id, "tag": tag,
             "manifest_hash": manifest.get("manifest_hash")}))


def _train_arm(ctx: StageContext, arm_id: str, *, updates: int, stage_name: str,
               learning_rate: float, out_dir: str) -> dict:
    from ..training.trainer import run_training_arm

    bundle = ctx.bundle_factory()          # FRESH load per arm (contract §7)
    examples = build_arm_examples(ctx, arm_id, bundle.tokenizer)
    cfg = _arm_config(ctx, arm_id, updates=updates, stage=stage_name,
                      learning_rate=learning_rate)
    if ctx.scheduler is not None:
        # the frozen §11 cadence (600 s / 200 steps) must reach the trainer
        ctx.scheduler.assert_checkpoint_cadence(cfg)
    result = run_training_arm(cfg, bundle, examples, out_dir,
                              _trainer_hooks(ctx, arm_id))
    return {"result": result, "bundle": bundle}


def dev_grid_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """The frozen 2-LR FL3 grid at 25 % U, its selection, and the freeze."""
    from ..training.arms import STAGE_GRID
    from . import dev_selector

    if ctx.updates is None:
        raise StageError("DEV_GRID needs the affordability-selected U")
    out_root = ctx.stage_dir(stage.stage_id)
    grid_lrs = dev_selector.frozen_grid(ctx.scope)
    n_updates = dev_selector.expected_grid_updates(ctx.updates)
    episodes = _dev_episodes(ctx, stage)
    runs = []
    reports = []
    for lr in grid_lrs:
        out_dir = os.path.join(out_root, f"lr_{lr:g}")
        trained = _train_arm(ctx, "FL3", updates=n_updates, stage_name=STAGE_GRID,
                             learning_rate=lr, out_dir=out_dir)
        records = _run_dev_eval(ctx, trained["bundle"], episodes,
                                arm_tag=f"FL3_GRID_lr{lr:g}")
        dev = _dev_metrics(records, "FL3_GRID")
        runs.append(dev_selector.GridRun(
            learning_rate=lr, scope=ctx.scope, updates=n_updates, dev=dev,
            arm_config_hash=trained["result"].arm_config_hash,
            checkpoint_tags=tuple(c["tag"] for c in trained["result"].checkpoints)))
        reports.append({"learning_rate": lr,
                        "arm_result": trained["result"].to_dict(),
                        "dev": dev.to_dict()})
    selection = dev_selector.select_learning_rate(
        runs, scope=ctx.scope, updates_U=ctx.updates)
    ctx.learning_rate = selection.chosen_learning_rate

    hashes = dict(ctx.extra.get("campaign_hashes", {}))
    hashes.setdefault("grid_arm_config_hashes",
                      [r.arm_config_hash for r in runs])
    stage_states = dict(ctx.extra.get("stage_states", {}))
    stage_states.setdefault("BENCH", "COMPLETE" if "BENCH" in ctx.results else "SKIPPED")
    stage_states.setdefault("FL0", "COMPLETE" if "FL0" in ctx.results else "SKIPPED")
    stage_states["DEV_GRID"] = "COMPLETE"
    checkpoint_tags = {f"FL3_GRID_lr{r.learning_rate:g}": list(r.checkpoint_tags)
                       for r in runs}
    decisions_path = os.path.join(ctx.out_dir, sealed_gate.DEV_DECISIONS_NAME)
    decisions = dev_selector.freeze_dev_decisions(
        decisions_path, selection=selection, updates_U=ctx.updates,
        stage_states=stage_states, checkpoint_tags=checkpoint_tags,
        hashes=hashes, runs=runs, guard=ctx.guard)
    payload = {
        "schema": "flb200.dev_grid_report.v1",
        "selection": selection.to_dict(),
        "runs": reports,
        "dev_decisions_path": decisions_path,
        "dev_decisions_sha256": decisions["dev_decisions_sha256"],
    }
    ctx.write(stage.stage_id, "dev_grid_report.json", payload)
    ctx.results["DEV_GRID"] = payload
    return payload


def core_arm_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """One matched core arm (FL1 / FL2 / FL3) plus its DEVELOPMENT evaluation."""
    from ..training.arms import STAGE_CORE

    arm_id = stage.arm_id or stage.stage_id
    if ctx.updates is None:
        raise StageError(f"{arm_id}: no U selected")
    if ctx.learning_rate is None:
        raise StageError(
            f"{arm_id}: no learning rate; the DEV_GRID stage must select it "
            "before any core arm runs (contract §8)")
    out_dir = os.path.join(ctx.stage_dir(stage.stage_id), "arm")
    trained = _train_arm(ctx, arm_id, updates=int(ctx.updates),
                         stage_name=STAGE_CORE,
                         learning_rate=float(ctx.learning_rate), out_dir=out_dir)
    episodes = _dev_episodes(ctx, stage)
    records = _run_dev_eval(ctx, trained["bundle"], episodes, arm_tag=arm_id)
    dev = _dev_metrics(records, arm_id,
                       stability_events=[e.get("kind", "EVENT")
                                         for e in trained["result"].stability_events])
    payload = {
        "schema": "flb200.core_arm_report.v1",
        "arm_id": arm_id,
        "arm_result": trained["result"].to_dict(),
        "dev": dev.to_dict(),
    }
    ctx.write(stage.stage_id, f"arm_result_{arm_id}.json", payload)
    ctx.results[arm_id] = payload
    ctx.results.setdefault("_records", {})[arm_id] = records
    ctx.results.setdefault("_dev_metrics", {})[arm_id] = dev
    return payload


def second_seed_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """The second predeclared seed: the same core comparison, seed 20260810."""
    from ..training.arms import ROOT_SEED_SECONDARY

    if int(ctx.root_seed) == int(ROOT_SEED_SECONDARY):
        raise StageError("the campaign is already running the second seed")
    return {"status": "PREDECLARED_NOT_RUN",
            "note": ("the second predeclared seed repeats the frozen core "
                     "comparison with root seed 20260810; it is launched as a "
                     "fresh campaign run with --seed 20260810 rather than "
                     "mutating this run's frozen seed"),
            "seed": int(ROOT_SEED_SECONDARY)}


# -- mechanism rungs (W3); feature-detected -------------------------------

def _mechanism_stage(ctx: StageContext, stage: StageDefinition,
                     entry_point: str) -> dict:
    """Run a mechanism rung, or record SKIPPED_MECHANISMS when absent."""
    missing = [m for m in stage.requires_modules if not module_available(m)]
    if missing:
        payload = {
            "schema": "flb200.mechanism_stage.v1",
            "stage": stage.stage_id,
            "status": "SKIPPED_MECHANISMS",
            "missing_modules": missing,
            "note": ("the mechanisms package is not importable in this "
                     "runtime; the rung is recorded as skipped, never as a "
                     "null result"),
        }
        ctx.write(stage.stage_id, stage.outputs[0], payload)
        ctx.results[stage.stage_id] = payload
        return payload
    run = resolve_dotted(entry_point)     # present module without the entry
    payload = run(ctx, stage)             # point is an error, not a skip
    ctx.results[stage.stage_id] = payload
    return payload


def fl4_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.value_head:run_fl4_stage")


def fl5_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.fast_state:run_fl5_stage")


def fl6_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.value_gating:run_fl6_stage")


def fl7_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.fast_adapter:run_fl7_stage")


def fl8_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.consolidation:run_fl8_stage")


# -- sealed opening --------------------------------------------------------

def sealed_eval_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """The single sealed opening and the sealed evaluation (contract §12)."""
    from ..episodes.schema import Episode
    from ..evaluation.learning_curve import LearningCurveConfig, run_episodes
    from ..evaluation import metrics as _metrics

    out_dir = ctx.stage_dir(stage.stage_id)
    unlock = sealed_gate.open_sealed(
        ledger_path=os.path.join(ctx.out_dir, sealed_gate.LEDGER_NAME),
        dev_decisions_path=os.path.join(ctx.out_dir,
                                        sealed_gate.DEV_DECISIONS_NAME),
        split_manifest_path=os.path.join(ctx.pregen_root,
                                         "family_split_manifest.json"),
        stage_states=ctx.extra.get("stage_states"),
        guard=ctx.guard,
        opened_by=f"{ctx.label}:SEALED_EVAL")
    cap = ctx.eval_cap(stage)
    rows: list[dict] = []
    for path in _episode_shards(ctx.pregen_root, "SEALED_TEST", "scripted",
                                ctx.guard):
        rows.extend(unlock.read_shard(path))
    episodes = [Episode.from_dict(row) for row in rows][:cap]
    bundle = ctx.bundle_factory()
    records = run_episodes(bundle, episodes, env_factory,
                           cfg=LearningCurveConfig(
                               arm_tag="SEALED_EVAL",
                               generation=_generation_config(ctx)))
    records = [dict(r, split="SEALED_TEST") for r in records]
    summary = _metrics.summarize(records)
    unlock.write_result_records(os.path.join(out_dir, "sealed_records.jsonl"),
                                records)
    report = {
        "schema": "flb200.sealed_eval_report.v1",
        "n_episodes": len(episodes),
        "summary": summary,
        "unlock": unlock.to_dict(),
        "policy": ("no model modification may be justified from these "
                   "outcomes; a further cycle requires a NEW sealed set"),
    }
    unlock.write_result(os.path.join(out_dir, "sealed_eval_report.json"), report)
    unlock.revoke()
    ctx.results["SEALED_EVAL"] = report
    return report


# --------------------------------------------------------------------------
# entry conditions (all promotion truth comes from campaign.promotion)
# --------------------------------------------------------------------------

def _core_dev(ctx: StageContext, arm_id: str) -> promotion.DevMetrics | None:
    return ctx.results.get("_dev_metrics", {}).get(arm_id)


def fl3_extension_gate(ctx: StageContext) -> promotion.PromotionDecision:
    fl3 = _core_dev(ctx, "FL3")
    fl1 = _core_dev(ctx, "FL1")
    if fl3 is None or fl1 is None:
        return promotion.PromotionDecision(
            stage="FL3_TO_EXTENSIONS", admitted=False,
            reasons=("FAIL: the core comparison has not produced DEV metrics "
                     "for both FL3 and FL1",),
            fallback=tuple(promotion.fallback_for("FL3_TO_EXTENSIONS")))
    return promotion.promote_fl3_to_extensions(fl3, fl1)


def fl4_entry(ctx: StageContext) -> promotion.PromotionDecision:
    gate = fl3_extension_gate(ctx)
    if not gate.admitted:
        return gate
    counts = ctx.extra.get("fl4_scoreable_items_per_episode", [])
    return promotion.promote_fl4(fl3_finished="FL3" in ctx.results,
                                 scoreable_items_per_episode=counts)


def fl5_entry(ctx: StageContext) -> promotion.PromotionDecision:
    gate = fl3_extension_gate(ctx)
    if not gate.admitted:
        return gate
    complete = all(arm in ctx.results for arm in CORE_ARM_STAGES)
    return promotion.promote_fl5(core_comparison_complete=complete,
                                 scheduler_admits=True)


def fl6_entry(ctx: StageContext) -> promotion.PromotionDecision:
    fl4 = ctx.results.get("FL4") or {}
    accuracy = fl4.get("pairwise_ranking_accuracy")
    if accuracy is None:
        return promotion.PromotionDecision(
            stage="FL6", admitted=False,
            reasons=("FAIL: no FL4 DEV pairwise ranking accuracy is available",),
            fallback=tuple(promotion.fallback_for("FL6")))
    return promotion.promote_fl6(pairwise_ranking_accuracy=float(accuracy))


def fl7_entry(ctx: StageContext) -> promotion.PromotionDecision:
    fl4 = ctx.results.get("FL4") or {}
    return promotion.promote_fl7(
        scheduler_admits=True,
        fast_update_stable=bool(ctx.extra.get("fast_update_stable", False)),
        fl4_succeeded=bool(fl4.get("status") == "COMPLETE"),
        only_remaining_variant_is_fl6_gated=bool(
            ctx.extra.get("only_fl6_gated_variant_remains", False)))


def fl8_entry(ctx: StageContext) -> promotion.PromotionDecision:
    evidence = ctx.extra.get("persistence_evidence")
    if not evidence:
        return promotion.PromotionDecision(
            stage="FL8", admitted=False,
            reasons=("FAIL: no DEV context-reset persistence evidence",),
            fallback=tuple(promotion.fallback_for("FL8")))
    return promotion.promote_fl8(
        persistence_point=float(evidence["point"]),
        persistence_ci=evidence["ci"])
