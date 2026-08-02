from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from utilities.tests.manual.run_hs_v2_ls20_count_graph_goal_v1 import (
    _acquisition_evidence,
)


def _transition(before: np.ndarray, after: np.ndarray, *, before_stage: int = 1, after_stage: int = 1, completed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        before=SimpleNamespace(
            observation=SimpleNamespace(frame=before, stage=before_stage),
        ),
        after_observation=SimpleNamespace(frame=after, stage=after_stage),
        action=SimpleNamespace(key=(1, -1, -1)),
        outcome=SimpleNamespace(
            completed=completed,
            boundary=SimpleNamespace(value="level_completed" if completed else "none"),
        ),
    )


def test_immediate_swap_shadow_count_diagnostic_is_policy_inert() -> None:
    initial = np.asarray([[11, 11, 11, 11], [0, 0, 0, 0]], dtype=np.int64)
    middle = np.asarray([[11, 11, 0, 0], [0, 0, 0, 0]], dtype=np.int64)
    successor = np.full_like(initial, 7)
    trace = SimpleNamespace(
        transitions=(
            _transition(initial, middle),
            _transition(middle, successor, before_stage=1, after_stage=2, completed=True),
        )
    )

    evidence = _acquisition_evidence(
        trace,
        boundary_bridge_change_fraction=0.05,
    )
    shadow = evidence["shadow_count_diagnostics"]
    value_11 = shadow["values"]["11"]

    assert shadow["target_inference"] == "abstain"
    assert shadow["policy_influence"] is False
    assert value_11["target"] is None
    assert value_11["diagnostic_only"] is True
    assert value_11["policy_influence"] is False
    assert value_11["pre_completion_count"] == 2
    assert value_11["step_index_aligned"] is True
    assert evidence["semantic_transitions"][-1]["protocol"] == "immediate_stage_swap"
