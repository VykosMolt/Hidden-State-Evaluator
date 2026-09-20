#!/usr/bin/env bash
# Downstream pipeline for mmlu_science_branch_parser_repair_v3, run after calibration.
# Best-effort: runs every stage in order, logs each stage's verdict + exit code,
# continues even if a stage fails (each script saves its own partial artifacts/verdict).
# Hands each stage's python PID to the thermal guard via the managed-PID file.

set -u
cd /home/moloch/ouro_project

PIDFILE=/tmp/mmlu_v3_active_pid.txt

STAGES=(
  evaluate_bg_mmlu_science_recipe_v3_heldout.py
  analyze_bg_mmlu_science_v3_source_failures.py
  run_bg_mmlu_science_l47_v3_ablation.py
  analyze_bg_mmlu_science_budget_breadth_v3.py
  analyze_bg_mmlu_science_soft_hair_no_good_v3.py
  check_bg_reasoning_guardrail_v3.py
  analyze_bg_mmlu_science_v3_readiness.py
  analyze_bg_pre_steering_domain_decision_v3.py
  analyze_bg_mmlu_science_branch_parser_repair_v3.py
)

fail=0
for stage in "${STAGES[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] START $stage ==="
  venv/bin/python -u "utilities/tests/manual/$stage" &
  p=$!
  echo "$p" > "$PIDFILE"          # let the thermal guard track this stage
  wait "$p"; rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] STAGE_OK $stage rc=0 ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] STAGE_FAIL $stage rc=$rc ==="
    fail=$((fail+1))
  fi
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] DOWNSTREAM_COMPLETE stages_failed=$fail ==="
