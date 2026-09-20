#!/usr/bin/env bash
# Thermal guard for the MMLU science v3 run (and successor GPU pipeline stages).
#
# Behaviour:
#   - Freezes the managed process with SIGSTOP when CPU OR GPU temperature
#     stays at/above its pause threshold for HOT_STREAK_NEED consecutive polls.
#   - Resumes it with SIGCONT once BOTH are at/below their resume thresholds
#     for COOL_STREAK_NEED consecutive polls.
#   - Hysteresis (separate pause/resume thresholds) + debounce avoid flapping.
#   - Reads the managed PID from PID_FILE every loop, so it automatically
#     follows successive pipeline stages: just write the new PID into PID_FILE.
#   - A trap guarantees a paused process is always resumed if the guard exits,
#     so the run is never left frozen.
#
# Stop cleanly with:  touch "$PID_FILE.stop"   (or kill -TERM the guard)
#
# Env overrides: GPU_PAUSE GPU_RESUME CPU_PAUSE CPU_RESUME INTERVAL
#                HOT_STREAK_NEED COOL_STREAK_NEED PID_FILE STATE_FILE THERMAL_LOG

set -u

PID_FILE="${PID_FILE:-/tmp/mmlu_v3_active_pid.txt}"
STATE_FILE="${STATE_FILE:-/tmp/mmlu_v3_thermal_state.txt}"
LOG="${THERMAL_LOG:-/home/moloch/ouro_project/artifacts/logs/thermal_guard.log}"

GPU_PAUSE="${GPU_PAUSE:-85}"
GPU_RESUME="${GPU_RESUME:-72}"
CPU_PAUSE="${CPU_PAUSE:-95}"
CPU_RESUME="${CPU_RESUME:-85}"
INTERVAL="${INTERVAL:-15}"
HOT_STREAK_NEED="${HOT_STREAK_NEED:-3}"
COOL_STREAK_NEED="${COOL_STREAK_NEED:-2}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
logline() { echo "[$(ts)] $*" >> "$LOG"; }

managed_pid() { cat "$PID_FILE" 2>/dev/null | tr -dc '0-9'; }

gpu_temp() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'
}

cpu_temp() {
  # hottest coretemp reading (package or any core), in whole degrees C
  local t
  t=$(sensors -u coretemp-isa-0000 2>/dev/null | awk '/_input:/{v=$2+0; if(v>m)m=v} END{printf "%d", m}')
  if [ -z "$t" ] || [ "$t" = "0" ]; then
    local z
    for z in /sys/class/thermal/thermal_zone*; do
      if [ "$(cat "$z/type" 2>/dev/null)" = "x86_pkg_temp" ]; then
        t=$(( $(cat "$z/temp" 2>/dev/null)/1000 )); break
      fi
    done
  fi
  echo "${t:-0}"
}

state="running"
hot=0
cool=0

cleanup() {
  local p; p=$(managed_pid)
  if [ "$state" = "paused" ] && [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
    kill -CONT "$p" 2>/dev/null && logline "cleanup: resumed pid=$p on guard exit (was paused)"
  fi
}
trap 'cleanup; logline "thermal_guard received TERM/INT, exiting"; exit 0' TERM INT
trap 'cleanup' EXIT

echo "$state" > "$STATE_FILE"
logline "thermal_guard start pid_file=$PID_FILE gpu_pause=${GPU_PAUSE} gpu_resume=${GPU_RESUME} cpu_pause=${CPU_PAUSE} cpu_resume=${CPU_RESUME} interval=${INTERVAL}s hot_need=${HOT_STREAK_NEED} cool_need=${COOL_STREAK_NEED}"

while true; do
  if [ -f "${PID_FILE}.stop" ]; then
    logline "stop flag present; exiting"
    break
  fi

  PID=$(managed_pid)
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    if [ "$state" = "paused" ]; then
      logline "managed pid ($PID) gone while paused; clearing state to running"
      state="running"; hot=0; cool=0; echo "$state" > "$STATE_FILE"
    fi
    sleep "$INTERVAL"
    continue
  fi

  g=$(gpu_temp); c=$(cpu_temp)
  g=${g:-0}; c=${c:-0}
  [ -z "$g" ] && g=0
  [ -z "$c" ] && c=0

  if [ "$state" = "running" ]; then
    if [ "$g" -ge "$GPU_PAUSE" ] || [ "$c" -ge "$CPU_PAUSE" ]; then
      hot=$((hot+1))
    else
      hot=0
    fi
    if [ "$hot" -ge "$HOT_STREAK_NEED" ]; then
      if kill -STOP "$PID" 2>/dev/null; then
        state="paused"; cool=0; echo "$state" > "$STATE_FILE"
        logline "PAUSE pid=$PID gpu=${g}C cpu=${c}C (thresholds gpu>=${GPU_PAUSE} or cpu>=${CPU_PAUSE}, sustained ${HOT_STREAK_NEED} polls)"
      fi
    fi
  else
    if [ "$g" -le "$GPU_RESUME" ] && [ "$c" -le "$CPU_RESUME" ]; then
      cool=$((cool+1))
    else
      cool=0
    fi
    if [ "$cool" -ge "$COOL_STREAK_NEED" ]; then
      if kill -CONT "$PID" 2>/dev/null; then
        state="running"; hot=0; echo "$state" > "$STATE_FILE"
        logline "RESUME pid=$PID gpu=${g}C cpu=${c}C (cooled to gpu<=${GPU_RESUME} and cpu<=${CPU_RESUME})"
      fi
    fi
  fi

  sleep "$INTERVAL"
done
