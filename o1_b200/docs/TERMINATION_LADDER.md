# Termination ladder — who stops the bill, and what the log says

One page. If a failure class is not on this page, the system does not know
how it stops, and that is a defect. Every row is backed by code named in
the right-hand column; `test_combined_session_seam.py` executes the rows
marked ★ against the real driver and the real FL supervisor.

Budget facts: per-pod allowance = remaining allocation net of earlier pods
(`O1_SESSION_AUTHORIZED_SECONDS`); at most `MAX_POD_ACQUISITIONS = 4` pods
per session; the driver refuses to create a pod once the USD allocation is
exhausted, regardless of the slot count.

## Stops, strongest first

| # | Stop | Lives where | Fires when | Independent of |
|---|------|-------------|------------|----------------|
| 1 | **Driver terminate** (`terminate_and_confirm`) | operator machine, `zero_touch.py` | every session outcome — COMPLETE, ABORTED_*, eviction cleanup | the pod entirely |
| 2 | **Independent watchdog process** (`watchdog_terminate.py`) | operator machine, own session (`start_new_session`) | hard allowance reached, or the driver dies | the driver process; arming is confirmed per pod (`confirm_armed`, pod-scoped) or the pod is refused |
| 3 | **Provider `terminateAfter`** | RunPod | allowance + 30 min | everything on the operator machine; write-only at RunPod (cannot be read back), so never relied on alone |
| 4 | **Driver `monitor_timeout`** = allowance + 1800 s | `zero_touch.py` | a pod that neither completes nor exits | the pod's log protocol |
| 5 | Pod-side `BudgetWatchdog` (95 % soft stop / 100 % hard) | the pod, `production_entry` | O1 phase overruns | the driver (but not a SIGKILL) |
| 6 | FL `terminate_command` via `_emergency_close` / `--terminate-only` | the pod, FL supervisor / entry script | any FL refusal, including before the supervisor exists | the O1 driver's polling cadence |

A billing pod always has stops 1–4 armed. Stop 2 is the one the session
*relies* on (confirmed per pod); 3 and 4 are backstops.

## What the driver reads

The driver tails the container log (normalised to real lines) and takes
the LAST whole-line marker:

| Marker line | Driver verdict | Then |
|-------------|----------------|------|
| `ZERO_TOUCH_COMPLETE` | COMPLETE | download + digest-verify results, terminate |
| `ZERO_TOUCH_ABORTED_AT_<STATE>` | deterministic failure | collect logs, terminate, **no reacquisition** |
| none, pod RUNNING | keep polling | until marker, eviction, or stop 4 |
| none, pod EXITED | **eviction** | terminate remnant, reacquire (≤ 4 pods, ≤ 2 consecutive zero-progress) |
| `REFUSED (transient)` + exit | eviction (by design) | a fresh pod may not repeat it |

In a combined session the FL supervisor rewrites the O1 child's markers
to `O1_PHASE_*` and prints the session marker exactly once, at the end.
★ verified at three cut points.

## Failure classes → stop

| Failure | Marker | Stop | Reacquire? |
|---------|--------|------|------------|
| spot eviction, any phase | none (EXITED) | 1 on the remnant | yes ★ |
| pre-entry refusal: scope, fetch (401/403/404), artifact verify, env | `…_PRE_ENTRY_<STEP>` / `…_HF_SCOPE` / `…_ARTIFACT_FETCH` | 1 | no |
| pre-entry transient: hub 5xx/reset/timeout, mkdir on unattached volume | `REFUSED (transient)` | 1 | yes |
| O1 identity env missing / bad destination | `…_PRE_ENTRY_CONFIG` | 1 | no |
| O1 state abort (gate, affordability, digest) | `…_<STATE>` (O1-only) or `O1_PHASE_…` + `…_O1_HALT_OR_COMPLETE` (combined) | 6 then 1 | no |
| FL preflight refusal (config, scope, pregen) | `…_FL_PRE_ENTRY_<STEP>` | 6 (`--terminate-only`) then 1 | no |
| FL supervisor setup: FileNotFound/Permission/EROFS | `…_SUPERVISOR_SETUP` | 6 (raw-config terminate) then 1 | no |
| FL supervisor setup: ENOSPC/EIO/ENOMEM/connection | `REFUSED (transient)` | 6 then 1 | yes |
| FL resume refusal (e.g. allowance env missing) | `…_SUPERVISOR` | 6 (`_emergency_close`) then 1 ★ | no |
| FL state abort (ladder crash, transfer failure) | `…_<STATE>` | 6 then 1 ★ | no |
| FL ladder admitted nothing | `…_RUN_FL_LADDER` | 6 then 1 | no |
| driver process dies mid-session | — | 2, then 3 | n/a (operator restarts the driver; spend carryover is persisted) |
| operator machine dies | — | 3 | n/a |
| pod hangs silently | none | 4 then 1 | no (MONITOR_TIMEOUT is ABORTED_*) |

## The money bound

Worst realistic case: a deterministic pre-entry failure misclassified as
transient → up to 4 acquisitions × (pull + fetch ≈ 10 min) ≈ 40 min of
billing, then `ABORTED_ACQUISITION_LIMIT`. Every other path ends at the
first pod or at a genuine eviction. Nothing here can exceed the USD
allocation: the driver stops creating pods when it is spent.
