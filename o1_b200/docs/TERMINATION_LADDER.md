# Termination ladder — who stops the bill, and what the log says

One page. If a failure class is not on this page, the system does not know
how it stops, and that is a defect. Every row is backed by code named in
the right-hand column; `test_combined_session_seam.py` executes the rows
marked ★ against the real driver and the real FL supervisor.

Budget facts: per-pod allowance = remaining allocation net of earlier pods
(`O1_SESSION_AUTHORIZED_SECONDS`); at most `MAX_POD_ACQUISITIONS = 4` pods
per invocation and `max_pod_creations` (set it to 4) per authorization; the
spend ledger and the creation ledger live beside the AUTHORIZATION file
(not `--out`), the running pod's spend is checkpointed to the ledger every
poll, and the ledger is consulted BEFORE a pod is created. The provider
`terminateAfter` margin is bounded by the USD 5 non-compute reserve.

## Stops, strongest first

| # | Stop | Lives where | Fires when | Independent of |
|---|------|-------------|------------|----------------|
| 1 | **Driver terminate** (`terminate_and_confirm`) | operator machine, `zero_touch.py` | every session outcome — COMPLETE, ABORTED_*, eviction cleanup | the pod entirely |
| 2 | **Independent watchdog process** (`watchdog_terminate.py`) | operator machine, own session (`start_new_session`) | hard allowance reached, or the driver dies | the driver process; arming is confirmed per pod (`confirm_armed`, pod-scoped) or the pod is refused |
| 3 | **Provider `terminateAfter`** | RunPod | allowance + 30 min | everything on the operator machine; write-only at RunPod (cannot be read back), so never relied on alone |
| 4 | **Driver `monitor_timeout`** = allowance + 1800 s | `zero_touch.py` | a pod that neither completes nor exits | the pod's log protocol |
| 5 | Pod-side `BudgetWatchdog` (95 % soft stop / 100 % hard) | the pod, `production_entry` | O1 phase overruns | the driver (but not a SIGKILL) |
| 6 | FL `terminate_command` via `_emergency_close` / `--terminate-only` | the pod, FL supervisor / entry script | any FL refusal, including before the supervisor exists | the O1 driver's polling cadence |

A billing pod has stops 1–4 armed once `provision()` returns. Stop 2 is the
one the session *relies* on (confirmed per pod); 3 and 4 are backstops.
**Exception:** an ambiguous create (gateway 5xx after the pod was made, a
lost read-back) can leave a pod alive that only stop 3 covers; the driver
reconciles by name + launch nonce, and if it still cannot adopt the pod it
ends with `ABORTED_NO_POD_RECORDED_BUT_PODS_PRESENT` — an operator-visible
halt with termination **unconfirmed**.

## What the driver reads

The driver fetches the container log as `text/event-stream` (the pinned
v2 spec), decodes each `data:` frame's `line`, and takes the LAST
line-anchored marker. A body that mentions a marker token but yields no
anchored line is a decoder defect and stops the session after one pod
(`ABORTED_WITNESS_SHAPE_UNRECOGNISED`) rather than reacquiring.

| Marker line | Driver verdict | Then |
|-------------|----------------|------|
| `ZERO_TOUCH_COMPLETE` | COMPLETE | download + digest-verify results, terminate |
| `ZERO_TOUCH_ABORTED_AT_<STATE>` | deterministic failure | collect logs, terminate, **no reacquisition** |
| none, pod RUNNING | keep polling | until marker, eviction, or stop 4 |
| none, pod EXITED | **eviction** | terminate remnant, reacquire (≤ 4 pods, ≤ 2 consecutive zero-progress) |
| `REFUSED (transient)` + exit | eviction (by design) | a fresh pod may not repeat it |
| **log endpoint unavailable** (3 retries) | `COMPLETION_WITNESS_LOG_UNAVAILABLE` | O1-only session **and** pod EXITED: the durable O1 archive + nonce-bound sidecar may witness completion; combined session or RUNNING pod: never — keep polling / treat as eviction |

In a combined session the FL supervisor rewrites the O1 child's markers
to `O1_PHASE_*` and prints the session marker exactly once, at the end.
★ verified at three cut points.

## Failure classes → stop

| Failure | Marker | Stop | Reacquire? |
|---------|--------|------|------------|
| spot eviction, any phase | none (EXITED) | 1 on the remnant | yes ★ |
| pre-entry refusal: scope, fetch (401/403/404), artifact verify, env | `…_PRE_ENTRY_<STEP>` / `…_HF_SCOPE` / `…_ARTIFACT_FETCH` | 1 | no |
| pre-entry transient: hub 5xx/reset/timeout, full container disk (`No space left`) | `REFUSED (transient)` | 1 | yes |
| mkdir fails for any other reason (read-only rootfs, mis-permissioned mount, no reason) | `…_PRE_ENTRY_MKDIR` | 1 | no |
| O1 identity env missing / bad destination | `…_PRE_ENTRY_CONFIG` | 1 | no |
| O1 state abort (gate, affordability, digest) | `…_<STATE>` (O1-only) or `O1_PHASE_…` + `…_O1_HALT_OR_COMPLETE` (combined) | 6 then 1 | no |
| FL preflight refusal (config, scope, pregen) | `…_FL_PRE_ENTRY_<STEP>` | 6 (`--terminate-only`) then 1 | no |
| FL supervisor setup: FileNotFound/Permission/EROFS | `…_SUPERVISOR_SETUP` | 6 (raw-config terminate) then 1 | no |
| FL supervisor setup: ENOSPC/EIO/ENOMEM/connection | `REFUSED (transient)` | 6 then 1 | yes |
| FL allowance env missing, fresh pod | `…_COMPUTE_REMAINING_AUTHORIZED_TIME` | 6 then 1 | no |
| FL resume refusal (e.g. allowance env missing on a replacement pod) | `…_SUPERVISOR` | 6 (`_emergency_close`) then 1 | no |
| O1 phase exceeds `o1_timeout_seconds` | `…_RUN_O1_CALIBRATION` | 6 then 1 | no |
| FL state abort (ladder crash, transfer failure) | `…_<STATE>` | 6 then 1 ★ | no |
| FL `terminate_command` as shipped | — | prints a handover note and exits 0: the pod carries no provider credential, so stop 6 is **the session marker**, and stops 1–3 do the terminating | — |
| FL ladder admitted nothing | `…_RUN_FL_LADDER` | 6 then 1 | no |
| driver process dies mid-session | — | 2, then 3 | n/a — a restarted driver REFUSES while a billable pod is active (`REFUSED_UNEXPECTED_ACTIVE_POD`); recovery is stops 2/3, then a fresh session |
| operator machine dies | — | 3 | n/a |
| pod hangs silently | none | 4 then 1 | no (MONITOR_TIMEOUT is ABORTED_*) |
| container status ERROR | — | 1 | no (`ABORTED_TERMINATED`) |
| two consecutive zero-progress interruptions | — | 1 | no (`ABORTED_REPEATED_FAILURE_NO_PROGRESS`; durable rows recorded) |
| downloaded archive digest ≠ the pod's sidecar | — | 1 | no (`ABORTED_RESULT_DIGEST_MISMATCH`) |
| sidecar belongs to another launch nonce | — | 1 | no (`ABORTED_FOREIGN_RESULT_WITNESS`) |
| log shape not decodable | — | 1 | no (`ABORTED_WITNESS_SHAPE_UNRECOGNISED`) |
| ambiguous create, leftover pods found | — | 1 on every leftover | no (`ABORTED_NO_POD_RECORDED_BUT_PODS_PRESENT`; `termination_confirmed` only if every leftover confirmed) |
| ambiguous create, nothing visible yet or listing failed | — | 3 (and an operator check) | no (`ABORTED_CREATE_OUTCOME_UNKNOWN`, never reported as "confirmed") |
| evicted pod's termination unconfirmed | — | stop 2/3 on the remnant | **no** (`ABORTED_TERMINATION_UNCONFIRMED`; a second pod on top of a remnant is a double spend) |
| remaining allocation < 1,800 s | — | — | no (`ABORTED_BUDGET`: not worth a pull) |
| deterministic failure recorded by an earlier invocation | — | — | no (`REFUSED_DETERMINISTIC_FAILURE_ON_RECORD`, durable marker beside the authorization) |
| `HF_TOKEN` unset with an hf:// source | — | — | no (`REFUSED_HF_TOKEN_UNSET`, before any pod) |
| transient O1-child failure in a combined session | `REFUSED (transient)` | 6 then 1 | yes |

## The money bound

Worst realistic case: a deterministic pre-entry failure misclassified as
transient → up to 4 acquisitions × (pull + fetch ≈ 10 min) ≈ 40 min of
billing, then `ABORTED_ACQUISITION_LIMIT`. Every other path ends at the
first pod or at a genuine eviction. Nothing here can exceed the USD
allocation: the driver stops creating pods when it is spent.
