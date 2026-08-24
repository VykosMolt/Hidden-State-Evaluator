# O1_B300_RUNNER v0.3.11 — pre-rental handoff

**Status: B300 PREEMPTIBLE SOFTWARE COMPLETE / HARDWARE UNVALIDATED**

This document is the operator-facing record of the B300 migration: what
changed, what was proven, and exactly what remains before the accelerator
can be rented. It supersedes nothing — the validated B200 packages
(`O1_B200_RUNNER_v0.1.0`, `v0.2.0`) are preserved untouched.

## 1. Target

| item | value |
|---|---|
| provider / product | RunPod / Pod / Secure Cloud |
| purchase mode | **INTERRUPTIBLE (spot)** — frozen; on-demand is no longer a production path |
| primary profile | `NVIDIA B300 SXM6 AC` — Blackwell Ultra, sm_103, CC 10.3, 288 GB HBM3e |
| fallback profile | `NVIDIA B200` — sm_100, CC 10.0, 180 GB HBM3e — **explicit, never silent** |
| GPU count | exactly 1 |
| precision | BF16 eager only |

Both profiles are covered by one rental authorization. `profile_preference`
in the session config orders which is tried first (default
`["B300","B200"]`; set `["B200","B300"]` to prefer the B200 when B300
demand makes it unobtainable). Reordering adds no capability: the frozen
profile set is closed, unknown names are refused, and a profile omitted
from the list remains available as a fallback. A first-choice B200 is
reported as a deliberate selection, not as a fallback.

## 2. Why spot acquisition uses GraphQL

The pinned RunPod REST API v2 contract has **no interruptible surface at
all** — no spot price in the catalog, no purchase-mode field on
`CreatePodRequest`, no eviction state. (RunPod's *other* REST surface,
`rest.runpod.io/v1`, does expose `PodCreateInput.interruptible` and a
readable `Pod.interruptible`, verified 2026-08-22 — but it has no bid
field and no GPU catalog/pricing path.) What needs GraphQL is therefore
**bid control and price discovery** (`gpuTypes.lowestPrice`:
`minimumBidPrice`, `stockStatus`), which exist nowhere else, plus the
`podRentInterruptable` mutation that takes the bid.

So acquisition alone runs on a separately pinned GraphQL surface
(`provider/runpod/graphql_spot.py`, contract in
`openapi/GRAPHQL_SPOT_CONTRACT.json`); everything else — status polling,
logs, billing, stop, terminate/DELETE — stays on the pinned REST v2
contract, because a spot pod is a pod. Live GraphQL introspection is
disabled by the provider, so the mutation's input contract is pinned from
the published spec and fails closed *before execution* (Apollo validates
every document against the schema, so drift errors without creating a pod
or billing anything). The read-only query surface **is** live-verified on
every preflight.

## 3. Budget (unchanged policy, mechanically derived limits)

The committed budget is untouched: **USD 45.00 total authorized / 40.00
max compute / 5.00 reserved non-compute** (the three constants are now
asserted coherent, so the reserved figure can no longer silently drift).

The obsolete `$5.89/h` static cap is gone. A quote is refused only by a
mechanical viability rule: the compute allocation must buy at least
`MIN_VIABLE_SESSION_SECONDS` (2 h), an effective ceiling near **$20.00/h**
derived from the budget rather than guessed. At the rates observed live,
that is ≈5.07 h of runtime on B300 ($7.89/h) and ≈5.89 h on B200 ($6.79/h).

Every per-pod deadline — the independent watchdog *and* RunPod's
provider-side `terminateAfter` — is armed from the **remaining** allocation
net of what earlier evicted pods already spent. An eviction/reacquisition
sequence therefore cannot authorize more than the compute allocation even
if the driving orchestrator dies.

Spend is metered from **pod creation** (RunPod bills from provisioning, so
a pod evicted before RUNNING is not free) and frozen only **after
termination is confirmed** (a pod keeps billing while terminating).

## 4. Preemption behaviour

Eviction is normal infrastructure behaviour, not scientific failure.

* **O1** — committed rows and the sealed records file stream continuously
  to the durable store. On a fresh pod the records are restored before the
  sealed v2.1 orchestrator runs, and its own record sink resumes at the
  next missing canonical row, refusing duplicates and mixed bindings. A
  partially generated row is never reconstructed; work after the last
  durable sync is simply regenerated. No O1 scientific configuration
  changes because a machine was evicted.
* **Foundation Learner** — the supervisor journal and every atomic
  training checkpoint mirror off-pod (payload first, manifest last, so a
  durable manifest implies a complete durable checkpoint). A fresh pod
  restores them before reading the journal, then the existing resume path
  proceeds exactly as after a same-pod restart: completed states skip and
  training resumes from the last valid checkpoint with hyperparameters
  bound by `arm_config_hash`. No live hyperparameter improvisation.
* **Reacquisition** re-runs every gate — fresh quote, profile
  re-selection, authorization binding, hardware gate, environment
  validation, artifact verification — and is bounded by the hard dollar
  budget, by `max_pod_creations` in the authorization, by
  `MAX_POD_ACQUISITIONS`, and by a zero-progress repeat-failure guard that
  treats a second consecutive interruption with no durable progress as a
  container defect rather than an eviction.

## 5. Environment

`torch==2.12.1+cu130` (stable), CUDA 13.0, cuDNN 9.20.0.48, NCCL 2.29.7,
Triton 3.7.1, Python 3.14, NumPy 2.4.4, **`transformers==4.54.1` exact**.
No FP8/FP4/quantization/speculative decoding/vLLM/TensorRT-LLM/SGLang —
their absence is asserted at build time.

The cu128 stack could not have launched at all: RunPod's B300 *and* B200
hosts now report CUDA 13.0/13.2 only, and the old wheel carried no sm_103
support. Native execution is evidenced by `cuobjdump` against the shipped
fatbinary (recorded in `deploy/FATBINARY_ARCH_EVIDENCE.json`): **448
`sm_100` + 59 `sm_100a` + 59 `sm_103a` cubins and zero PTX entries**. So
B200 is native and direct; B300 runs the `sm_100` cubins natively under
NVIDIA's same-major/higher-minor rule plus its own arch-tuned kernels; and
with no PTX, JIT fallback is structurally impossible — a kernel either
loads as native SASS or fails loudly in the hardware gate.

The pod keeps `HF_HUB_OFFLINE=1` so model loading provably cannot reach the
network. Durability, external commitment and result transfer reach the hub
**only** through isolated subprocess helpers spawned with the offline flags
stripped, which refuse loudly if the flag survives. Every transfer is
SHA-256 verified end to end.

## 6. Scientific isolation

No O1 axis, task, endpoint, calibration or confirmatory protocol changed.
No Foundation Learner task generator, split, stage definition, objective,
promotion rule or metric changed. The sealed O1 v2.1.0 package is imported
byte-hash-verified and never edited; calibration runs through the sealed
orchestrator, never a re-implementation. The Foundation Learner sealed test
remains unopened. Confirmatory generation remains unauthorized.

The one deliberate infrastructure substitution is `code.torch` in a
replacement freeze manifest (the sealed orchestrator runtime-asserts it
against the live interpreter). The build asserts that **only** that field
and its provenance note differ from the sealed manifest, so the claim
cannot rot.

Structural equivalence is measured **per benchmark configuration**. A
configuration without its own verdict is ineligible for selection, so the
deepest batch — which throughput ranking would otherwise prefer — can never
be chosen on a verdict measured for a different batch size.

## 6b. What real hardware measured (2026-08-24)

Five B300 pods, USD 3.07 total. Full record in
`reports/B300_HARDWARE_EVIDENCE.json`. This section exists because until this
date **every** claim here about B300 behaviour was inference.

**sm_103 native execution: CONFIRMED.** Device reports compute capability
10.3, 148 SMs, 287.4 GB. `torch.cuda.get_arch_list()` does **not** contain
sm_103 -- the wheel targets sm_100 -- and a bf16 8192-cubed matmul ran at
1763.8 TFLOP/s. With zero PTX embedded, a kernel either loads as native SASS
or raises. The same-major/higher-minor argument holds on silicon.

**Hardware gate: ACCEPTED**, identity-only and full-workload, `problems: []`.
The O1 intervention hook lands its perturbation to six significant figures
(injected_rms 0.06737900525, realized_delta_rms 0.06737968326).

**Provisioning was 5x over-stated.** Checkpoint fetch is 5.34 GB in **9 s**
(567 MB/s); full staging 5.85 GB in 10 s; model load 2.4 s; image pull 216 s
(and that happens before the container starts, so it is outside
`O1_POD_ENTRY_EPOCH` entirely). `PROVISIONING_ALLOWANCE_SECONDS` has been
corrected 1200 -> 300, returning ~15 min per pod to the FL ladder.

**The affordability model is conservative, not optimistic.** The gate assumes
~1.7 s/row = 2,118 rows/h. Measured with full 256-token rows at a modest
batch 256: **11,317 rows/h**, i.e. the whole 4,608-row calibration in
**0.41 h**. The 15x laptop extrapolation that three review rounds treated as
the biggest risk has ~5x of margin.

**Per-row latency is architectural.** `total_ut_steps: 4` over 48 layers is
~192 layer-passes per generated token, and per-row latency is FLAT (~47 s for
a 256-token row) from batch 1 to batch 64. Throughput comes entirely from
batch parallelism and never from per-row speed.

### An open scientific decision: the frozen benchmark order stops at b48

`policies/BENCHMARK_ORDER.json` declares `frozen_before_hardware_access:
true` and its deepest stage is `B200_BATCHED_w1_b48`. Measured on a B300, at
32 new tokens, throughput scales essentially linearly far past that:

| batch | tok/s | peak HBM | note |
|---|---|---|---|
| 48 | 251.8 | 12.1 GB | deepest frozen stage |
| 256 | 1121.0 | 28.1 GB | ~4.5x b48 |
| 1024 | 3178.5 | 96.3 GB | |
| 2048 | 5206.0 | 187.2 GB | ~20x b48 |
| 4096 | OOM | | ceiling at 32 tokens |

**This file has deliberately NOT been changed.** Extending a benchmark order
after seeing hardware results is precisely the post-hoc tuning the freeze
exists to prevent: the order decides which backend is selected, and choosing
it with knowledge of the outcome is not the same experiment. Whether to
re-freeze a deeper order is an explicit scientific decision for the operator,
made and recorded deliberately -- not a silent edit. (Note also that
`policies/` is inside the image source identity, so any change requires a
rebuild and re-push.)

The caveat that matters if it is revisited: the deep-batch numbers above are
at **32** new tokens. KV cache grows with sequence length, so the ceiling for
real 256-token rows is well below 2048; b128 and b256 were the only deep
batches measured at full row length.

## 7. What is NOT proven

* No B300 or B200 hardware has executed anything from this stack.
  Throughput, HBM headroom, achievable batch/microbatch sizes and the
  selected backend remain unmeasured by construction.
* The container image exists locally only; its registry digest is
  unresolved until the operator pushes it.
* That a real spot pod appears on the REST surface exactly as the mock
  models it (reconciliation consults the GraphQL listing as a redundant
  second witness, and records loudly when ownership cannot be verified by
  launch nonce).
* Live spot capacity at any future moment. Availability is a market state,
  not a software property.
