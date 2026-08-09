# Foundation Learner B200 V0 — runtime integration

How this package runs on the accelerator **without changing anything in the O1
package**. Everything below is either a fact bound from the O1 runner survey
(contract §22) or an explicitly labelled open item. Nothing here authorizes a
rental, and nothing here spends money.

## 1. It runs inside the existing O1 container

| item | value |
|---|---|
| image | `o1-b200-runner:v0.2.0` (local id `sha256:75582fe4…`) |
| base | `python:3.14-slim-bookworm`, pinned by digest `python@sha256:86f975ac…` |
| venv | `/opt/venv` |
| torch | `2.12.0.dev20260408+cu128` in the image (local venv is `…0407+cu128`) |
| transformers | `4.54.1` — build-asserted in the image, re-asserted at runtime by FL |
| numpy | `2.4.4` |
| env | `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `PYTHONDONTWRITEBYTECODE=1` |
| ports | none |
| container disk | 60 GB |
| checkpoint | mounted at `/artifacts/ouro_rltt_local`, never baked into the image |

**The FL package adds no image change.** No Dockerfile edit, no package
installation at start-up, no new third-party dependency: FL runs on stdlib +
numpy + torch + transformers + safetensors. `peft` is absent from
`requirements.b200.lock` and therefore absent from the container, which is why
every low-rank adapter in this campaign is implemented in
`foundation_learner/training/lora.py` (Amendment 3). `peft` 0.19.1 is used
locally, in one unit test, purely as a numerical oracle.

The exact recorded facts live in `deploy/environment_lock.json`.

## 2. Directory layout on the accelerator

```
/artifacts/ouro_rltt_local          shared, READ-ONLY (both programmes)
/artifacts/TOKENIZER_BINDING.json   shared, READ-ONLY
/workspace/o1_calibration/          O1 — FL-forbidden
/outputs/                           O1 — FL-forbidden
/opt/o1_b200/                       O1 — FL-forbidden
/artifacts/AXIS_PACKAGE_V2          O1 — FL-forbidden
/artifacts/COHORTS                  O1 — FL-forbidden
/artifacts/calibration_seed_matrix.json  O1 — FL-forbidden
/artifacts/policies                 O1 — FL-forbidden
/workspace/foundation_learner/      FL — owns runs, outputs, pylib
```

`campaign/o1_isolation.py` realpath-resolves every campaign path and refuses
anything under a forbidden root, plus any *write* to a shared read-only input.
The frozen list above is extended at supervisor start with any roots discovered
in the O1 transfer manifests: discovery only ever widens the refusal set, so a
missing O1 manifest can never enlarge FL's reach.

## 3. Session topology (why a new session-level configuration is required)

The O1 zero-touch rental **as currently sealed** terminates its pod
unconditionally after the O1 transfer, freezes its pod-spec hash into the
rental authorization, and exposes no channel for a second process. A combined
two-workload session therefore needs a session-level configuration whose pod
entry is the FL supervisor:

```
deploy/fl_b200_entry.sh --config <flb200.session_config.v1> --out <dir>
  -> python -m foundation_learner.campaign.session_supervisor
       START_SESSION
       RUN_O1_CALIBRATION            (opaque configured subprocess)
       O1_HALT_OR_COMPLETE           (declared completion markers present?)
       VERIFY_O1_RECORDS             (recompute hashes vs O1's own manifests)
       TRANSFER_O1_RECORDS           (configured command; writes a receipt)
       CLOSE_O1_PROCESS              (writes a receipt)
       RELOAD_PRISTINE_OURO          (checkpoint tree-hash verified)
       COMPUTE_REMAINING_AUTHORIZED_TIME
       RUN_FL_LADDER                 (campaign/scheduler.py)
       CHECKPOINT_AND_VERIFY
       TRANSFER_FL_ARTIFACTS         (deterministic archive)
       TERMINATE_ACCELERATOR
```

The supervisor copies *patterns* from the O1 provider machinery and never edits
or imports O1 files; the two shared hash conventions are re-implemented
identically in `ecology/base.py` instead.

### Verification without reading outcomes

`VERIFY_O1_RECORDS` recomputes SHA-256 over exactly the files that O1's own
manifests list, and compares. It parses only path/digest pairs. FL never reads
O1 calibration outcomes, transport results, difficulty selection, or analysis,
and no O1 output is ever used as a feature, a target, task selection, or
training data.

Because the FL isolation guard refuses O1 roots outright, this one step runs
through a separate, deliberately crippled `O1RecordCustodian` that (a) accepts
only paths under the declared O1 roots and (b) has **no method that returns
file content**. Hashing bytes is not reading outcomes: a digest carries no
calibration result. Every path it touches is journalled, so the claim is
auditable rather than asserted.

## 4. What is UNRESOLVED, and why

| field | why it is open |
|---|---|
| `o1_entry_command` | **operator-bound.** The sealed O1 package's pod entrypoint `o1_b200/deploy/start_b200.sh` is currently a refusing stub (`exit 64`) pending the O1 provider-adapter step. Repairing it is explicitly out of FL scope, so the real O1 launch command must be supplied by the operator in the session config. |
| `image_digest_ref` | filled from the GHCR push output of the O1 image; owned by the O1 pre-rental checklist. |
| `session_authorized_seconds` | derived at session time from the authorized budget and the observed hourly rate. |
| `available_foundation_learner_seconds` | computed on the pod after O1 closes; it is an INPUT to the FL scheduler, never an assumption. |
| measured throughput, `U`, PEFT vs FULL, eval batch size | mechanically derived on the accelerator by `BENCH` + the frozen §11 affordability rule + the batched/unbatched equivalence gate. |

A session config carrying any value that begins with `UNRESOLVED` **refuses a
real run** — both in `fl_b200_entry.sh` (pre-flight) and in
`campaign/session_supervisor.py`. A rehearsal config (`"rehearsal": true`) is
exempt and every artefact it writes is labelled `DRESS_REHEARSAL`.

`deploy/FL_SESSION_CONFIG.template.json` is the shipped template; as shipped it
refuses, by design. A real session additionally requires `fl_transfer_command`
and `terminate_command`: without them the FL artefacts would be stranded on a
pod that is still billing.

A HALT of the O1 phase (declared completion markers absent) aborts the session
at `O1_HALT_OR_COMPLETE`. FL does not proceed, because there would be no O1
records to verify or transfer, and §13 makes FL conditional on both.

## 5. Artifact staging

FL follows the same pattern the O1 runner already uses: artefacts are staged
into and retrieved from a private Hugging Face repository rather than being
baked into the image or committed to Git.

* **In:** the pregenerated data (`artifacts_fl/pregen/**`, plain JSONL plus
  `SHARD_SUMS.json`) and the FL package zip are uploaded to the staging repo
  and downloaded on the pod into `/workspace/foundation_learner/`. Every file
  is hash-verified against `PREGEN_MANIFEST.json` / `SHA256SUMS` before any
  scientific process starts. The pregenerated data is additionally *bitwise
  reproducible from code* (`scripts/pregenerate_all.py` contains no wall-clock
  value), so staging is a convenience, not a trust root.
* **Out:** `TRANSFER_FL_ARTIFACTS` builds a deterministic zip
  (`campaign/result_verifier.py`: sorted entries, 1980 timestamps, `0o644`,
  deflate 9, secret deny-list) with a `.sha256` sidecar, then runs the
  configured transfer command. The FL results repository is SEPARATE from the
  O1 results repository.
* The 5.3 GB checkpoint is never archived and never placed in Git; it is
  referenced by tree hash and mounted.

## 6. Budget

`campaign/FL_BUDGET_POLICY.json` is FL's own policy file. It is **separate**
from `o1_b200/runner/budget.py`, which hard-asserts the O1 45/40/5 split and
cannot be reused. FL freezes `SAFETY_FACTOR = 1.25` and
`FINAL_TRANSFER_RESERVE = 1200 s`, admits a stage only when
`projected * 1.25 + reserve <= remaining`, and never consumes the reserve.

The USD 45.00 total session budget remains authoritative at session level, and
**rental confirmation remains NOT AUTHORIZED**. Nothing in this package
contacts a provider, quotes an instance, or spends money.

## 7. Local validation status

`scripts/run_all_tests.py` runs the unit suite, the hostile fixtures, the
offline dress rehearsal, and (when `SHA256SUMS` exists) the package checksum
verification. All of it is hardware-independent: it uses the tiny
nonscientific Ouro model, tiny pools, a stub O1 command, and seconds-scale
budgets. **No B200 claim is made or implied by any of it.**
`scripts/local_smoke_test.py` is the one path that touches the real 2.6 B
checkpoint; it is explicitly nonscientific, watchdog-limited to 15 minutes, and
produces `SMOKE_REPORT.json` marked `NONSCIENTIFIC`.
