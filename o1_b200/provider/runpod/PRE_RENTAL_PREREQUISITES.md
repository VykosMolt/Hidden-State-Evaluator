# External Pre-Rental Prerequisites — current state

Software readiness is PASS (263/263 local checks across 19 modules; see
`reports/RUNPOD_PRE_RENTAL_READINESS.*`). Rental remains inappropriate until
ALL of the following flip to done. None of them is runner development; each
is credential-gated account/staging work.

## The gate (all must be green before credits/authorization)

```text
RUNPOD READ-ONLY PREFLIGHT:      PASS
REMOTE IMAGE DIGEST:             RESOLVED AND VERIFIED
PRIVATE HF ARTIFACT STAGING:     PASS
POD-SIDE DOWNLOAD HASH TEST:     PASS   (stage_artifacts_hf verify)
RESULT DESTINATION ROUND-TRIP:   PASS   (stage_artifacts_hf result-roundtrip)
HF TOKEN SCOPE (READ + WRITE):   PASS   (runner.check_hf_scope)
FL PREGEN STAGED + ANCHORED:     PASS   (scripts/stage_pregen_hf.py; campaign.fetch_pregen verify)
FL HF SCOPE (pregen read +
  FL_B300_CAMPAIGN write):       PASS   (campaign.check_hf_scope --config FL_SESSION_CONFIG.json)
FL SESSION CONFIG:               RESOLVED, BAKED, validates as a real session
RUNPOD_SESSION_CONFIG:           ZERO UNRESOLVED REQUIRED FIELDS
REFERENCE RATE PROBE:            RECORDED (reports/LOCAL_REFERENCE_RATE_PROBE.json)
GHCR DIGEST RESOLVES:            docker manifest inspect <image_digest_ref>
```

Current state, 2026-08-22:

```text
RUNPOD READ-ONLY PREFLIGHT:      PASS   (2026-08-18, GET-only; 0 pods owned, $0 lifetime spend)
REMOTE IMAGE DIGEST:             RESOLVED (public GHCR; bound in RUNPOD_SESSION_CONFIG.json,
                                          CONTAINER_IMAGE_RECORD.json, FL environment_lock.json)
PRIVATE HF ARTIFACT STAGING:     PASS   (31 files, private, 5.0 GB)
POD-SIDE DOWNLOAD HASH TEST:     PASS   (6/6 re-downloaded, hashes exact)
RESULT DESTINATION ROUND-TRIP:   PASS   (probe up, down, hash-compared)
HF TOKEN SCOPE (READ + WRITE):   PASS   (auth_check + real write probe)
FL PREGEN STAGED + ANCHORED:     PASS   (2026-08-21: 64 files; anchors match the package manifest)
FL HF SCOPE:                     re-run before launch (token may have been rotated)
FL SESSION CONFIG:               RESOLVED (o1_timeout 14,000 s from the rate probe)
RUNPOD_SESSION_CONFIG:           ZERO UNRESOLVED REQUIRED FIELDS
REFERENCE RATE PROBE:            27 s/row on a laptop 5070 Ti -> ~7,000-9,000 s projected for 4,608 rows on B200
GHCR DIGEST RESOLVES:            verified with docker manifest inspect after each push
```

`package_zip_sha256` is pasted into the worktree copy after packaging (it
cannot be baked in, because this file ships *inside* the archive it would
hash). The authorization must be minted AFTER every such change: each one
changes `deployment_spec_sha256`. Set `max_pod_creations: 4` — it is the
durable counterpart of the in-process `MAX_POD_ACQUISITIONS`.

Token scope: the fine-grained token `lifetime-rltt-b300` (identity `Vykos`)
carries `repo.content.read` + `repo.write` across the `Vykos` namespace;
`auth_check` succeeded on the staging repo and a write probe uploaded and
deleted `.preflight/scope_<nonce>` in the results repo, which was verified
clean afterwards. Both repos exist and are PRIVATE.

Staging: the checkpoint round-tripped **bit-exact** — tree hash
`a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889`, the same
value the sealed freeze manifest and the FL session config pin. Record in
`HF_STAGING_RECORD.json`; round-trip record in `RESULT_ROUNDTRIP_RECORD.json`.

Live capacity at this preflight (informational only — availability, spot price
and profile selection are re-derived immediately before launch and before
every reacquisition): **secure B300 availability NONE**, so the selector fell
back to **B200** at $6.79/h secure spot in US-NE-1, stock LOW, reported as
`profile_role: FALLBACK` with `fallback_reason: "B300: secure B300
availability is NONE right now"` — an explicit refusal recorded in the
evidence, never a silent substitution.

## 1. Read-only preflight (GET-only; creates nothing)

```sh
# create a least-privilege key in the RunPod console first
# (see KEY_PERMISSIONS.md); never paste it into chat or the repo
export RUNPOD_API_KEY='<from your password manager>'
cd /home/moloch/ouro_worktrees/o1-v2-b200-runner
./o1_b200/scripts/runpod_pre_rental_readonly_check.sh
```

Must confirm live: key works; REST v2 schema still matches its pin; the
GraphQL spot contract still matches `GRAPHQL_SPOT_CONTRACT.json`; a
single-GPU Secure INTERRUPTIBLE offer exists for the primary profile (B300)
or, if refused, the explicit fallback profile (B200) — no static price cap,
the live quote is authoritative subject to the budget-viability rule; no
unexpected Pod runs.

Live catalog facts observed 2026-08-17 (read-only, informational only —
always re-query live, do not hardcode): B300 secure list $7.89/h (community
$6.94), availability NONE at query time, datacenters EU-NL-1 / EUR-IS-1;
B200 secure $6.79/h, availability LOW, US-CA-2/US-NC-2/US-NE-1.

Current live state (still to be re-confirmed at rental time, not hardcoded):
B300 and B200 both showing Low stock, Secure-filtered minimum bids equal to
the list figures above, 0 pods currently owned on the account, and the
master preflight check PASS.

### Operator profile preference

The session config's `profile_preference` field (default `["B300",
"B200"]`) lets the operator choose which of the two authorization-committed
profiles is tried first. Both profiles' canonical bodies are already
authorized by the rental authorization regardless of this order — reordering
is a preference, not a new capability, and dropping a profile from the list
does not remove it as a fallback. Set it to `["B200", "B300"]` in
`RUNPOD_SESSION_CONFIG.json` to prefer the B200, e.g. when B300 demand makes
it unobtainable (as observed above, B300 is currently NONE). Unknown profile
names are refused. A first-choice B200 acquired under this preference is
reported as `operator_preference_applied: true`, not as a fallback; a
`fallback_reason` only appears when an earlier-preference profile was
actually refused.

### Budget derivation

The committed budget is unchanged: USD 45.00 total authorized / 40.00 max
compute / 5.00 reserved non-compute. A quote is refused only when the $40
compute allocation cannot buy at least `MIN_VIABLE_SESSION_SECONDS` (2 h) —
an effective ceiling near $20.00/h. At the live rates above that is ~5.07 h
of runtime on B300 ($7.89/h) and ~5.89 h on B200 ($6.79/h). Every per-pod
deadline (in-process watchdog and RunPod's provider-side `terminateAfter`)
is armed from the REMAINING allocation net of what earlier evicted pods
already spent, so an eviction/reacquisition sequence can never authorize
more than the $40 compute allocation even if the driving orchestrator dies.
Spend is metered from POD CREATION (a pod evicted before RUNNING is not
free) and frozen only after termination is confirmed (a pod keeps billing
while terminating).

### Sealed precommit across evictions

The sealed-format calibration precommit is minted ONCE per session and
reused by every later pod from the durable store, not re-minted per pod;
per-pod hardware commitments are stored under per-digest keys so an earlier
pod's commitment is never overwritten.

## 2. Remote image publication

Follow `REGISTRY_PUSH_PROCEDURE.md` (3 commands + one visibility click).
The B300 image is already built locally as `o1-b300-runner:v0.3.11`, local
image id `sha256:6591e5a5d4554c4f031e548aa73702a80310a76da9229d39e20c15dab016ee11`
(rebuilt after the adversarial-review fixes);
the registry digest is UNRESOLVED until the operator pushes. After pushing,
use the printed REMOTE manifest digest — the immutable reference
`ghcr.io/vykosmolt/o1-b300-runner@sha256:<remote-manifest-digest>` — in
`RUNPOD_SESSION_CONFIG.json` (`image_digest_ref`). Do NOT rely on the
`v0.3.11` tag after pushing; mutable tags are refused by the adapter.
Cross-check that GHCR reports the same digest the push returned:

```sh
docker buildx imagetools inspect ghcr.io/vykosmolt/o1-b300-runner:v0.3.11 \
  | grep Digest        # must equal the digest printed by docker push
```

Note: the credential-piping step (`gh auth token | docker login …`) is
deliberately left to the operator.

## 3. Private large-artifact staging (before any billing clock)

```sh
hf auth login       # WRITE-capable token, private-repo scope
cd /home/moloch/ouro_worktrees/o1-v2-b200-runner
PYTHONPATH=. python -m o1_b200.provider.runpod.stage_artifacts_hf \
    --repo Vykos/o1-b200-staging upload
PYTHONPATH=. python -m o1_b200.provider.runpod.stage_artifacts_hf \
    --repo Vykos/o1-b200-staging verify   # = ARTIFACT DOWNLOAD TEST
```

Stages the 5.0 GB Ouro-RLTT checkpoint, tokenizer binding, axis package,
verified O1 package zip, calibration manifest, and seed matrix into a
PRIVATE Hugging Face repo (free tier; refuses to proceed if the repo is not
private), then re-downloads everything through the pod's fetch path and
verifies every SHA-256 against the transfer manifest. The pod later needs
only a READ-scoped `HF_TOKEN` env value at launch (never baked into the
image).

## 4. Result-destination round-trip (separate repo, separate scope)

```sh
PYTHONPATH=. python -m o1_b200.provider.runpod.stage_artifacts_hf \
    --repo Vykos/o1-b200-results result-roundtrip
```

Creates the PRIVATE results repo, uploads a probe archive through the pod's
upload path, re-downloads it through the driver's `hf://` download path,
and hash-compares (writes `RESULT_ROUNDTRIP_RECORD.json`). The results repo
is deliberately separate from the staging repo: the pod's WRITE token is
fine-grained to results only and can never touch the checkpoint.
`RUNPOD_SESSION_CONFIG.json` `result_source` is already resolved to
`hf://Vykos/o1-b200-results/O1_B200_CALIBRATION/results.tar.gz`; after
this test, `image_digest_ref` is the only unresolved field left.

## 5. One token, both directions (seconds; do this before adding credits)

The pod carries a single `HF_TOKEN` and needs it in BOTH directions: read,
to pull the staged checkpoint from the staging repo, and write, to push
committed rows and checkpoints to the results repo. A token fine-grained to
results only — the natural reading of §4's "WRITE token is fine-grained to
results only" — passes artifact ingestion never, and a read-only token
passes ingestion and every gate and then loses every durability push. Under
INTERRUPTIBLE capacity that second failure is the expensive one: the run
looks healthy right up to the eviction that destroys it.

So prove the token locally first:

```sh
HF_TOKEN=<the token the pod will carry> PYTHONPATH=. \
python -m o1_b200.runner.check_hf_scope \
    --read-source      hf://Vykos/o1-b200-staging \
    --write-destination hf://Vykos/o1-b200-results
```

Read is proven by `auth_check`; write is proven by writing — it uploads a
few bytes under `.preflight/` in the results repo and deletes them again, so
the token must be scoped to reach both repos. Any failure prints `REFUSED:`
and exits 2. The same check runs on the pod as the FIRST step of
`start_b300.sh`, before the multi-gigabyte fetch, and the FL entry runs its
own equivalent (`foundation_learner.campaign.check_hf_scope`) against
`fl_durable_destination` before downloading the episode corpus.

## Then, and only then

Add the separately approved credits, create the one-use rental
authorization (schema `o1b300.rental_authorization.v2`, template
`B300_RENTAL_AUTHORIZATION.template.json`), and launch
`./o1_b200/o1_runpod_b300_zero_touch.sh --authorization … --execute-authorized-rental`.
