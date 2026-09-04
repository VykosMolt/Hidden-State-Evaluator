# B300 handoff — superseded and blocked

The original rental recipe is preserved at `history/2026-09-04-opus/HANDOFF_B300.md`. Do not run it.

The 2026-09-04 paid run spent approximately `$52.74` and retained no B300 validation or fit artifact. The old recipe created a paid pod without durable supervision, uploaded only on process exit, excluded unique shards, and did not terminate on successful completion.

Paid execution remains blocked until all of the following are true:

- the exact supervised-controller fault-injection suite passes;
- the account has sufficient positive balance above both reserve and maximum-budget requirements;
- a staged bundle manifest verifies locally and remotely;
- before any lease, the controller downloads that immutable stage, verifies
  its clean-commit source policy, and takes its bootstrap entrypoint from the
  verified archive rather than the live local checkout;
- the detached user service and linger preflight pass;
- every completed shard and sidecar is published under an immutable run ID and locally acknowledged by SHA-256;
- repeated API uncertainty, balance floor, budget, and runtime limits terminate the exact pod;
- termination is verified after success, failure, or controller exception.

The current safe command is the read-only preflight documented in `REPRODUCE.md`. No `create` command is an accepted workflow.
