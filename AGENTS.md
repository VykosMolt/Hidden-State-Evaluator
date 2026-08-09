# Rules for all agents in this worktree (Foundation Learner B200 V0)

These are USER rules. They bind every agent and subagent working here.

- BUILD/VALIDATION/PACKAGING ONLY. No cloud spend, no B200/RunPod rental, no
  pod creation, ever. Rental confirmation is NOT AUTHORIZED.
- Never modify `/home/moloch/ouro_worktrees/o1-v2-b200-runner` (sealed O1
  package) or `/home/moloch/ouro_project` (canonical repo). The frozen
  checkpoint `/home/moloch/ouro_project/models/ouro_rltt_local` is read-only;
  its tree SHA-256 must remain
  a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889.
- SEALED_TEST data stays unopened: never derive the sealed key, never read
  sealed plaintext, never consult sealed outcomes for any decision.
- Never weaken tests, hostile fixtures, gates, thresholds, or acceptance
  criteria to obtain a pass. No mocks/hard-coded outputs/silent fallbacks in
  place of requested capability. A failing check is reported, not hidden.
- The scientific contract is frozen: `foundation_learner/docs/
  IMPLEMENTATION_CONTRACT.md` + `CONTRACT_AMENDMENTS.md` (append-only
  amendments; never edit existing ones) + `FOUNDATION_LEARNER_V0_
  PREREGISTRATION.md`. No new hyperparameters, objectives, thresholds, or
  design changes beyond what an appended amendment honestly records — and
  amendments may only be made BEFORE any accelerator run, never to fit
  results.
- Python: `/home/moloch/ouro_project/venv/bin/python`. transformers must be
  exactly 4.54.1 in every scientific path. No global RNG / wall-clock in
  scientific code paths; derived seeds only.
- The real 2.6B checkpoint may be loaded only by the smoke test or a ≤15-min
  probe; never a full local training arm.
- Implementation, verification, and review must be separate fresh contexts;
  an implementer never self-certifies. Silence or missing output from a
  reviewer is never approval. Preserve BLOCKED/INCONCLUSIVE honestly.
- Evidence language: no success narrative; a null result is a result. Never
  call anything here "recursive self-improvement" or a "generally
  self-improving system".
- Current mission for the orchestrating agent:
  `foundation_learner/docs/handoff/HANDOFF_TO_SOL.md`.
