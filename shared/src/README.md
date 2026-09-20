# Source Tree

Active Python source now lives here so the repository root stays navigable.

- `hunter_seeker_core/`: Hunter-Seeker, Stockfish, ARC harnesses, diagnostics, and
  component maps.
- `hunter_seeker_v2/`: compact transactional Hunter-Seeker rebuild with scoped
  evidence, exact state graphs, role taps, and a verified ARC runner.
- `evaluator_core/`: pairwise evaluator and frozen anchor-loss primitives.
- `local_agent/`: local Ouro/RLTT wrapper, UI/server, tools, and runtime modules.
- `ouro_rltt/`: local Ouro/RLTT model-definition mirror used with
  `shared/models/ouro_rltt_local/`.

The project venv has a `.pth` entry for this source tree, and `pytest.ini`
adds the same source roots for test runs. Evaluator probes live under
`../utilities/evaluator/`; local-agent tests live under
`../utilities/tests/local_agent/`; local-agent writable state lives under
`../artifacts/local_agent/` by default.
