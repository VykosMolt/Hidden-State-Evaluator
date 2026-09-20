# Utility Tests

Project-side tests and measurement probes.

- `unit/`: Hunter-Seeker/core component and behavior tests.
- `integration/`: cross-module integration tests.
- `reports/`: report and diagnostic tooling tests.
- `local_agent/`: local-agent wrapper tests.

Evaluator/domain-transfer probes live under `../evaluator/`.

Pytest disables bytecode writes for the suite and clears repo-local
`__pycache__/` directories at session start and finish.
