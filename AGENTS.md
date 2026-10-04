# Repository Guidelines

## Documentation and interface boundaries
- Frame the project as an all-in-one model brain surgery kit; distinguish available instruments from qualified end-to-end workflows.
- `aegis-neurosurgery` is the core CLI. Advanced commands currently use `python3 -m aegis_lab.editing.neurosurgery.cli_extended`; its dispatcher does not handle inherited core commands.
- Read `docs/neurosurgery/CAPABILITIES.md` before documenting workflow/campaign results: default wrapper steps include placeholder and synthetic behavior.
- Keep current examples relative to the repository root; do not publish fixture counts as real-model quality or hardware performance.

## Project Structure & Module Organization
Primary Python sources live inside `src/aegis_lab/`, with `orchestrator/`, `workers/`, `state/`, `editing/`, `verification/`, `api/`, and `gui/` subpackages reflecting the runtime surfaces. Rust helpers are under `src/native/vpu_core/`, while shared native artifacts (QIHSE, OpenVINO archives) sit in `QIHSE/` and `scripts/`. Core relaxed data lives in `models/`, configuration sets in `configs/`, and reusable prompt pairs for ablation experiments belong in `data/ablation-smoke/`. Tests are grouped by scope in `tests/unit/`, `tests/integration/`, `tests/recovery/`, and `tests/performance/`. Documentation, diagrams, and hardware notes appear in `docs/` and `hardware_docs/`, and orchestration helpers such as `bootstrap.sh`, `launch.sh`, and `run_mission.sh` live at the repo root.

## Build, Test, and Development Commands
Install the package with `python3 -m pip install -e .` and add extras for API (`.[api]`), GUI (`.[gui]`), or hardware (`.[hardware]`). `bash bootstrap.sh` prepares directories, checks OpenVINO runtimes, and downloads required native blobs, while `bash launch.sh` brings up orchestrator, API, and workers together. Focused services can run via `python3 aegis.py orchestrator`, `python3 aegis.py api --api-port 18000`, `python3 aegis.py worker`, or `python3 aegis.py gui`. Verify hardware discovery through `./scripts/vpu_probe.sh` and map devices with `./scripts/vpu_env.sh --shell` before attempting real Myriad runs.

## Coding Style & Naming Conventions
Use four-space indentation, prefer `snake_case` for functions and modules, and `PascalCase` for classes. Keep imports organized by standard library, third-party, and local packages. Inline typing hints are acceptable where they clarify intent, but avoid over-annotating; docstrings should be short and focused on behaviour differences. Rust code in `src/native/vpu_core/` should follow `rustfmt` defaults. Formatting and linting is primarily manual, though `ruff` and `pytest` are listed as dev extras in `pyproject.toml` for future automation.

## Testing Guidelines
The suite relies on Python `unittest` discovery; add new regression coverage via `python3 -m unittest discover -s tests/<scope> -p 'test_*.py'` (e.g., `tests/integration`). Keep file names in the `test_<feature>.py` pattern and ensure test classes/methods describe the workflow, e.g., `TestWorkflowIntegration.test_full_path`. When touching validation, update `SemanticAuthority.validate_edit` to drive a real scoring model or tighten thresholds before running outside fallback paths. Coverage expectations are implicit: changes in orchestration or hardware fallback should update the appropriate scoped tests.

## Commit & Pull Request Guidelines
Commits follow sentence-style, subsystem-specific summaries (e.g., `Stabilize VPU detection`). Prefer imperative verbs, describe the change, and document any hardware assumptions in the message body. PRs require a behaviour summary, verification commands executed, and notes on required hardware or config (mention `AEGIS_REQUIRE_NATIVE_QIHSE`, `AEGIS_ENABLE_LEVEL_ZERO`, or auth tokens if relevant). Attach screenshots only for GUI surfaces and never include transient logs or build artifacts that already exist in `src/native/vpu_core/target/`.

## Hardware & Configuration Notes
Models live in `models/` for reuse, while lightweight prompt pairs for future experiments stay under `data/ablation-smoke/`. Before enabling real VPU execution, run `./scripts/vpu_probe.sh` (reports `vpu_runtime_usable`, `MYRIAD` status, and serial numbers) and then `./scripts/vpu_env.sh` with the appropriate Python interpreter to source OpenVINO 2022.3. Set `AEGIS_AUTH_TOKEN` when using IPC between components, and toggle `AEGIS_ENABLE_LEVEL_ZERO` or `AEGIS_REQUIRE_NATIVE_QIHSE` only when those runtimes are needed. When replacing stubbed validation logic, rerun the pipeline outside fallback mode to confirm the tighter scoring thresholds.
