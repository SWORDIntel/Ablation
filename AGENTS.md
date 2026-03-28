# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives in `src/aegis_lab/`, split by subsystem: `orchestrator/`, `scheduler/`, `workers/`, `hardware/`, `editing/`, `state/`, `api/`, and `gui/`. The unified entry point is `aegis.py`. Tests are grouped under `tests/unit/`, `tests/integration/`, `tests/recovery/`, and `tests/performance/`. Native components live in `src/native/vpu_core/` (Rust `cdylib`) and `QIHSE/` (external native library used by the state layer). Operational docs are in `docs/`; hardware references and diagrams are in `hardware_docs/`. Runtime configs and sample data live in `configs/` and `data/`.

## Build, Test, and Development Commands
Install the Python package in editable mode with `python3 -m pip install -e .`. Run `bash bootstrap.sh` before hardware-aware work; it creates expected directories, checks OpenVINO, and validates native dependencies. Use `bash launch.sh` for the full local stack, or run focused services with `python3 aegis.py orchestrator`, `python3 aegis.py api`, `python3 aegis.py worker`, and `python3 aegis.py gui`. For the canned ablation flow, use `./run_mission.sh --model models/qwen2.5.gguf --target refusal`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and short docstrings or comments only where logic is non-obvious. Keep imports grouped as standard library, third-party, then local modules. Match the current codebase’s lightweight typing style when touching typed files. Rust code in `src/native/vpu_core/` should follow standard `rustfmt` conventions even though no repo-wide formatter config is checked in.

## Testing Guidelines
The test suite is built around `unittest` discovery. Run everything with `python3 -m unittest discover -s tests -p 'test_*.py'`, or target a suite such as `python3 -m unittest discover -s tests/integration -p 'test_*.py'`. Name new tests `test_<feature>.py`, and keep test classes and methods descriptive, for example `TestWorkflowIntegration` and `test_full_workflow`. Add or update integration coverage when changes affect orchestration, IPC, or hardware fallback behavior.

## Commit & Pull Request Guidelines
Recent history uses concise, descriptive subjects with sentence-style capitalization, often starting with an action verb such as `Update`, `Complete`, or `Final production polish:`. Keep commit messages specific to the subsystem changed. Pull requests should summarize behavior changes, list verification commands run, note hardware assumptions, and attach screenshots only for GUI changes. Do not include generated logs or build outputs such as `orchestrator.log` or `src/native/vpu_core/target/` unless the change explicitly targets generated artifacts.
