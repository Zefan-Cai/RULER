# Repository Guidelines

## Project Structure & Module Organization
- `scripts/data/`: dataset prep and templates (e.g., `prepare.py`, `synthetic/`).
- `scripts/pred/`: model serving and client wrappers (e.g., `serve_vllm.py`, `call_api.py`).
- `scripts/eval/`: evaluation and report generation (`evaluate.py`).
- `scripts/*.sh`: orchestration (`run.sh`, `config_models.sh`, `config_tasks.sh`).
- `docker/`: container build files (`Dockerfile`, `requirements.txt`).
- Outputs: `benchmark_root/<MODEL>/<BENCHMARK>/<SEQ>/data|pred/`.

## Build, Test, and Development Commands
- Build image: `docker build -t ruler:dev -f docker/Dockerfile docker`.
- Local deps (no container): `pip install -r docker/requirements.txt`.
- Run pipeline: `bash scripts/run.sh <model_name> <benchmark>`
  - Example: `bash scripts/run.sh llama3.1-8b-chat synthetic`.
- Evaluate predictions: `python scripts/eval/evaluate.py --data_dir benchmark_root/<MODEL>/<BENCHMARK>/<SEQ>/pred --benchmark <benchmark>`.

## Coding Style & Naming Conventions
- Python: 4‑space indents, PEP 8; `snake_case` for files/functions, `UPPER_CASE` for constants.
- Bash: `set -euo pipefail` in new scripts; keep functions small and composable.
- YAML/JSONL: stable keys; follow existing fields: `index`, `input`, `outputs`, `pred`.
- Imports: standard → third‑party → local; prefer explicit over wildcard imports.

## Testing Guidelines
- Primary check is evaluation. For small changes, run a minimal task and verify `summary*.csv`:
  - `bash scripts/run.sh <model> synthetic` (reducing `NUM_SAMPLES` in `scripts/config_tasks.sh` can speed up).
- Add lightweight unit tests only where logic is isolated (e.g., text post‑processing).

## Commit & Pull Request Guidelines
- Commits: use Conventional Commits (e.g., `feat(pred): add gemini client timeout`).
- PRs: include purpose, config used, sample command(s), and before/after metrics (attach `summary.csv`). Link any issues.
- Keep diffs focused; update README snippets or comments if behavior/paths change.

## Security & Configuration Tips
- Do not commit secrets. Provide API keys via env vars used in `scripts/run.sh` (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `AZURE_*`).
- Prefer `.env`/CI secret stores; never hardcode credentials in scripts.

## Architecture Overview
- Pipeline: data prep → serving/client → prediction JSONL → evaluation CSV.
- Supported backends: `vllm`, `trtllm`, `sglang`, `openai`, `gemini`, `hf` (select via `--server_type` or `config_models.sh`).
