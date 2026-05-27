# Repository Guidelines

## Project Structure & Module Organization

`tradingagents/` contains the main Python package. Agent roles live under `tradingagents/agents/`, market and news providers under `tradingagents/dataflows/`, graph orchestration under `tradingagents/graph/`, and provider adapters under `tradingagents/llm_clients/`. The Typer CLI is in `cli/`, with static terminal text in `cli/static/`. Tests are in `tests/` and follow the same feature areas as the package. Visual documentation assets are in `assets/`. Generated run output belongs in `reports/`; avoid committing new ad hoc reports unless they are intentional examples.

## Build, Test, and Development Commands

- `pip install .`: install the package and the `tradingagents` console command.
- `python -m cli.main`: run the CLI directly from the source tree.
- `tradingagents`: run the installed interactive CLI.
- `pytest`: run the full test suite using settings from `pyproject.toml`.
- `pytest tests/test_model_validation.py`: run one focused test module.
- `docker compose run --rm tradingagents`: run the app in Docker.
- `docker compose --profile ollama run --rm tradingagents-ollama`: run with the Ollama profile.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and 4-space indentation. Keep modules and functions in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Preserve the existing factory and config patterns in `tradingagents/llm_clients/` and `tradingagents/default_config.py` when adding providers or options. Prefer small, typed helper functions where behavior is shared across agents or dataflows.

## Testing Guidelines

Tests use `pytest`; configured markers are `unit`, `integration`, and `smoke`. Name test files `test_*.py` and test functions `test_*`. Shared fixtures in `tests/conftest.py` set placeholder API keys and mock LLM clients, so unit tests should not require real credentials or network access. Add focused tests beside the relevant behavior, especially for provider routing, ticker handling, graph propagation, and environment overrides.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit style, for example `fix(llm): ...` and `fix(graph): ...`, plus merge commits referencing issue or PR numbers. Use a scoped subject when practical: `fix(dataflows): handle empty news response`. Pull requests should include a short problem statement, implementation summary, test results, linked issues, and screenshots or CLI output when user-facing behavior changes.

## Security & Configuration Tips

Do not commit real secrets. Keep local credentials in `.env` or `.env.enterprise`; use `.env.enterprise.example` as the Azure OpenAI template. Validate ticker and path inputs carefully, since generated reports write under `reports/`.
