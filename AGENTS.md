# Repository Guidelines

## Project Structure & Module Organization

`tini/` is the installable Python package. Its main areas are gateways (`gateway/`), the
model/tool cycle (`loop/`), workflows (`graph/`), capabilities (`tools/`), memory backends
(`memory/`), and observability (`ops/`). Static dashboard files are under `tini/ops/static/`
and require no frontend build. Tests live in `evals/deterministic/` and `evals/judge/`.
Contributor skills go in `skills/community/<skill-name>/SKILL.md`; supporting material lives
in `examples/`, `scripts/`, `docs/`, and `sql/`. Runtime data under `.tini/` is gitignored.

## Build, Test, and Development Commands

- `uv venv && uv pip install -e ".[dev]"` creates an environment and installs the package,
  pytest, and the pinned Ruff version.
- `uv run tini` starts the terminal assistant; `uv run tini dashboard` serves the local UI
  at `http://localhost:9000`.
- `make eval` runs deterministic, network-free tests; target one case with
  `uv run pytest -q evals/deterministic/test_models.py`.
- `make lint` checks `tini`, `evals`, and `scripts` with Ruff.
- `make gate` runs the release gate. Judge evaluations require a configured provider key.

## Coding Style & Naming Conventions

Use Python 3.11+, four-space indentation, public-interface type annotations, and a
100-character line limit. Ruff 0.16.0 is authoritative. Use `snake_case` for modules and
functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Favor direct code
over speculative abstractions. Put optional or heavy dependencies behind a `pyproject.toml`
extra.

## Testing Guidelines

Pytest is the test framework. Every behavior change needs a deterministic test named
`test_<behavior>.py`; bug fixes should add a regression case that fails before the fix.
Deterministic tests must not require network access. Keep subjective, model-scored checks in
`evals/judge/`, separate from binary correctness tests.

## Commit & Pull Request Guidelines

History favors brief subjects such as `Add provider base URL` and `Update README.md`. Write an
imperative subject under about 70 characters; reserve the body for essential rationale. Work
on a topic branch because `main` is protected. Before opening a PR, run lint and the gate. PRs
should explain behavior and verification, link the issue, and include screenshots for dashboard
changes. Discuss core dependencies or top-level packages first; CI must pass before squash merge.

## Security & Local State

Copy `.env.example` to `.env` and keep keys out of commits, logs, fixtures, and screenshots.
Do not add hidden network calls or install-time execution. Treat `.tini/` as user data: inspect
before changing it and obtain explicit approval before any reset or cleanup.
