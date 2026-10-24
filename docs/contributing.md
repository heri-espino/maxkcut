# Contributing and review

See [AI agent guide](agent-guide.md), [architecture](architecture.md), and root `AGENTS.md`.

## Development setup

```bash
conda env create -f environment.yml
conda activate maxkcut
python -m pytest -q
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Do not use Ruff as a gate. Fast tests and clear mathematical rationale matter more than formatting automation.

## Pull requests

Branch from up-to-date `main`, keep changes scoped and include:

- mathematical formulation and exact source reference;
- full explanation of new API/CLI inputs and behavior;
- tests, including numerical and invalid-input cases;
- commands that were **actually executed**, including skipped checks;
- effects on compute cost/solver memory and reproducibility;
- updated Sphinx pages and experiment notes when applicable.

Leave large runs under manual GitHub Actions invocation. PRs must not be merged without the user's explicit instruction.

## Repository data

Treat `G-Set/` as source data, not a scratch directory. Put generated results in the ignored `reports/` path and link curated results from `notes/experiments/` with provenance. Avoid committing large solver matrices or artifacts accidentally.

## Package changes

Keep `src/maxkcut/` public functions importable and preserve `scripts/gset_benchmark.py` compatibility. Add commands through `maxkcut.cli`, and ensure docs/examples run after editable install.
