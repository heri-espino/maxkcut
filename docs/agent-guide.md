# AI coding agent guide

This is a companion explanation to the **authoritative [AGENTS.md](https://github.com/heri-espino/maxkcut/blob/main/AGENTS.md)**. If instructions differ, follow `AGENTS.md` and the user's latest explicit request.

## Three agent roles

**Research reader:** Ground mathematical notes in the 2021 dissertation's Chapters 6–7 and the two supplied papers. Extract exact page/equation/algorithm numbers, distinguish author statements from deductions, and track unresolved proofs.

**Algorithm engineer:** Implement small, importable functions under `src/maxkcut/` while preserving mathematical invariants, random seeds, numerical diagnostics and old public APIs. Add tests using small graphs with known cut values.

**Experiment analyst:** Design a prespecified comparison, select baselines, record complete configurations/results and uncertainty. A Sphinx/README claim of better accuracy **requires experiment records**, never speculation.

## Task templates

### Add a rounding algorithm

1. Read `src/maxkcut/rounding.py` and relevant thesis section.
2. Document the initialization distribution (Gaussian, unit sphere, antipodal, etc.) and label/tie semantics.
3. Keep the original cut scorer from `objective.py`; never replace it with cluster inertia.
4. Add a unit test on a triangle, bipartite graph or signed graph.
5. Evaluate against FJ/GW using the **same** fixed Gram factor and comparable budgets.
6. Create `notes/experiments/<short-descriptive-name>.md`; attach plots/results only when genuinely generated.

### Add an SDP solver/backend

1. State the exact primal objective and feasible region. Any nonconvex Burer–Monteiro rank-limited approach is a *different method*, not an exact SDP reproduction.
2. Record installed solver/version, tolerances and SDP dual feasibility where available.
3. Numerically validate symmetry, minimum eigenvalue, diagonal constraints and $k>2$ off-diagonal bound after any projection. When changing the optimization formulation or its interpretation, **bump `CACHE_FORMAT_VERSION` in `src/maxkcut/cache.py`** and add invalidation tests; graph checksum alone cannot detect a changed model equation.
4. Leave `--max-n` safeguards enabled; add explicit controls for memory and iteration limits. Benchmark SCS direct versus `--scs-indirect` rather than assuming either faster.
5. Do not label approximate primal objective as a certified upper bound.

### Add a new graph family

1. Add a loader/generator with fixed RNG input and explicit weight conventions.
2. If using external benchmark data, record source URL/license/checksum. Never rename G-set files to force a match.
3. Tests cover sparse adjacency, symmetric input, signed weights and self-loops.
4. Report $n$, $m$, weight distribution and independent graph-generation seeds.

### Extend Sphinx docs

1. Add a MyST Markdown page under `docs/`.
2. Register it in the `{toctree}` in `docs/index.md`.
3. Document public functions with `{automodule}` or explicit signatures; examples must use the current importable package.
4. Build `sphinx-build -b html docs docs/_build/html -W --keep-going`.
5. Never embed large experimental outputs in doc builds. Building documentation must not run SDPs.

## Non-goals and current limitations

- This is not an automatic proof generator.
- This codebase is not a production optimizer with guaranteed solutions or full G-set replication.
- Chapter 7 already contains the spherical-$k$-means general-$k$ proposal: do not claim novelty for that step.
- The fixed-point method of Felzenszwalb–Klivans–Paul (2022) is **different** from spherical $k$-means.
- The original code's stopping and empty-cluster policies may differ from historical MATLAB implementations; document deviations.

## How to hand off an agent task

Include the repository/branch, exact files touched, source section and mathematical objective, a test command, expected artifacts, observed outputs versus untested areas, and whether a PR was opened. Prefer a small complete change over a large unfinished rewrite.
