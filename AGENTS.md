# AGENTS.md — Research and coding agent instructions

This file is the **canonical instruction set for AI coding agents** working in this repository. Read it **before editing any code or documentation**, then read `docs/agent-guide.md`, `docs/architecture.md`, and the source-grounded notes relevant to the task.

## Project identity

**Name:** `maxkcut`. **Purpose:** research and reproducible experiments on Max-Cut and Max-$k$-Cut, using semidefinite programming (SDP), randomized rounding, and clustering-based rounding. Primary motivation: Rodríguez Fernández (2021), Chapters 6–7. Do not conflate chapter 6 **G-set Max-Cut** with chapter 7 **A/B/C/D random-graph Max-$k$-Cut**.

This project is early-stage research infrastructure, **not** a mathematically certified solver. Distinguish source-grounded results, our derivations, conjectures, hypotheses, implementation, and newly reproduced measurements.

## Must-read paths

- `docs/index.md`, `docs/architecture.md`, `docs/experiments.md`, `docs/experiment-archives.md`, `docs/gset-reference-values.md`, `docs/research.md`: documentation site, design, experimental protocol, sources.
- `notes/background/README.md` and `notes/background/{01..05}-*.md`: theory and prior-art map.
- `notes/experiments/gset-cvxpy-reproduction.md`: reproduction limits and sample commands.
- `src/maxkcut/`: **all production Python code**.
- `tests/`: synthetic correctness tests.
- `environment.yml` and `pyproject.toml`: authoritative Conda environment and package metadata.

## Directory contract

```text
src/maxkcut/       reusable importable algorithms including sparse local search and sweep
scripts/           thin, backwards-compatible entry points only
tests/             fast, deterministic unit tests; small numerical smoke tests
notes/background/  sources, derivations, prior art; cite exact pages
notes/experiments/ research questions, commands, metrics, figures and reproducibility logs
docs/              Sphinx + Furo + MyST; includes API and agent docs
G-Set/             input graph files; preserve raw data and provenance
reports/           generated/ignored output, not a source of truth for published results
.github/workflows/ heavy work is manual-only
```

When contributing a new experiment, **add reusable numerical routines under `src/maxkcut/`**, a thin experiment entry under `scripts/` (or a CLI option), a corresponding test under `tests/`, and a new descriptive note under `notes/experiments/`. Never hide the only implementation in a notebook or Sphinx configuration.

## Mathematical invariants (must preserve)

1. Graph (G=(V,E,w)) is undirected and symmetrically weighted. For labels (c(i)\in\{0,\ldots,k-1\}), the objective is
   [
   F(c)=\sum_{i<j}w_{ij}\mathbf1[c(i)\ne c(j)].
   ]
2. Use the weighted Laplacian (L=D-W), (D_{ii}=\sum_j w_{ij}).
3. The SDP for (k\ge2) is
   [
   \max_{X\succeq0,\operatorname{diag}(X)=1}
     \frac{k-1}{2k}\operatorname{Tr}(LX),
     \quad X_{ij}\ge -1/(k-1)\ (k>2).
   ]
4. The (k=2) constraint (X_{ij}\ge-1) is redundant and may be omitted. **Do not omit** the (k>2) lower bound.
5. An approximate solver may return a slightly indefinite Gram matrix. The
   rounding-only PSD clip/row-normalization path is governed by a documented
   relative negative-eigenvalue threshold (currently 1e-3), warns and logs
   original defects. **Never** call the repaired matrix a certified optimum,
   original feasible SDP point, or valid dual upper bound; for (k>2) pairwise
   lower bounds may not survive repair. Reject substantial defects, and
   tighten solver tolerances before scientific claims.
6. The factorization is a Gram representation (X\approx VV^\top) with **one row per graph vertex**. PSD clipping/projection may change constraints, so log repairs and validate.
7. **GW (k=2):** one Gaussian hyperplane. **Frieze–Jerrum (k>2):** maximum projection on (k) independent **unnormalized** Gaussian directions. **Spherical (k)-means:** normalized centroids updated by cosine similarity.
8. **K-MeansNM (k=2):** antipodal initialization representing the paired GW initial partition; preserve the highest **actual cut** encountered, including the starting state. K-Means2N uses independent initialization.
9. **Clustering distortion/inertia is not the graph cut objective.** Measure and optimize each under its correct name.
10. For signed weights, shifting AIG or CG denominators requires documented formulas; **never** interpret raw cut divided by the primal SDP score as a universal approximation ratio.
11. A numerical maximization **primal objective is not automatically a certified upper bound**. Approximate solver statuses, PSD residuals, eigenvalue clipping, and dual bounds are separate concerns.

## Data provenance, claims, and citations

- Persist the **normalized weighted graph** and all rounded partitions
  using the default `src/maxkcut/archive.py` mechanism, plus graph SHA256,
  Git SHA, versions, input settings and solver diagnostics. Never overwrite
  finished archive directories; do not save one giant full Gram matrix per
  restart. Failed/incomplete archives must not be treated as successes.
- The **default remains 20 restarts per method**, not 100. Statistical
  inference must use all scores, not just maxima, and distinguish matched
  GW/K-MeansNM starts from independent K-Means2N starts. Repeated restarts
  on one graph do not constitute independent G-set instances.
- Canonical G11 optimum **564** has a matching, theoretically justified
  upper bound in Glass and Feder (2026), Table 1
  (https://arxiv.org/html/2606.28478). Other G-set reference numbers
  must be labeled as proven optimum, best-known feasible cut or upper
  bound as appropriate. Verify identity of local weighted graph before
  applying the published result.
- Maintain immutable original `G-Set/*.mat` input files. Verify identifiers, sizes and hashes before publishing results. Some checked-in graph identifiers do not match the canonical G-set enumeration; document, don't silently rename.
- Ground a claim in the original dissertation/paper or in a **reproducible experimental artifact**. Cite sections, equations and PDF pages. Do not claim a new theorem from benchmark plots.
- For **new** benchmark results record git commit SHA, command line, Conda export, solver/version/termination status, CPU/RAM/time, graph ID, number of edges, (k), seed, restarts and preprocessing.
- Do not describe a solver's `optimal_inaccurate` response as a certified optimum.
- Do not silently drop failed instances, use different SDP solutions to compare rounding methods, or compare unmatched compute budgets without disclosure.
- For all-corpus runs, use `maxkcut-sweep` as a **sparse heuristic
  baseline**, not `maxkcut --graphs ALL` (which would attempt many
  dense SDP solves). A five-restart, 20-sweep, soft 60-second-per-graph
  bounded pass is appropriate for initial exploration, not formal
  statistical significance. Results are resumable and retain every
  restart/partition. Do not conflate local-search cuts with GW/FJ
  or K-Means rounding over SDP Gram vectors.
- Use **one PR per coherent research task**. Preserve the old results when running new experiments, and label proposed variants distinctly.

## Build / test / documentation steps

```bash
conda env create -f environment.yml
conda activate maxkcut
python -m pytest -q
python -m maxkcut --graphs G11 --inspect-only
sphinx-build -b html docs docs/_build/html -W --keep-going
```

`--inspect-only` validates graph input and does **not** run an SDP. Avoid costly G-set SDPs in default tests. The Conda environment uses an editable installation (`pip: -e .`) for the `src/` layout.

**Do not introduce Ruff or impose Ruff as a required check.** Readability and tests take priority; regular Python style still applies.

## Cross-platform command examples

When responding to a user's terminal logs, identify their shell before providing copy/paste commands. **PowerShell does not use Bash backslash continuation**. For PowerShell, provide one-line commands or use a backtick as the last character on each continued line. Keep `docs/windows-powershell.md` up to date whenever CLI flags or benchmark commands change.

## Numerical hardware/accelerator policy

- Use `--hardware-info` before advising CPU/GPU flags; a GPU without a
  model/driver identification does not establish NVIDIA CUDA support.
- `--cpu-threads` limits threadpoolctl-recognized BLAS libraries and,
  when `--solver SDPA` is selected, passes `numThreads` to the native SDPA
  solver. Do not claim it parallelizes all SCS operations or guarantees that
  a native SDPA `bad_alloc` will be resolved. Windows commit limitations
  and solver internal allocations must be diagnosed separately.
- `--scs-cudss` is an explicitly requested **optional** SCS GPU linear-system
  backend; it requires a custom CUDA/cuDSS-linked SCS package. Do not
  auto-install CUDA, cuDSS or expensive GPU dependencies for regular users.
  Do not claim a GPU speedup until equivalent-fidelity SDP benchmarks exist.
- CPU-thread count and GPU mode are part of the cache identity; when timing
  backends use `--no-sdp-cache` and separate JSONL output names.
- Keep GPU-enabled convex SDP and nonconvex low-rank Burer-Monteiro methods
  separate, with different scientific labels and guarantees.

## Workflow authorization and scope

- **Never** launch a large graph SDP, numerical figure generation, or paper/PDF build as an automatic push/PR job. Heavy work requires `workflow_dispatch`, explicit operator selection and bounded inputs/resources. Lightweight tests/lint are allowed.
- Avoid compute surprises: default CLI guard `--max-n=1000`; for larger (n), require an explicit opt-in. Do not disable caps merely to make examples work.
- **The user's latest preference is to create and merge PRs automatically after appropriate validation, unless the user explicitly requests otherwise.** Do not bypass branch protection, force a merge, or claim CI succeeded when it did not. Never force-push, rewrite someone else's branch, delete user data, or modify unrelated projects.
- Do not claim tests ran unless they ran. If `cvxpy`, Sphinx or graph files are unavailable, report skipped/untested scopes.
- Do not add network calls or automatic downloads to module imports, tests or doc builds.
- Preserve backwards compatibility of `scripts/gset_benchmark.py` and `maxkcut` CLI, and document any deliberate breaking change.

## Pull request checklist

- [ ] New algorithm resides in `src/maxkcut/` with a clear API; CLI/script is thin.
- [ ] Tests cover smallest exact graph examples, signed edge weights, seeds, objective scoring and invalid inputs.
- [ ] Defaults cannot accidentally launch large SDPs or consume extensive GitHub Actions minutes.
- [ ] Notes identify source-derived vs new results and contain full reproducibility metadata.
- [ ] Sphinx pages/API updated for public functions and options.
- [ ] Existing G-set data, past notes and scripts remain accessible.
- [ ] State what was **actually run**, what passed, and what remains untested.
- [ ] PR raised against `main`; if the user has not opted out, merge after reviewing changed files and available checks.

## Recommended development flow

Read instructions → inspect the current branch/files → state the mathematical invariant being changed → create a feature branch → implement reusable module and test → run fast tests → update notes/docs → review diff for provenance and surprising compute → open PR → inspect checks and merge if permitted and the user has not opted out.

**For additional explanation and task templates**, see `docs/agent-guide.md`.
