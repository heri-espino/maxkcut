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
data/              canonical complete 71 G-set original text instances
G-Set/             legacy MATLAB graphs; preserve raw data and provenance
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
- `data/G*.txt` is now the **canonical complete 71-instance G-set**,
  including G81; the legacy `G-Set/*.mat` remains for past experiments.
  Both CLIs default to `data/` but honor `--gset-dir G-Set`. Before
  claiming that old G11 cuts apply to canonical G11, compare all weighted
  edges with `maxkcut-sweep --compare-inputs`.
- Published Glass–Feder Table 1 covers only 17 toroidal graphs. Never
  label a best-known cut as a proven optimum unless its independent valid
  upper bound coincides. The automatic comparison.csv must distinguish
  absent, best-known, proven, size mismatches and apparent bound violations.
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

## Optional PyTorch CUDA low-rank research backend

- `src/maxkcut/lowrank.py` uses optional PyTorch with an explicit
  `cuda` or `cpu` device and **no silent CPU fallback**. It solves a
  rank-restricted, **nonconvex Max-Cut (k=2)** vector problem, **not**
  the full CVXPY/SCS SDP. It does not certify an SDP upper bound, and
  must NOT be used directly for k>2 without the Frieze-Jerrum
  pairwise constraints.
- `maxkcut-gpu` is its separate CLI. Use `--device cpu` for tiny tests;
  never initiate expensive GPU or full-G-set runs in automatic workflows.
- The GPU runs graph-times-factor sparse products and Adam updates, but
  existing thesis GW/K-Means rounding is CPU/NumPy after optimization.
  No dense n-by-n X must be constructed by this backend.
- Store one rank-r float32 factor, all optimization-start histories,
  signed graph weights and **every** rounded partition in archives.
  Distinct factor caches are keyed to input SHA256, optimizer settings,
  Torch/CUDA version and optimizer seed. `--rounding-seed` must not
  invalidate the factor cache.
- Report `GW_torchlowrank`, `K-Means2N_torchlowrank` and
  `K-MeansNM_torchlowrank` **separately** from the same names evaluated
  on full SDP vectors. Compare exact cut weights, runtime and device
  on identical graphs under disclosed budgets. Do not claim a speedup
  without measuring the actual RTX 4500 Ada GPU.
- Keep torch an **optional dependency**; ordinary environment, package
  imports, docs and small tests must not require a CUDA installation.
  PyTorch Windows CUDA wheels can be installed user-locally into a
  separate Conda environment; do not auto-install gigabytes of CUDA
  packages as part of the standard environment.

## Full convex SDP versus CUDA study

- The unchanged `maxkcut` CVXPY/SCS solver creates a full n-by-n
  convex PSD variable and runs the original three GW/K-Means rounders.
  Its primal SDP objective and `optimal_inaccurate` status are **not**
  proofs of optimality or certified dual upper bounds.
- `maxkcut-sdp-compare` is an **offline comparison and preflight tool**:
  it never launches a solver, checks normalized signed graph snapshots
  against canonical `data/Gxx.txt`, and marks missing full SDPs
  `NOT_RUN`. Preserve separate solver families and per-restart
  metrics, solver status, and actual solver wall-clock time.
- Seven previously excluded dissertation instances are G55, G56,
  G60, G61, G70, G77, G81. Inspect dimensions first with
  `--preflight-seven`. Do not launch a high-n dense CVXPY PSD
  experiment merely because one n-by-n matrix fits in system RAM.
  Large PSD cone eigenprojections and canonicalization can exceed
  available memory; retain the default full-SDP `--max-n 1000` guard.
- The CUDA low-rank approach is a separate **nonconvex** approximation,
  not a solved full convex SDP on those seven cases. Document
  actual limitations and partial completion rather than asserting
  the thesis's unsolved convex SDPs were resolved.

## Resumable G-set full convex SDP batch

- `maxkcut-sdp-batch` attempts the **original CVXPY/SCS full convex
  SDP**, one bounded subprocess per graph, with exact checksum-backed
  archive verification and persisted per-graph success, failure, timeout
  or size-skip. A completed CUDA low-rank feasible cut is *not*
  evidence of the convex SDP having been solved.
- Safe default graph cap is n <= 1000. Raising above 1000 requires
  `--allow-large`; above 3000 requires another
  `--acknowledge-huge-risk` because 64 GB Windows RAM may not
  accommodate dense PSD cone internals. Wall timeouts do not cap peak RAM.
- Large jobs are **never** launched automatically in GitHub Actions.
  CPU full-SDP cost can be orders of magnitude higher than low-rank CUDA
  cost. A batch selecting all 71 canonical graphs will report some as
  `skipped_size`, not fake success. Never remove these safety guards.
- Per-graph artifacts, original SDP numeric status, actual CPU duration,
  source graph SHA, all roundings and solver diagnostics must remain
  auditable. The offline comparison with CUDA requires normalized
  weighted graph identity.
- The user's existing confirmed G11 comparison: full SDP best rounded
  cut 546, CUDA low-rank strong best 536, canonical proven optimum 564;
  full SDP solver said `optimal` (not proof of integer Max-Cut optimum).

## Append-only CUDA result comparisons and no-solver recovery

- CUDA low-rank JSONL files may contain multiple independently archived
  attempts for the same graph. The study comparison **must never
  concatenate roundings across distinct runs** or silently select the
  maximum cut across budgets. When duplicates occur, choose the
  **latest completed independently verified archive** and record
  selected run ID/count/policy in output.
- The full SDP batch has `--compare-only`, which reads previously
  saved successful cases and CUDA JSONL and regenerates the comparison
  without ever invoking CVXPY/SCS or CUDA. A failed optional comparison
  does not invalidate previously verified full-SDP numerical results.
- The user's first n<=1000 full-SDP batch: 30 completed, 41 size
  skipped, 0 failed, 0 timed out, ~952.579 seconds. Optional
  comparison originally failed on duplicate append-only G1 CUDA
  runs; this does not imply SDP failure.

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


## Unlimited wall-time full convex SDP continuation (2026-10-25)

- `maxkcut-sdp-batch --no-wall-limit` explicitly disables both
  per-graph and global **wall-clock** timeouts while retaining
  the original **full convex SDP** under CPU CVXPY/SCS.
  Never conflate this with an unlimited SCS iteration budget;
  `--max-iters` remains a separate numerical termination condition.
- The first batch saved 30 complete canonical n<=1000 SDPs
  and 41 size-skipped graphs. Under the same output directory,
  changing ONLY execution time limits or the max-n safety gate
  permits verified resume; the original solver parameters, input
  checksum, archived partitions and rounding settings must remain
  identical. Record each invocation's execution policy without
  rewriting prior provenance.
- For n>1000 explicit `--allow-large` is still mandatory;
  for n>3000 `--acknowledge-huge-risk` is also required.
  No timeout is NOT a RAM limit; massive PSD cone eigendecompositions
  can exhaust the user's 64 GB Windows memory or crash the system.
  Prefer staged n<=2000 then n<=3000, and individually inspected
  large SDP attempts. Never automatically launch >3000 PSD cones.


## Read-only full SDP batch progress (2026-10-26)

- `maxkcut-sdp-batch --status` and `--watch-progress` may inspect
  per-graph JSON statuses, latest solver log update time and SCS
  convergence trace without starting ANY new numerical computation.
  A `started` record is not proof of process liveness after a hard crash.
- `--solver-progress` is a logging-only execution policy that
  forwards the original CLI's `--verbose` flag to **new** SCS child
  processes; old complete graphs remain verified/resumable under the
  same numerical settings. Do not silently change `--max-iters`,
  tolerances, seed, restarts, CPU threads or convex solver.
- Never present SCS iteration index divided by the iteration cap
  as true completion percentage or predict remaining wall-clock time
  from it. Primal/dual residual and duality gap are diagnostic
  convergence measures, not integer Max-Cut proof.
- No retroactive introspection of native SCS solver internals and
  no SCS iteration checkpointing are claimed. Verbose logs may buffer
  for long periods during eigendecompositions, so stale log timestamps
  are not proof of a freeze.


## GitHub publication of independently verified SDP results

- `maxkcut-sdp-publish` reads saved cases/archives from ignored
  `reports/` and constructs public, deterministic, sanitized
  `results/gset/full-sdp/` Markdown/CSV snapshots. This **never**
  launches a numerical SDP or CUDA optimizer. Do not publish raw
  Gram caches, dense matrices, solver logs, local absolute paths,
  personal account identifiers or secret data.
- The publisher must independently verify each claimed completed
  SDP's signed weighted graph SHA, method partitions and all rounding
  cuts using the existing archive validator before publishing any
  feasible cut. Incomplete/started records are not successes.
- Publish method distributions and full rounding trial scores,
  not just best cuts; preserve solver status (including
  `optimal_inaccurate`), seed, solver/BLAS settings, Git SHA and
  graph identity metadata. Do not present primal SDP scores as
  mathematically certified upper bounds or numerical solver optimal
  status as integer Max-Cut proof.
- Git operations are explicit: `--push` requires the checked-out
  `main` branch, stages only the known public result snapshot,
  never force-pushes or auto-pulls, and no-op commits when results
  have not changed. `--watch-seconds` runs synchronously in a
  second user terminal and can be stopped with Ctrl+C without
  disturbing the solver. It is not an asynchronous ChatGPT job.
- Documents: `docs/full-sdp-publication.md`; tests:
  `tests/test_sdp_publish.py`. No heavy solver calculation belongs
  in publish/test workflows.

## G81 original full convex SDP on ephemeral 128GB Linux VM

- The canonical final graph G81 has n=20,000, m=40,000.
  Preserve the **original convex CVXPY/SCS SDP** for research; never
  silently substitute the separate low-rank nonconvex CUDA method.
- `src/maxkcut/g81_vm.py` with thin `scripts/run_g81_vm.py` runs only G81 with the G-set text dataset,
  eps=1e-3, max_iters=3000, seed=2026, 20 restarts per rounding
  method, and a configurable BLAS thread count (default 16).
  No wall timeout implies neither a RAM guarantee nor convergence.
- A 128GB VM may still exhaust memory while canonicalizing a 20k
  dense PSD cone or take extremely long for PSD eigenprojections.
  Store precise failure diagnostics; no fabricated cut or completion.
- Every 15 minutes the **local VM** (not this chat) creates/pushes
  sanitized file/solver-log checkpoints into
  `results/gset/g81-128gb-vm/`, not into the pre-existing Windows
  study `results/gset/full-sdp/`. Original `reports/` remains ignored.
- On success, independently rescore all archived method partitions
  and publish compact original graph and bitpacked partition NPZ,
  sanitized manifest, full method summary and all rounding trials.
  Do not Git-push huge raw SDP matrices. No original signed graph
  or partitions must be invented or discarded.
- **Do not conflate diagnostic Git snapshots with genuine solver
  iteration checkpoints.** Current SCS/CVXPY pipeline does NOT
  serialize/restore its x/y/s or resume numerical iterations after
  VM teardown. True warm-start state storage is future engineering,
  potentially multi-GB and unsuitable for ordinary Git.
- Git commits/pushes target only dedicated G81 path; never force
  push/pull/rebase during a running solver. On push failures
  preserve local commit and report issue; on VM teardown only
  previously pushed GitHub data are guaranteed to survive.
- Tests: `tests/test_g81_vm_runner.py`; VM setup documentation:
  `docs/g81-128gb-vm.md`.


## G81 direct AMD failure and Windows/indirect fallback (2026-10-26)

- The first 128GB Windows VM G81 CVXPY/SCS 3.3.1 full convex SDP
  successfully compiled (about 436 seconds) but failed before the
  first SCS iteration: `AMD permutatation error`,
  `ERROR: init_lin_sys_work failure`,
  `ValueError: ScsWork allocation error!`.
  **No valid SDP solve or rounded cut was produced** (case status
  `failed`, elapsed 553.47s). GitHub checkpoints were published.
- Do not misdiagnose this as evidence of unavailable virtual memory:
  initialization error can reflect QDLDL ordering/index/factorization
  resource limits. 128GB is not automatically sufficient.
- The initial VM runner assumed Linux, explaining empty `system_memory`
  in checkpoints on Windows. It now calls native Windows system memory
  APIs when `/proc/meminfo` is unavailable.
- `python scripts/run_g81_vm.py --scs-indirect` uses original
  CVXPY/SCS full **convex** SDP with iterative CG linear-system solves,
  not the CUDA low-rank method. It uses separate original data
  `reports/gset/g81-vm-128gb-indirect/` and public results
  `results/gset/g81-128gb-vm-indirect/`, preserving the original
  QDLDL failure evidence, solver settings, artifacts and Git history.
  This avoids numerical-config conflicts, but does not guarantee
  memory capacity, convergence or an SCS-state checkpoint.
- Tests cover indirect flag propagation and path isolation.
  User-local test execution and numerical feasibility still required.
