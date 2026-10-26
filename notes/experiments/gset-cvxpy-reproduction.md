# G-set SDP reproduction: experimental runbook

**Status:** implementation and synthetic tests are committed; the full G-set SDPs have **not yet been executed or validated against the checked-in graph files**. The initial development environment could not run CVXPY, so no G-set cut scores are claimed here. The package refactor does not itself constitute a benchmark run.

## Why this experiment?

Ángel Eduardo Rodríguez Fernández's 2021 dissertation, *On the Study of Fitness Landscapes and the Max-Cut Problem*, **Chapter 6 §6.3.2, printed p. 56**, reports use of:
- **MATLAB/CVX** on G1–G21;
- Mathematica **SemidefiniteOptimization** on G22–G54, G57–G59, G62–G67, G72;
- failures to obtain the SDP for G55, G56, G60, G61, G70, G77, G81.

The related [2020 journal article](https://doi.org/10.3390/computation8030075), §§2–3, describes the same G-set Max-Cut study and 50 repetitions of randomized rounding/clustering. This runbook aims at **faithfully reproducing the SDP + vector rounding components**, then extending to Max-$k$-Cut as a separate experiment. Note that chapter **7** uses a **different** A/B/C/D random graph benchmark, not G-set.

## Source files

The repository already contains a directory of MATLAB v5 files: `G-Set/G1.mat` through `G-Set/G71.mat` (including G68, G69 and G71, which are not part of the canonical G-set enumeration). The loader in `src/maxkcut/io.py` (exposed through `maxkcut.load_graph` and the backwards-compatible CLI script) uses `scipy.io.loadmat` and can handle a dense/sparse square weighted adjacency or a single 3-column edge list. It refuses mismatched symmetry, self-loops, invalid weights, and ambiguous candidate matrices.

**Important:** the repository's sequence includes G68/G69/G71 but is missing the canonical G72, G77 and G81 entries. These must not be silently considered equivalent. These exact MATLAB files have **not** been decoded and inspected during authoring. Run `--inspect-only` first. Verify, for example, G1 is expected to have $n=800$ and $m=19,176$; G11 $n=800,m=1,600$; G22 $n=2,000,m=19,990$, per [G-set reference metadata](https://github.com/KULeuven-MICAS/openising/blob/main/ising/benchmarks/G/README.md). If inspection differs, stop before benchmarking and investigate the input file conventions.

## Installation

Python 3.11+ is recommended. From the repository root:

```bash
conda env create -f environment.yml
conda activate maxkcut
python -m pytest tests -q
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Or use any virtual environment with `numpy`, `scipy`, `cvxpy`, and a PSD-capable conic solver (the default is `SCS`).

CVXPY is the **modeling language**, not the solver. Available solvers depend on the environment; see [CVXPY solver documentation](https://www.cvxpy.org/tutorial/solvers/).

### Verify the G-set files without solving an SDP

```bash
maxkcut --graphs G11 G1 G22 --inspect-only
python scripts/gset_benchmark.py --graphs ALL --inspect-only
```

This is a fast source-audit command and **does not import CVXPY**.

## The exact SDP

With symmetric adjacency matrix $W=(w_{ij})$, degree $D_{ii}=\sum_jw_{ij}$, and weighted Laplacian $L=D-W$:

$

\begin{aligned}
\max_{X}\quad&\frac{k-1}{2k}\operatorname{Tr}(LX)\\
\text{s.t.}\quad &X\succeq0,\qquad X_{ii}=1,\\
&X_{ij}\ge-\frac{1}{k-1}\quad(k>2).
\end{aligned}

$

For **$k=2$** the lower bound is redundant, and this reduces to the standard Goemans–Williamson SDP in Chapter 6:

$

\max_{X\succeq0,\operatorname{diag}X=\mathbf1}\frac14\operatorname{Tr}(LX).

$

For $k\ge3$, the extra lower-bound inequalities recover the Frieze–Jerrum relaxation in Chapter 7 (thesis Eq. 7.6).

An approximate SDP's **primal** score is not automatically a certified upper bound. Current script logs solver status and PSD/diagonal/$k$-way diagnostics. Solvers that return `optimal_inaccurate` are accepted for exploratory rounding but should be flagged and rerun with stricter tolerances before drawing inference; a certified approximation ratio requires a reliable dual or otherwise valid SDP upper bound.

### Rounding algorithms

- **$k=2$** baseline: Goemans–Williamson with one Gaussian random **hyperplane**; this is not two independently normalized random centroids.
- **$k>2$** baseline: Frieze–Jerrum, choosing the maximum projection onto $k$ independent **unnormalized Gaussian** directions.
- **$k=2$ clustering baselines:** **K-Means2N** uses two independent normalized random centroids; **K-MeansNM** initializes opposite centroids, matches the associated GW initial bipartition, and returns the best cut encountered along the clustering trajectory (including the initial one).
- **$k>2$ alternative:** spherical $k$-means, initialized with $k$ independently sampled and normalized Gaussian centroid directions.
- Both run from the **same factored SDP solution**; the script uses independent recorded seeds and restarts.
- Scores are recomputed using the **original weighted edges**, including signed weights.

The k-means implementation uses conventional normalized centroid updates and handles empty centroids. This is a functional reproduction of the rounding idea, not a guarantee of bitwise identity with the original MATLAB initialization, convergence tolerances, or tie-breaking. Chapter 6 also explores fuzzy c-means, medoids, MST clustering and local search; those variants are **not implemented in this first version**. K-MeansNM and K-Means2N are included. They should be added before claiming a full paper-level reproduction.

## Recommended staged execution

### Stage 0: synthetic tests and data audit

```bash
python -m pytest tests -q
python scripts/gset_benchmark.py --graphs G11 G1 G22 --inspect-only
```

### Stage 1: an 800-vertex Max-Cut instance

```bash
python scripts/gset_benchmark.py \
  --graphs G11 --k 2 --restarts 10 \
  --solver SCS --eps 1e-3 --max-iters 3000 \
  --output reports/gset/g11-k2.jsonl
```

G11 has fewer edges than G1, but **both require an 800-by-800 SDP matrix**. G11 therefore is *not guaranteed* to use much less memory. If G11 succeeds, attempt G1 separately:

```bash
python scripts/gset_benchmark.py --graphs G1 --k 2 \
  --restarts 20 --output reports/gset/g1-k2.jsonl
```

### Stage 2: test $k=3$ only after establishing $k=2$

```bash
python scripts/gset_benchmark.py --graphs G1 --k 3 \
  --restarts 20 --output reports/gset/g1-k3.jsonl
```

This adds $O(n^2)$ pairwise lower-bound constraints and can be much harder than the $k=2$ SDP.

### Stage 3: attempt 2,000 vertices only on a sufficiently large system

```bash
python scripts/gset_benchmark.py --graphs G22 --k 2 \
  --max-n 2000 --restarts 10 \
  --output reports/gset/g22-k2.jsonl
```

The CLI refuses graphs with $n>1000$ by default. The $n=2000$ problem has $2{,}001{,}000$ independent symmetric matrix entries before solver overhead. Avoid opening several SDPs simultaneously. An SDP cone's memory and factorization requirements can far exceed the matrix storage alone, and runtime is solver/hardware-dependent.

### Stage 4: controlled comparisons

Use identical SDP solutions, paired graph instances, documented seeds and solver settings. Compare cut, wall clock, solver residuals and the best-of-restarts trajectory. Compare against *best-known literature cuts* as benchmarks, but do **not** call those values proved optima without a certificate.

The script appends every restart to JSONL (not only maxima). **Use distinct output paths for separate experiments** or remove an output file before rerunning to avoid accidental duplicate records. Rounding labels are not saved in this initial version; best cut scores alone are insufficient to verify a new record.

## Running GitHub Actions

The repository provides a **manual-only** workflow: `.github/workflows/gset-sdp-manual.yml`. Use **Actions → G-set SDP benchmark (manual) → Run workflow**, select one graph and $k$. It runs tests and the requested SDP and uploads a JSONL artifact. The workflow is **not** triggered by pushes or pull requests, honoring the project's policy that heavyweight numerical runs must be explicitly launched.

Default job timeout: 35 minutes. The GitHub-hosted runner might run out of memory, or a large instance might exceed the time cap. Workflow existence does **not** imply a solve has completed, or that it can reliably solve G22.

## What counts as a scientific result?

- Input and edge counts verified against independent metadata.
- PSD solver reports acceptable feasibility diagnostics and statuses.
- Graph cut recomputed exactly from original edge weights.
- Multiple random seeds and meaningful spread, not just a single lucky maximum.
- Shared SDP solution when comparing rounding algorithms.
- Failure reports, timeouts, environment specs, and elapsed time archived.
- Max-Cut $k=2$ replication results kept separate from exploratory Max-$k$-Cut $k>2$.

**Current status:** code, tests, runbook, and opt-in workflow provided; no real G-set SDP run completed during authoring.

## SDP reuse and profiling (code update)

The command-line interface now caches the original solved Gram matrix by
SHA256(input file), $k$, solver and solver-version/tolerance settings. This is
**not** an additional scientific result. Use `--no-sdp-cache` if an independent
solve is desired; use `--scs-indirect` to separately profile SCS's iterative
KKT solver. Each JSONL row includes cache-hit and timing fields. Always use a
**new JSONL output file per run** to avoid mixing old repeats with new data.
See `docs/performance.md` for commands and restrictions.

## October 2026 benchmark protocol update: archiving and certified references

A recent Glass–Feder preprint (arXiv:2606.28478, June 2026) derives valid
toroidal G-set upper bounds. Table 1 confirms that canonical G11's published
feasible 564 equals its upper bound 564: **G11 has a proven optimum**, not
just a heuristic best-known cut. This does not imply all 71 G-set instances
have proven optima; see `docs/gset-reference-values.md`. The local
`G-Set/G11.mat` needs a verified exact graph identity before transporting
the published proof to that file.

Current library experiments use 20 restarts/method by default, preserving
the full score distributions. Every graph/k run additionally archives all
vertex partitions, a signed-weight graph snapshot and a provenance manifest
under `<JSONL-parent>/artifacts/` with unique run IDs. The original SDP
Gram matrix is separately cached by graph SHA256 + solver settings and need
not be regenerated just to examine the previous cuts. Check archive integrity
with `maxkcut --verify-archive PATH`, which rescales no SDP and simply
recomputes the stored cut weights. See `docs/experiment-archives.md`.

## Full checked-in G-set first pass: protocol only (no experiments run here)

Repository inventory in October 2026 contains 71 `G-Set/G*.mat`
files (`G1.mat` through `G71.mat`). To avoid 71 large CVXPY PSD cones,
`maxkcut-sweep` has a separate **sparse one-flip local-search baseline**.
Five independent random starts, 20 sweeps and a soft time budget of
60 seconds per graph are the initial exploratory defaults; this is
**not** SDP rounding and not itself a statistical significance study.
Graph, weighted edge snapshot, all per-seed partitions, objective scores,
elapsed time, and per-graph statuses are saved under
`reports/gset/full-first-pass/`; re-running with identical parameters
verifies and skips completed cases. See `docs/full-gset-sweep.md`.

Run the existing SDP/GW/K-Means pipeline separately on feasible small
G-set instances, preserving matched problem data and distinct method
labels. **No 71-graph benchmark has been executed by the authoring agent.**

## 2026-10-24 canonical data/ correction

The repository now contains 71 canonical original-format `data/G*.txt`
files (G1–G67, G70, G72, G77 and G81), **not** only the legacy 71
`G-Set/*.mat` files. Therefore the previous claim that G81 was absent
applies only to the old directory, not the full canonical input. Both
CLIs now default to `data/`. Use `maxkcut-sweep --compare-inputs`
for an exact signed-edge audit against the legacy directory. The G11
cut 546 was produced using an older MAT run; do not claim its ratio
against canonical G11 is verified until the input comparison succeeds.

The full-corpus first pass produces `comparison.csv` with the 17
Glass–Feder Table 1 instances labeled either proven optimum or unproved
best-known cut. There are no measurements reported here from an executed
full 71-instance run. A status is only a result after actual execution.

## Optional GPU low-rank branch (2026-10-24, implementation only)

A separate Python/PyTorch CUDA path was added under
`src/maxkcut/lowrank.py`, accessible by `maxkcut-gpu`. This is **not**
a GPU reimplementation of CVXPY's full SDP: it optimizes
`X=VV.T, ||V_i||=1` using sparse graph-times-factor operations,
rank-r projected Adam and multiple independent optimizer starts.
Only Max-Cut k=2 is supported. GPU performs factor optimization;
existing GW / K-Means2N / K-MeansNM rounders then operate on CPU.
The relaxation objective is only a nonconvex primal heuristic value,
NOT a certified upper bound; all rounded partitions are scored using
the original signed weighted graph.

Repeatable experiment command (needs separately installed CUDA PyTorch):

```powershell
maxkcut-gpu --graphs G11 --device cuda --rank 32 --epochs 200 --optimizer-starts 2 --restarts 20 --seed 2026 --output reports/gset/g11-torch-cuda.jsonl
```

A distinct float32 factor cache and full experiment archives preserve
all seeds, optimizer history, factors, signed graph edges, cut partitions
and software/GPU metadata. The CPU tests do not establish an RTX 4500 Ada
runtime advantage. This branch must remain an explicit experimental
comparison against the thesis full-SDP+rounding pipeline. See
`docs/cuda-lowrank.md`. **No real GPU G11/G81 solve was performed by the
authoring agent.**

## 2026-10-25: next-stage full convex SDP comparison (planned)

The user successfully ran rank-64/500 epoch/four-start CUDA low-rank
G11 (best cut 536), plus a 71-graph CUDA rank-32 baseline. These
are nonconvex factors, NOT full SDP results. We introduced
`maxkcut-sdp-compare` to join newly computed `maxkcut` (full CVXPY
convex SDP) JSONL with saved CUDA JSONL, with *exact* canonical
normalized-graph fingerprint matching and conditional paper optima.
Unsolved graph instances remain marked `NOT_RUN`.

The first intended numerical solve is **canonical G11 (n=800)**:
`maxkcut --graphs G11 --gset-dir data --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --cpu-threads 8 --restarts 20 --seed 2026 --output reports/gset/g11-fullsdp.jsonl`.

Seven graphs (G55, G56, G60, G61, G70, G77, G81) were excluded in
Ángel Rodríguez Fernández (2021, Chapter 6) because previous SDP
software could not solve them. Check their real dimensions using
`maxkcut-sdp-compare --preflight-seven`; 20,000 vertices would need
2.98 GiB for **one** float64 Gram, but considerably more for the dense
PSD cone numerical solve. Completing GPU low-rank cuts on all seven
is not evidence that their **convex SDPs** have been solved.

**No fresh full-SDP solve has yet been executed on the user's Windows
workstation by the authoring agent.** The study runner is an offline
comparison and safety inspection tool only. Any new measured SDP
results must come from actual saved runs and independently verified
archives. See `docs/full-sdp-comparison.md`.

## 2026-10-25: verified canonical G11 full-SDP result

The user ran the original full convex CVXPY/SCS SDP for canonical G11
and joined it to the previously run CUDA low-rank result. Exact
normalized archived graph identity was `VERIFIED_CANONICAL`.
The best rounded cut was **546** (full SDP; solver status
`optimal`) versus **536** (CUDA low-rank, rank 64, 500 epochs,
4 optimizer starts and 20 rounding restarts); the proven G11 Max-Cut
optimum is **564**. A solver status `optimal` concerns the
*continuous relaxation*, not the exact discrete Max-Cut optimum.

Next, the user requested full SDP attempts on all 71 G-set graphs.
`maxkcut-sdp-batch` provides resource-capped one-graph-per-process
attempts with per-graph failure/timeout/skip reasons, archive validation
and CSV comparisons. It has a safe n<=1000 default and separate
operator authorizations for n>1000 and n>3000. **No new 71-graph
full-SDP batch has been executed by the authoring agent.** This
does not erase the seven previously excluded thesis instances:
G55, G56, G60, G61, G70, G77 and G81 remain targets for suitable
scalable *convex* SDP numerical methods. Details:
`docs/full-sdp-batch.md`.

## 2026-10-25: first 71-graph full SDP batch completed to n<=1000

The user ran the real `maxkcut-sdp-batch` on 71 canonical files,
with `--max-n 1000 --seconds-per-graph 600 --total-seconds 3600
--cpu-threads 8 --restarts 20`. Results: **30/30 attempted SDPs
complete**, 41 skipped by size cap, 0 failures, 0 timeouts, total
wall **952.579 seconds**. Outputs are saved under
`reports/gset/full-sdp-batch/` on the user's workstation. This
does not prove any integer Max-Cut optimum or solve any of the seven
originally excluded large convex SDPs.

A post-batch comparison failed due to duplicate archived runs for G1
in the user's append-only `all71-cuda-baseline.jsonl`, NOT because
SDP failed. The study comparator now selects the latest complete
archived CUDA run per graph without mixing restarts or cherry-picking
the largest cut. A `--compare-only` flag rebuilds the comparison
CSV from the saved SDP results without launching any numerical solver.
The selected CUDA run ID and candidate count are recorded for audit.


## 2026-10-25: unlimited wall-time staged full convex SDP continuation

Following the 30/30 completed n<=1000 original full convex CVXPY/SCS
SDPs, the user requested removing per-graph and overall wall-clock
timeouts for as-yet unsolved graphs. The experimental runner now
accepts `--no-wall-limit` and permits same-directory continuation
when only execution time budgets and n cutoff change, retaining
the old solver settings and archived successful partition evidence.
Example: `maxkcut-sdp-batch --max-n 2000 --allow-large
--no-wall-limit --cpu-threads 8 --restarts 20
--output-dir reports/gset/full-sdp-batch`.
It continues only eligible uncompleted graphs, and automatically
produces the verified SDP-versus-CUDA report after normal completion.

This change removes **time limits only**. It does NOT turn the
sparse/nonconvex CUDA solver into a full SDP, eliminate finite SCS
max-iterations, or guarantee that very large dense convex SDPs fit
in 64 GB. The seven excluded thesis instances remain an eventual,
separately controlled research target.

## 2026-10-25: n<=2000 full convex SDP extension completed

The user continued the existing resumable `maxkcut-sdp-batch` with
`--max-n 2000 --allow-large --no-wall-limit --cpu-threads 8
--restarts 20 --output-dir reports/gset/full-sdp-batch`.
All 30 previously completed n<=1000 graph archives were verified
and skipped. **Twenty-one additional 2000-vertex full convex SDPs
(G22–G42) completed**, with zero solver failures and zero timeouts
in this invocation. New runtime **4,940.172 seconds**. Total
successful full SDP graph runs: **51 of 71**, with **20 larger
instances size-skipped**. The automatic offline comparison generated
71 rows (51 successful full SDP graph sources), including full SDP
vs previous CUDA low-rank results. The seven originally omitted thesis
instances have not yet been attempted with the new full SDP backend.

The user's `pytest -q` output showed **89 passed / 1 failed**
because a synthetic regression test for mixed SDP run IDs omitted
the newly required `restart` field. This test fixture has now
been corrected, without modifying the optimization algorithm or
the numerical results. A test pass still requires verification
on the user's Windows checkout.

Next experiment is the explicit n=3000 stage (G48, G49, G50),
preserving the same numerical settings and output directory. Users
should review solver logs and memory use before attempting 5000+
vertex dense convex SDPs. No wall-clock timeout is not a RAM bound.

## 2026-10-26: read-only monitoring and verbose full-SDP convergence

While running the very large original convex SDP batch, the user
observed that G56 appeared stuck; CPU-time increases showed substantial
compute but did **not** establish convergence. The batch launched SCS
with `verbose=False`, so no SCS primal/dual residual or iteration
trace could be recovered from the currently running process.

An optional `--solver-progress` execution flag now forwards verbose
SCS logging for **newly launched** graphs, recording convergence traces
in per-attempt `solver.log` without altering the full SDP
mathematical program, solver parameters, cache key or earlier
verified results. The separate `--status` and `--watch-progress`
commands read saved case status and parse any available SCS numerical
trace, without creating any numerical solver subprocess. It identifies
a `started` record as potentially stale and never invents a percent
complete or an ETA.

`--solver-progress` is not retroactive, requires launching a new
graph process, and native stdout may be buffered. The 3000 SCS
iteration stopping criterion remains a cap, not a guaranteed number
of iterations to successful convergence.


## 2026-10-26: publishing numerical SDP progress to supervisor on GitHub

The supervisor uses GitHub for project review. Actual full convex
CVXPY/SCS batch data live under the ignored `reports/`, and local
numerical results are **not automatically visible on GitHub**.
A new `maxkcut-sdp-publish` CLI exports source-verified compact
Markdown/CSV snapshots into tracked `results/gset/full-sdp/`
without running any optimizer. Published source files include
full rounding distributions per method, per-graph solver statuses,
provenance metadata and optional exact weighted graph-verified
SDP vs low-rank CUDA comparisons. The exporter excludes dense
Gram matrices, raw logs and identifying local absolute paths.

To publish once: `maxkcut-sdp-publish --push` (explicit push,
requires local Git authentication and `main` checked out).
To publish during a long local solve from another PowerShell
terminal: `maxkcut-sdp-publish --watch-seconds 900 --push`.
It checks for updated finished cases every 15 minutes, does nothing
if files are unchanged, never force-pushes, and does not interfere
with the optimizer. **No user's local numerical reports were copied
to GitHub by the code authoring agent.** Data appear only after
the user executes the publisher on their Windows workstation.

## 2026-10-26: first result summary committed to GitHub

The tracked `results/gset/full-sdp/summary.csv` was first
published by the user's workstation. At the time inspected, that
file showed **59 completed graph instances** (G1–G59) and G60
marked `interrupted`. It reported:

| Graph | Best feasible rounded cut | Numerical CVXPY/SCS status |
| --- | ---: | --- |
| G55 | 9,935 | `optimal_inaccurate` |
| G56 | 3,665 | `optimal_inaccurate` |
| G57 | 3,316 | `optimal` |
| G58 | 18,486 | `optimal` |
| G59 | 5,322 | `optimal` |

G56 was listed as successful on the **second** attempt. These
are source-file *reported outputs*, not independently rerun or
numerically certified by the authoring agent. An
`optimal_inaccurate` solver status is materially weaker than
`optimal`, and neither proves an integer Max-Cut optimum.

The initial CSV commit accidentally included absolute local
Windows source and artifact paths, which the new publication
mechanism eliminates in its regenerated public snapshots. Original
signed-edge graph identity checks, individual rounding score
validation and solver provenance are emphasized for the supervisor.

## 2026-10-26: dedicated G81 128 GB VM experiment plan

G81 is the last and largest canonical G-set case, with 20,000
vertices and 40,000 edges. A new `scripts/run_g81_vm.py`
infrastructure harness automates running the **original full convex
CVXPY/SCS** relaxation at eps=1e-3, max 3000 SCS iterations,
20 restarts per method (FJ, K-Means2N, K-MeansNM), seed=2026,
and a configurable CPU BLAS worker pool (16 by default).
It does not use the CUDA low-rank surrogate.

The user has access to a temporary Linux VM with 128GB RAM.
This RAM budget is materially larger than the original 64GB
desktop machine but cannot guarantee that the dense PSD cone
will fit or complete; cubic eigendecomposition costs remain.
No numerical results or timings for G81 are claimed by this
code change.

Because ephemeral VM contents disappear, the runner periodically
exports UTC snapshots of SCS residuals/iteration, OS CPU/RAM metrics,
case status and sanitized solver/orchestration log excerpts, then
Git commits/pushes only `results/gset/g81-128gb-vm/`
approximately every 900 s. This does not disturb the earlier
59-graph Windows study's own publication directory.

After a genuine completed G81, it revalidates the signed weighted
graph and all 60 rounded partitions and publishes their compressed
NPZ data, method stats and traceability into Git; giant dense
Gram matrices remain locally ignored. A hard VM failure may
lose the in-flight solver state and raw files. Crucially:
**no SCS x/y/s warm-start checkpoint serialization is implemented**.
An interrupted G81 must restart its SDP numerical iterations.
See `docs/g81-128gb-vm.md` for exact clone/Conda/run/Git commands.
