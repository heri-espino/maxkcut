# maxkcut

Python research library for **Max-Cut** and **Max-$k$-Cut** using semidefinite relaxations, randomized rounding, and spherical $k$-means rounding.

The project is motivated by Ángel Eduardo Rodríguez Fernández's 2021 dissertation (Chapters 6–7). It is experimental research software: **solver success does not imply an exact optimum or a new approximation guarantee**.

## G77 (14,000 vertices): safe full convex GPU SDP preflight

G77 is the second-largest checked-in G-set graph by vertex count
(G81 has 20,000). Because full 14k×14k GPU PSD projection
may exceed RTX 4080 SUPER 16GB VRAM and WDDM limits, test
**one iteration only**, with the original convex SDP, and
**skip the separate 14k×14k CPU rounding factorization**:

```powershell
python scripts/precision_psd_benchmark.py --graph G77 --backends gpu-admm32-adaptive --rho 1 --admm-iters 1 --admm-abs 1e-5 --admm-rel 1e-4 --log-every 1 --restarts 1 --skip-rounding --timeout-seconds 1200 --output-dir reports/gset/g77-gpu-full-psd-1iter --publish-dir results/gset/g77-gpu-full-psd-1iter --push
```

**This is a resource/feasibility probe only**, without a
converged SDP solution or cut score. Inspect the GPU memory
and solver log before attempting ten iterations or more.
See [Windows G77 full SDP preflight](docs/g77-full-convex-gpu-preflight-windows.md).

## Adaptive penalty for full convex GPU SDP (RTX 4080 SUPER)

A controlled G11 comparison now supports **fixed vs adaptive rho**
for GPU float32/float64 full-matrix PSD-projection ADMM, preserving
the original SDP and original fixed-rho baseline. Run in **Windows 11
PowerShell** (one line):

```powershell
git pull --ff-only origin main
python -m pytest -q
python scripts/precision_psd_benchmark.py --graph G11 --backends gpu-admm32 gpu-admm32-adaptive gpu-admm64 gpu-admm64-adaptive --rho 1 --admm-iters 3000 --admm-abs 1e-5 --admm-rel 1e-4 --rho-update-interval 25 --rho-balance-mu 10 --rho-scale 2 --rho-min 0.0625 --rho-max 16 --cpu-threads 8 --restarts 20 --timeout-seconds 1800 --output-dir reports/gset/g11-adaptive-rho-strict --publish-dir results/gset/g11-adaptive-rho-strict --push
```

Every rho change is recorded, with correctly rescaled scaled-dual
iterates. GitHub publishing runs after each backend; adaptive convergence
and numerical precision are **not guaranteed**. See
[the Windows adaptive rho runbook](docs/g11-adaptive-rho-gpu-admm.md).

## G11 RTX 4080 SUPER precision and GPU PSD projection benchmark

A reproducible small-graph study compares original **CPU SCS
float64**, alternative **CPU SDPA float64**, and experimental
**GPU full-rank convex SDP PSD-ADMM float32/float64**.
Independent CPU/GPU full eigenprojection benchmarks measure kernel
speed separately from complete SDP solve time.

```bash
python scripts/precision_psd_benchmark.py --graph G11 --cpu-threads 8 --restarts 20 --admm-iters 120 --eigh-repeats 3
git add results/gset/precision-psd-benchmark
git commit -m "results: compare G11 CPU SDPA SCS with GPU full SDP precision"
git push
```

Requires compatible CUDA-enabled PyTorch and optional
`sdpa-python`; `docs/g11-precision-gpu-psd-sdpa.md` has the
complete Conda setup and validation procedure. Solver residuals,
precision differences, and all random-rounding trials are saved to
GitHub-ready results. **GPU PSD-ADMM uses the same convex SDP but
is a different first-order algorithm from SCS/SDPA**; unconverged
iterations cannot be interpreted as an optimal solution.



```bash
git clone https://github.com/heri-espino/maxkcut.git
cd maxkcut
conda env create -f environment.yml
conda activate maxkcut
python -m pytest -q
```

`environment.yml` installs the package **editable** from `src/maxkcut/` and includes CVXPY, SCS, NumPy, SciPy, pytest, Sphinx, Furo and MyST. If the environment already exists, use `conda env update -f environment.yml --prune`. Pip fallback: `python -m pip install -e '.[dev]'`.

## Inspect graphs first (no SDP solve)

```bash
maxkcut --graphs G11 G1 G22 --inspect-only
# or python -m maxkcut --graphs G11 --inspect-only
```

## Run a single controlled experiment

```bash
maxkcut --graphs G11 --k 2 --restarts 10 --solver SCS \
  --eps 1e-3 --max-iters 3000 --output reports/gset/g11-k2.jsonl
```

**Windows PowerShell users:** Bash uses a trailing backslash to continue long commands, but PowerShell does not. Use the one-line command below, or a PowerShell backtick as the last character of each continued line:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --restarts 20 --seed 2026 --output reports/gset/g11-first.jsonl
```

See the [Windows PowerShell guide](docs/windows-powershell.md) for the complete run, caching, solver comparison and result inspection.

The default `--max-n 1000` guard prevents unexpectedly large runs. The legacy entry point `python scripts/gset_benchmark.py` still works once the environment is active. Expensive GitHub Actions experiments remain **manual-only**.

## Import the library

```python
from pathlib import Path
import numpy as np
from maxkcut import load_graph, edge_arrays, cut_weight

graph = load_graph(Path("G-Set/G11.mat"))
edges = edge_arrays(graph)
print(graph.shape, len(edges[0]))
print(cut_weight(edges, np.zeros(graph.shape[0], dtype=int)))
```

Core public API: `load_graph`, `edge_arrays`, `cut_weight`, `solve_sdp`, `factorize`, `round_fj`, `round_spherical_kmeans`, `benchmark`.

## Performance tuning and reproducible SDP caching

**Numerical PSD note:** SCS may return a small negative Gram eigenvalue at
a loose tolerance. The rounding factorization now records and clips *small*
negative eigenvalues with a warning; it still rejects large violations. This
does **not** certify an SDP bound. To insist on a fresh, more accurate solve,
use `--eps 1e-5 --max-iters 20000 --no-sdp-cache` (potentially slower).
See [Windows PowerShell troubleshooting](docs/windows-powershell.md).

The first G11 run saves an SDP Gram matrix in `reports/gset/sdp_cache/`.
Re-running with a different seed or restart count **reuses the same SDP**
when the graph file and solver settings are unchanged.

```bash
maxkcut --list-solvers
maxkcut --graphs G11 --k 2 --restarts 10 --output reports/gset/g11-first.jsonl
maxkcut --graphs G11 --k 2 --restarts 20 --seed 2027 --output reports/gset/g11-second.jsonl
# Separately benchmark SCS iterative KKT mode
maxkcut --graphs G11 --k 2 --scs-indirect --output reports/gset/g11-indirect.jsonl
```

See [the performance guide](docs/performance.md) for bottlenecks, cache safety,
timings, and the limitations of CVXPY/SCS.

## First pass across every G-set instance (no SDP)

The repository includes **71 canonical text graphs in `data/`**: G1–G67, G70, G72, G77 and G81. The older `G-Set/` contains 71 legacy MATLAB files. They are separate datasets until weighted-edge identity is verified. Our
current dense SDP formulation is not a practical first-pass solver for all
71 at once, so there is a separate, **sparse signed-weight local-search**
baseline with resumable, fully archived results:

```powershell
git pull origin main
python -m pip install -e .
python -m pytest -q
maxkcut-sweep --inspect-only
maxkcut-sweep --compare-inputs
maxkcut-sweep --max-graphs 2 --output-dir reports/gset/full-first-pass
maxkcut-sweep --output-dir reports/gset/full-first-pass
maxkcut-sweep --compare-only
```

Default: **5 random starts, at most 20 one-flip sweeps/start, a soft
60-second graph budget**, sequential execution, per-graph restart-safe output.
The second batch command verifies/skips intact cases from the first. Results
include `summary.csv`, per-trial JSONL, graph weights, every cut partition
and full archive manifests. This is **not** the CVXPY SDP/GW/K-Means experiment;
see [the full G-set sweep guide](docs/full-gset-sweep.md).

## Reusable experiment archives and published benchmark proofs

The default is **20 rounding restarts per method**, not 100. The JSONL file
records every cut and numerical diagnostic. Every completed graph/k case
now additionally saves a self-contained archive of the normalized signed
graph edges, **every vertex partition** (bit-packed for Max-Cut), the
input file SHA256, Git commit, Python/package versions, solver parameters
and summary statistics.

The SDP Gram matrix remains in the existing fingerprinted cache and is
stored **once**, not once per restart. Future runs can reuse that SDP
without re-solving; archived partitions can be rescored without any SDP.
The command prints its archive path. Verify it using:

```powershell
maxkcut --verify-archive "reports/gset/artifacts/YOUR-RUN-ID/G11-k2"
```

See [experiment archives](docs/experiment-archives.md) for structure and
validation, and [proven G-set reference values](docs/gset-reference-values.md).
Notably, G11=564 was proved optimal in 2026 by a matching feasible cut and
toroidal upper bound, whereas many other G-set cut values are only
best-known heuristic results.

## Safely attempt the convex SDP across all 71 G-set graphs

G11 is now **verified on identical canonical input**: full convex
CVXPY/SCS SDP + Ángel clustering best cut 546 (numerical solver
`optimal`), versus CUDA nonconvex low-rank best 536, and the published
G11 optimum 564. To *attempt* every eligible full convex SDP without
launching huge numerical matrices in one process, the new
`maxkcut-sdp-batch` command runs each graph in a separate **CPU**
subprocess with bounded runtime and verified resume. It does **not**
silently replace the full SDP with CUDA low-rank. It reports larger
instances as **skipped**, never as solved.

```powershell
git pull --ff-only origin main
python -m pip install -e .
python -m pytest -q
maxkcut-sdp-batch --dry-run
maxkcut-sdp-batch --max-n 1000 --seconds-per-graph 600 --total-seconds 3600 --cpu-threads 8 --restarts 20 --output-dir reports/gset/full-sdp-batch
```

To continue the **remaining unsolved SDP graphs with no artificial
wall-clock timeout**, keep the original archive directory, preserve
solver/seed/restart settings, and increase the n cap gradually:

```powershell
maxkcut-sdp-batch --max-n 2000 --allow-large --no-wall-limit --cpu-threads 8 --restarts 20 --output-dir reports/gset/full-sdp-batch
```

The first 30 successful SDP cases are independently verified then
skipped; only newly eligible graphs run. The current implementation
still has **SCS's numerical max iteration cap of 3000** and the
machine's physical RAM limit. Raising `--max-n` beyond 3000 requires
`--acknowledge-huge-risk`: a huge dense PSD cone may exhaust RAM even
if there is no time limit. See
[unlimited-time full-SDP continuation](docs/full-sdp-batch.md).

The same command resumes verified successes. If your CUDA JSONL is
append-only and contains multiple archived runs of the same graph,
the comparison deterministically selects the latest **complete and
independently verified** run per graph (not the highest-cut run).
The comparison CSV records `cuda_run_id`, `cuda_candidate_runs`
and `cuda_selection_policy`. To regenerate an interrupted post-batch
CSV comparison **without rerunning any SDP**, use:

```powershell
maxkcut-sdp-batch --compare-only --output-dir reports/gset/full-sdp-batch --cuda-results reports/gset/all71-cuda-baseline.jsonl
```

The same command resumes verified successes. If
`reports/gset/all71-cuda-baseline.jsonl` exists, the batch also writes
an archive-verified comparison `sdp-vs-cuda.csv`; it separately saves
`summary.csv`, cases, solver logs and full partitions. The 64 GB
Windows PC should **not** attempt all 20,000-vertex dense full SDPs
blindly. `--allow-large` is needed to raise n > 1000; n > 3000
requires an additional explicit `--acknowledge-huge-risk`. Neither
a wall timeout nor these flags guarantees protection from RAM exhaustion.
Read [the full SDP batch safety and comparison guide](docs/full-sdp-batch.md).

## G81 — 128 GB VM, original full SDP, GitHub checkpoints

For the final and largest canonical G-set graph (**G81, 20,000
vertices**), use the dedicated Python VM launcher on Windows or Linux
with the existing Conda environment:

```bash
python scripts/run_g81_vm.py --cpu-threads 16 --checkpoint-seconds 900
```

It executes the **original full convex CVXPY/SCS SDP**, with SCS
numerical progress logging. Every 15 minutes the launcher commits
and pushes a small G81-only checkpoint (RAM/CPU, latest available
iteration/residuals, sanitized logs) to
[results/gset/g81-128gb-vm](results/gset/g81-128gb-vm/).
After independent validation of a successful G81, it also publishes
all 60 original rounded partitions and methods' cut scores,
excluding the multi-GB SDP Gram matrix and local secret paths.

**Observed first G81 failure:** CVXPY compiled successfully, but
SCS sparse-direct AMD/QDLDL initialization failed before any solver
iterations. On Windows the alternative SCS indirect mode can be
attempted without replacing the original convex SDP:

```powershell
python scripts/run_g81_vm.py --scs-indirect --cpu-threads 16 --checkpoint-seconds 900
```

This writes to separate `reports/gset/g81-vm-128gb-indirect/`
and `results/gset/g81-128gb-vm-indirect/` folders rather than
overwriting failed evidence. **These are file/log checkpoints,
not saved SCS iteration state.**
An interrupted G81 solve must still restart numerical SCS from the
beginning; 128 GB RAM does not guarantee it fits/finishes.
See [the complete clean-VM setup, commands and limitations](docs/g81-128gb-vm.md).

## Publish full convex SDP results to GitHub for supervisors

The actual full SDP G-set experiments run locally and save results
under ignored `reports/`. **Your supervisor cannot see those files
until you explicitly publish them.** The separate publication command
creates small, privacy-conscious, independently archive-verified Git
snapshots in tracked `results/gset/full-sdp/` without launching any
SDP or CUDA calculation.

```powershell
maxkcut-sdp-publish --output-dir reports/gset/full-sdp-batch --push
```

For ongoing GitHub progress updates **while the solver runs in another
PowerShell window**, keep this second command active:

```powershell
maxkcut-sdp-publish --watch-seconds 900 --push --output-dir reports/gset/full-sdp-batch
```

Only a changed publication creates a new Git commit. The publisher
requires `main`, never force-pushes or runs `git pull`, and
does not upload raw dense SDP matrices, Windows paths, or logs.
The GitHub folder includes `README.md`, case statuses, per-method
statistics, all recorded rounding scores, solver provenance and a
graph-verified CUDA comparison when available. Full
[publication instructions](docs/full-sdp-publication.md).

## Monitor full convex SDP convergence

The read-only progress view can be run in a second PowerShell terminal
without stopping or restarting an existing CPU convex SDP batch:

```powershell
maxkcut-sdp-batch --status --output-dir reports/gset/full-sdp-batch
maxkcut-sdp-batch --watch-progress --output-dir reports/gset/full-sdp-batch
```

For future graph attempts, use `--solver-progress` in the same
`maxkcut-sdp-batch` solve command to save SCS verbose iteration,
primal/dual residual and duality-gap diagnostics to each
`runs/Gxx/attempt-NNN/solver.log`. A monitor cannot obtain numerical
progress retroactively from an already non-verbose G56 solve, and
an SCS iteration/max iteration count is **not a completion percentage**.
See [full SDP batch progress](docs/full-sdp-batch.md).

## Full convex SDP versus CUDA low-rank comparison

The original `maxkcut` command **already solves the full convex SDP**
with CVXPY/SCS. A new lightweight `maxkcut-sdp-compare` command compares
its archived G-set cut results against an existing `maxkcut-gpu` CUDA
low-rank run **only after verifying identical normalized weighted graph
snapshots**. It reports the three GW/K-Means methods separately, per-method
best and mean cuts, numerical SDP solver status, cache-independent
optimization durations, published G-set references and conditional proven
optimality gaps. Graphs with no successful full SDP remain `NOT_RUN`.

From **Windows PowerShell** in the repository root:

```powershell
git pull --ff-only origin main
python -m pip install -e .
maxkcut-sdp-compare --preflight-seven
maxkcut --graphs G11 --gset-dir data --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --cpu-threads 8 --restarts 20 --seed 2026 --output reports/gset/g11-fullsdp.jsonl
maxkcut-sdp-compare --sdp reports/gset/g11-fullsdp.jsonl --cuda reports/gset/g11-cuda-strong.jsonl --output reports/gset/g11-sdp-vs-cuda.csv
```

The **seven graph instances omitted by the thesis** are
G55, G56, G60, G61, G70, G77, G81. The preflight command reports each
graph's n and single dense-Gram RAM **lower bound**. Do not blindly use
`--max-n 20000` with the full CVXPY SDP on a 64 GB RAM workstation:
semidefinite-cone eigenprojections and temporary allocations are much
larger than a single n-by-n matrix. CUDA low-rank cuts are **not** full
convex SDP solves. Read [the full SDP comparison protocol](docs/full-sdp-comparison.md).

## Native Windows CUDA low-rank Max-Cut (optional)

The original full SDP + GW/K-Means thesis experiment remains under
`maxkcut`. A separate `maxkcut-gpu` command uses **optional PyTorch CUDA**
for a *nonconvex rank-restricted vector Max-Cut optimization*, then feeds
its vectors into the same thesis-inspired GW/K-Means2N/K-MeansNM rounding
functions. This is a **different optimizer**, not a GPU port of the
full convex CVXPY/SCS SDP. It forms **no dense n-by-n Gram matrix**,
works with the original signed graph weights, and is supported only
for **k=2**. Its vector objective is **not** a certified SDP bound.

Your RTX 4500 Ada can use a precompiled Windows CUDA PyTorch wheel without
WSL, administrator privileges or the standalone `nvcc` compiler.
Keep your current Conda environment intact and install a separate GPU one:

```powershell
conda env create -f environment.yml -n maxkcut-cuda
conda activate maxkcut-cuda
python -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu132
python -m pytest -q
maxkcut-gpu --device-info
maxkcut-gpu --graphs G11 --device cuda --rank 32 --epochs 200 --optimizer-starts 2 --restarts 20 --seed 2026 --output reports/gset/g11-torch-cuda.jsonl
```

The default device is `cuda` and the program **refuses to run** if a
CUDA-enabled PyTorch installation is unavailable. `--device cpu` is
an explicit CPU testing option, not an automatic fallback.

To reuse the saved rank-r factors but try a new set of K-Means/GW
rounding seeds, pass the same optimization `--seed` and a different
`--rounding-seed`. All graph weights, vector factors, original
optimizer histories, exact cut scores, and partitions are archived.
Use `--max-n 20000` explicitly to try large canonical graphs such
as G81; `--graphs ALL` requires `--allow-all` to prevent accidental
large batches.

See [the CUDA setup, mathematics and reproducible benchmark guide](docs/cuda-lowrank.md).

## CPU threading and optional NVIDIA GPU

Your G11 SDP uses dense semidefinite-cone projections. The number of logical
processors does not imply SCS automatically uses them all. Check hardware
visibility and use a controlled BLAS thread limit:

```powershell
maxkcut --hardware-info
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-5 --max-iters 20000 --cpu-threads 8 --no-sdp-cache --restarts 20 --output reports/gset/g11-cpu8.jsonl
```

`--cpu-threads` affects recognized BLAS/LAPACK thread pools, **not** every
solver kernel, and may be slower with too many threads. A separate
`--scs-cudss` flag requests an optional **CUDA/cuDSS-enabled custom SCS build**
for compatible NVIDIA GPUs; standard Conda SCS does not guarantee this
backend. It is not an automatic GPU speedup, particularly for SDP cone
eigenprojections.

The same `--cpu-threads N` flag now also sets SDPA's native
`numThreads=N` option. This matters for Windows users who saw
`Memory Exhausted (bad_alloc)` despite 64 GB RAM: limiting native workers
may reduce resource demands but cannot guarantee a successful SDP solve.
See the [SDPA/Windows troubleshooting section](docs/windows-powershell.md).

See the [CPU/GPU performance guide](docs/gpu-cpu-performance.md) and
[Windows PowerShell guide](docs/windows-powershell.md).

## Documentation and agent instructions

```bash
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Read `docs/_build/html/index.html` for the Sphinx/Furo site, or `docs/index.md` for source. For contributors and AI agents, **start with [AGENTS.md](AGENTS.md)** and the [agent guide](docs/agent-guide.md).

- [Source-grounded research background](notes/background/README.md) — SDP mathematics, Chapter 7, prior art and open questions.
- [G-set reproduction runbook](notes/experiments/gset-cvxpy-reproduction.md) — inputs, protocol and numerical limitations.
- [Package architecture](docs/architecture.md) and [API](docs/api.md).

The `G-Set/` files are associated with Chapter 6's Max-Cut benchmark, **not** the 300 A/B/C/D random graphs used in Chapter 7. No full G-set SDP computation is claimed by this packaging refactor.

### Canonical input and published G-set comparisons

Both `maxkcut` and `maxkcut-sweep` now read the canonical `data/G*.txt`
files by default. Preserve previous experiments by selecting
`--gset-dir G-Set` explicitly for the older MATLAB inputs.
The new text reader validates the header, one-based edges, weights,
vertex identifiers and duplicate undirected edges.

To inspect graph equivalence and the published comparisons **without running
an SDP**:

```powershell
maxkcut-sweep --compare-inputs
maxkcut-sweep --compare-only
Import-Csv reports/gset/full-first-pass/comparison.csv | Where-Object {$_.paper_cut -ne ''} | Format-Table graph,our_best_cut,paper_cut,paper_upper_bound,reference_status,assessment
```

The 2026 Glass–Feder paper covers **17 toroidal instances**; only rows with
matching paper cut and certified upper bound have a proven optimal cut.
A filename match alone does not guarantee the older MAT file is canonical.
See [the new data comparison guide](docs/canonical-data-comparison.md).

## EVA vector-relaxation reproduction (experiment002)

A separate research backend now reproduces the sequential greedy
**Efficient Vector Ascent (EVA)** method (Gonzalez-Torres et al., 2026).
Unlike the full convex CVXPY/SCS method, EVA directly optimizes an
`n x p` factor and is **nonconvex**; its numerical objective is not
an SDP certificate. The initial target is published G-set graphs and
our existing full-SDP results, with independent GW, K-Means2N and
K-MeansNM roundings applied to the saved EVA factor.

The `maxkcut-eva` command writes full factor checkpoints, objective
histories, every signed weighted cut partition and verified graph
snapshots into the tracked `results/gset/experiment002-eva/` directory.
See the [EVA reproduction runbook](docs/eva-reproduction.md) and
[experiment protocol](notes/experiments/experiment002-eva-reproduction.md).
The numerical G-set reproduction has **not yet been run**; repository
code and documentation do not establish a result.


## All-graph EVA vs full-SDP benchmark

The independent experiment002 branch now supports an explicitly bounded
`maxkcut-eva --graphs ALL --max-n 20000 --continue-on-error` run
over the 71 checked-in canonical G-set graphs. Each graph archives the
complete EVA factor, all 60 rounded cuts, sweep logs, and a resumable
status; a per-graph summary CSV compares it against any existing
archive-verified full convex SCS SDP result on an identical input SHA256.

See [the all-graph EVA command and setup](docs/eva-reproduction.md)
and [experiment design](notes/experiments/experiment002-eva-reproduction.md).
No all-graph EVA numerical results have been claimed or produced in the
authoring environment.


### Extend archived EVA factors without another optimization solve

After EVA completes a graph, use `maxkcut-eva-round` to add 80 more
roundings per original method (20 + 80 = 100 each) without changing the
EVA factor, restarting its optimizer, or overwriting any partition:

```powershell
python -m maxkcut.eva_rounding_cli --graphs G77 --start-restart 20 --restarts 80 --methods FJ_EVA K-Means2N_EVA K-MeansNM_EVA --push --push-branch experiment002-eva-factor-reproduction
```

Two new optional hybrid methods are `FJ-Flip_EVA` and
`K-MeansNM-Flip_EVA`, each followed by signed-edge one-vertex
local improvement. New results include immutable factor snapshots and
all partitions under `results/gset/experiment002-eva/rounding-extensions/`.
See [the runbook](docs/eva-reproduction.md).


### Experiment005: exact one-edge MST rounding (no PCA)

For each **saved EVA factor**, form the conceptual complete graph of
full-dimensional row vectors and compute an exact MST using L1, L2 or
L-infinity distances. Remove the single longest MST edge, yielding
**two components**; score the resulting binary partition on the original
weighted G-set graph. The implementation uses Numba Prim (O(n^2 p)
time, O(n p) memory), not a dense distance matrix or PCA projection.
Each metric is deterministic: there are no meaningful MST restarts.

    python -m maxkcut.eva_mst_cli --graphs G63 --metrics l1 l2 linf --max-n 10000 --source-dir results/gset/experiment003-eva-numba --output-dir results/gset/experiment005-eva-mst

See [experiment005 protocol](notes/experiments/experiment005-eva-mst.md).


Full 71-graph mode (saved factors only, no optimizer) searches the
Numba output first, then the original NumPy output:

    python -m maxkcut.eva_mst_cli --graphs ALL --metrics l1 l2 linf --max-n 20000 --source-dir results/gset/experiment003-eva-numba --fallback-source-dir results/gset/experiment002-eva --output-dir results/gset/experiment005-eva-mst --continue-on-error --push --push-branch experiment005-eva-mst-single-linkage


### Experiment006: is the largest MST edge the right one to cut?

The longest geometric MST edge is only one option. The new exact
`maxkcut-eva-mst-center` command can reuse the same saved original-space
MST and compare: **most balanced edge** (largest tree-edge betweenness),
**weighted tree-diameter midpoint edge**, and **edge with the largest
original G-set weighted Max-Cut**. The third is score-informed rather
than a geometric center. Each rule yields one deterministic two-group
partition; no PCA, optimizer reruns, or random restarts.

    python -m maxkcut.eva_mst_center_cli --graphs G63 --metrics l1 l2 linf --rules balanced diameter_center bestcut --max-n 10000 --source-dir results/gset/experiment003-eva-numba --output-dir results/gset/experiment005-eva-mst

See [experiment006 notes](notes/experiments/experiment006-eva-mst-central-edge.md).


### Experiment009: multiple longest MST edges plus signed Max-Cut refinement

Starting from the saved exact full-dimensional EVA MSTs, remove
`q=1,2,4,8,16,32,64,128` longest edges. This creates `q+1`
connected tree fragments, which are optimized into just **two** Max-Cut
labels by contracting the original signed G-set graph and running
sparse local improvement. Then refine the resulting binary assignment
with unrestricted **vertex flips** on the original graph. The previous
best coarse result is inherited at increasing `q`, so the best
component-level cut cannot decrease. Exact trees are reused, not
recomputed; there is no PCA or second EVA optimization.

    python -m maxkcut.eva_mst_multicut_cli --graphs G63 --metrics l1 l2 linf --q-values 1 2 4 8 16 32 64 128 --coarse-restarts 20 --vertex-sweeps 20 --mst-dir results/gset/experiment005-eva-mst --output-dir results/gset/experiment009-mst-multiedge

The G63 benchmark to beat is **26,682** from the observed 1,000-restart
K-MeansNM plus local search run. No MST multi-edge result is claimed yet.
See [experiment009 protocol](notes/experiments/experiment009-mst-multiedge.md).


### Experiment010: Concurrent KL-style and Tabu local Max-Cut optimization

Experiment010 reuses the **archived, independently verified** G63
`K-MeansNM-Flip_EVA` partitions (experiment008's 1,000 rounds, best
feasible cut **26,682**) and refines *identical paired starting
partitions* with two independent signed-weight sparse heuristics:

- **Generalized KL-style** locked gain-sequence search: move each vertex
  at most once per pass and retain the best positive cumulative prefix.
- **Tabu** best-admissible signed gain search: allows temporary negative
  moves with tabu tenure, aspiration, stagnation cutoff and best-so-far
  recovery.

These solvers run in **two Windows-safe worker processes** by default,
using compiled Numba O(n+m)-memory routines. No new EVA factor, SDP
solve, MST, PCA or randomized rounding is required. Each chunk saves
all improved partitions and independently verified original weighted
Max-Cut scores to resumable archives.

```powershell
python -m maxkcut.eva_escape_cli --graphs G63 --source-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment010-eva-kl-tabu --methods KL Tabu --top-starts 200 --workers 2 --kl-passes 30 --tabu-steps 40000 --tabu-tenure 25 --tabu-stagnation 8000 --benchmark 26682
```

Use `--top-starts 1000` to run the full archive, automatically skipping
verified already-completed chunks under the same settings.
See [Experiment010 protocol](notes/experiments/experiment010-kmeans-kl-tabu.md).


### Experiment011 — Intensive elite KL search toward G63 27,047

G63's verified feasible 26,682 K-MeansNM-Flip baseline improved to
**26,886** through experiment010's KL; the new **27,047** target is
161 above this result, not a certified optimum. Experiment011 starts
from the independently verified **KL output partitions**, not the
original EVA vectors. Three independent strategies run concurrently:
**ILS-KL** (variable Hamming perturbations), **EliteMix-KL**
(complement-aligned elite recombination), and **Tabu-KL** (adaptive
Tabu bursts followed by KL). A compiled indexed max-heap reduces
KL's per-pass selection work from O(n²) scans to O((n+m)log n).

```powershell
python -m maxkcut.eva_elite_cli --graphs G63 --methods ILS-KL EliteMix-KL --top-elites 24 --replicates 5 --cycles 80 --kl-passes 12 --workers 4 --target 27047 --source-dir results/gset/experiment010-eva-kl-tabu --output-dir results/gset/experiment011-elite-search
```

No EVA rerun, PCA, MST or new rounding. All per-cycle traces and
signed-weight verified final partitions are archived with resumable
per-chunk checkpoints. See [experiment011 notes](notes/experiments/experiment011-kl-elite-search.md).


### Experiment012: improve K-MeansNM rounding before KL

Instead of applying KL only to the original archived EVA K-MeansNM-Flip
partitions, this experiment improves the **rounding geometry first**:
reproduce each seeded antipodal spherical K-Means trajectory in the
**full, original EVA ambient space**, optimize the signed Max-Cut
threshold EXACTLY along each resulting direction, and explore
additional graph-cut-scored directions before running accelerated KL.
A paired control applies KL to the original saved K-MeansNM-Flip cut
with the same restart and original experiment010 KL seed schedule.
Every restart archives all four stages and original weighted graph
score verification. It reuses saved EVA factors and rounding archives:
no PCA, no EVA optimization, no MST.

```powershell
python -m maxkcut.eva_cutaware_kmeans_cli --graphs G63 --top-starts 40 --direction-trials 48 --kl-passes 30 --workers 4 --target 27047
```

The observed prior KL best for G63 is 26,886; **27,047 is an
experimental target, not a certified optimum**. See
[Experiment012 protocol](notes/experiments/experiment012-cut-aware-kmeans-kl.md).


### Experiment013 — grid search across complete EVA rounding/Max-Cut pipelines

Instead of comparing one K-Means tweak at a time, Experiment013
systematically evaluates **43 checked-in combinations** of the
ALREADY solved EVA factor (G63 original R^119) with GW/FJ, antipodal
and ordinary spherical K-Means, objective-aware exact threshold
rounding, signed greedy vertex flips, generalized Numba KL with
independent repeat seeds, and optional iterated KL basin hopping.
Every configuration runs on identical verified experiment008
starting restart IDs, archives the **Round / Flip / KL / Final**
binary partitions, independently rescored original signed graph cuts,
settings, timing, and SHA256 identity. A parallel Windows-safe
runner supports a 40-start pilot, automated top-8 promotion to 200
starts, and top-3 promotion to all 1,000 saved restarts with partial
chunk resume. The grid is editable in
[configs/experiment013-grid.json](configs/experiment013-grid.json).

```powershell
python -m maxkcut.eva_pipeline_grid_cli --graphs G63 --top-starts 40 --workers 4 --chunk-size 10 --target 27047
```

Observed G63 benchmark remains **26,886** (Experiment010 KL);
27,047 is a target, not a certificate of optimality. See the
[full protocol and promotion commands](notes/experiments/experiment013-eva-pipeline-grid.md).
