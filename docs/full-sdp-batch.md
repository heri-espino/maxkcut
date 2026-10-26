# Full convex SDP batch across canonical G-set — bounded and resumable

## Research question

The user verified on canonical G11 that **full convex SDP/SCS + Ángel's
clustering** gave a best rounded cut of **546**, with CVXPY solver
status `optimal`, while the **different CUDA nonconvex low-rank**
experiment had best **536**; the proven G11 optimum is **564**.
`graph_identity=VERIFIED_CANONICAL` confirmed the same weighted graph.

Next, assess as many of the other 70 canonical G-set instances as
practically possible **with the actual convex SDP**, not the CUDA low-rank
replacement, and compare their GW, K-Means2N, K-MeansNM cuts against the
saved CUDA results and published references. Ángel Rodríguez Fernández
(2021, Chapter 6) listed **G55 G56 G60 G61 G70 G77 G81** as instances
his SDP software could not solve. We must keep separate:
**a completed convex SDP**, **a nonconvex feasible low-rank cut**, and
**a published/proven optimum**.

## Why we do not run all 71 in one CVXPY process

`src/maxkcut/sdp.py` uses an n-by-n PSD matrix variable. CVXPY
canonicalization, SCS cone projections and dense eigendecompositions
typically require far more memory and time than that one matrix.

| n | Memory for just ONE float64 n-by-n matrix |
| ---: | ---: |
| 800 | 0.0048 GiB |
| 2,000 | 0.0298 GiB |
| 3,000 | 0.0671 GiB |
| 5,000 | 0.1863 GiB |
| 10,000 | 0.7451 GiB |
| 20,000 | 2.9802 GiB |

**These are severe underestimates of solver peak RAM.** A 64 GB Windows
workstation can still fail or become unresponsive on large SDPs.
A wall-clock timeout is *not* an OS memory cap. Never interpret
attempts skipped by an explicit resource cap as solved SDP instances.

## Bounded all-graph batch (Windows PowerShell)

Update your existing `maxkcut-cuda` environment (the unchanged full
SDP still runs on **CPU**, not CUDA).

```powershell
git pull --ff-only origin main
conda activate maxkcut-cuda
python -m pip install -e .
python -m pytest -q
maxkcut-sdp-batch --dry-run
```

The default dry run prints all 71 canonical graph sizes and flags those
above 1,000 vertices. No optimization, output directory, or cached SDP
matrix is created.

**Stage 1.** Try every eligible graph with n <= 1000, one isolated
convex SDP subprocess per graph. Each has a 10-minute wall limit, with
a one-hour total per invocation. Failures and timeouts are recorded
rather than stopping the other graphs:

```powershell
maxkcut-sdp-batch --max-n 1000 --seconds-per-graph 600 --total-seconds 3600 --cpu-threads 8 --restarts 20 --output-dir reports/gset/full-sdp-batch
```

Re-run the **same command** to continue after the total time budget.
Completed archives are independently rescored and SHA-verified before
being skipped; timeouts/failures do not automatically re-run. Use
`--retry-failed` only after reading their per-graph solver logs.
Incremental calls under the same output directory must preserve the
solver settings. Increasing just the n cutoff is allowed when supplied
with the appropriate opt-in, but changing solver/timeout/threads/seed
requires a fresh output directory for reproducibility.

The exact child process is `python -m maxkcut --graphs Gxx --k 2
--solver SCS ...`, calling the **existing full convex PSD CVXPY**
implementation. The same SDP Gram factor is used to round
GW/FJ, K-Means2N, and K-MeansNM, default 20 restarts per method.
No GPU low-rank shortcut is used.

Outputs:

- `reports/gset/full-sdp-batch/summary.csv`: status for each graph,
  `complete`, `failed`, `timeout` or `skipped_size`, best feasible
  cut/method, SDP solver status and wall time.
- `.../cases/Gxx.json`: resumable per-graph record, settings, input SHA.
- `.../runs/Gxx/attempt-001/Gxx.jsonl`: all three rounding distributions
  and SDP residual/timing diagnostics for that exact graph.
- `.../runs/Gxx/attempt-001/artifacts/`: graph snapshot and all
  partition archives; unchanged solver's raw Gram cache remains separate.
- `.../runs/Gxx/attempt-001/solver.log`: numerical solver logs/errors.
- `.../sdp-vs-cuda.csv`: automatic exact-graph-verified quality comparison
  with the saved `reports/gset/all71-cuda-baseline.jsonl`, when present.
  Graphs without successful full SDPs are labeled `NOT_RUN`; they have
  no fabricated cuts, certified SDP bounds, or speedups.

Inspect the new numerical results:

```powershell
Import-Csv reports/gset/full-sdp-batch/summary.csv | Group-Object status | Format-Table Name,Count
Import-Csv reports/gset/full-sdp-batch/summary.csv | Where-Object {$_.status -eq 'complete'} | Format-Table graph,vertices,best_cut,best_method,sdp_solver_status,wall_seconds
Import-Csv reports/gset/full-sdp-batch/sdp-vs-cuda.csv | Where-Object {$_.sdp_best_cut -ne ''} | Format-Table graph,graph_identity,sdp_best_cut,cuda_best_cut,sdp_minus_cuda_cut,paper_cut,sdp_status
```

## Stage 2: medium instances, explicit opt-in

Only after Stage 1's resource profile, the next safe experimental class
is **n <= 2000**, in a separate result directory because the child
timeout is increased. This may still be expensive and may time out:

```powershell
maxkcut-sdp-batch --dry-run --max-n 2000
maxkcut-sdp-batch --max-n 2000 --allow-large --seconds-per-graph 1200 --total-seconds 3600 --cpu-threads 8 --restarts 20 --output-dir reports/gset/full-sdp-2000
```

This repeats small cases, but produces separately budgeted reports and
preserves Stage 1. To focus medium cases only, `--graphs G32 G33 G34`
can select an explicit small subset before other instances.

## Stage 3: the seven originally excluded graphs

Their raw graph sizes are available with:

```powershell
maxkcut-sdp-compare --preflight-seven
```

The batch runner refuses n > 3000 unless the operator provides both
`--allow-large` and `--acknowledge-huge-risk`, and sets an explicit
`--max-n` large enough to include the instance. **This is a warning,
not a claim that the computer can safely solve the SDP or a guarantee
that the Windows system cannot run out of memory**. We do not recommend
attempting G81's full 20,000-vertex CVXPY SDP in this 64 GB environment.

Reproducing Ángel's excluded **convex** SDPs may require a genuinely
large-scale SDP solver with primal/dual residuals and trustworthy bound
verification, rather than the current n-by-n CVXPY variable. CUDA
low-rank cuts on those graphs remain valuable comparisons but not
convex SDP solutions.

## Interpretation

A `complete` case means the CVXPY solver returned `optimal` or
`optimal_inaccurate` and the program produced numerically acceptable
rounding factors and verifiable feasible partitions. It **does not
prove an integer Max-Cut optimum**. A solver status `optimal_inaccurate`
must be reported as such. The value `sdp_objective_primal` is **not
automatically a certified upper bound**.

The `sdp-vs-cuda.csv` compares the best achieved **exact weighted
cut** per method after checking every archived graph's weighted edges.
Elapsed solver time excludes cached optimizer runs; review whether
`cache_hit` or `factor_cache_hit` was true before claiming an
algorithmic speedup. The 71-graph CUDA baseline used rank 32, 100
epochs, 1 optimizer start, 5 roundings per method; the full-SDP
default has 20 roundings. **Those are deliberately different compute
budgets**, so do not use differences in best cuts as a controlled
method superiority claim without matched budgets and uncertainty
analysis.

## 2026-10-25: completed Stage 1 and append-only CUDA comparison repair

The user ran the default n<=1000 full convex SCS SDP batch
over the entire canonical 71-graph list. **Thirty of thirty attempted
SDPs completed successfully**, 41 larger graphs were size-skipped,
0 failed and 0 timed out, with total invocation elapsed time
**952.58 seconds**. This was a real solver run on the user's
Windows machine. The archived per-graph best rounded cuts remain
in `reports/gset/full-sdp-batch/summary.csv`.

After this numerical success, the optional CUDA CSV comparison raised
`ValueError: ... all71-cuda-baseline.jsonl: multiple runs for G1`.
The CUDA JSONL is **append-only** and contains multiple distinct
archived runs for at least G1. The error occurred *after* the full
SDP batch was saved. **No SDP needs recomputing to recover the CSV.**

The comparison now chooses the **latest complete, independently
verified** archived CUDA run for each graph when its JSONL has multiple
runs. The choice is based on append order and archive validity, **not
cut score** (no silent cherry-picking), and never merges rounding
restarts between different runs. Each resulting CSV row records
`cuda_run_id`, `cuda_candidate_runs` and
`cuda_selection_policy` so operators can audit the policy.
The original strict one-run-per-graph SDP requirement is unchanged.

After updating the local checkout, run the **comparison-only** command
in PowerShell. It **never launches SDP or CUDA optimization**:

```powershell
git pull --ff-only origin main
python -m pip install -e .
python -m pytest -q
maxkcut-sdp-batch --compare-only --output-dir reports/gset/full-sdp-batch --cuda-results reports/gset/all71-cuda-baseline.jsonl
Import-Csv reports/gset/full-sdp-batch/summary.csv | Group-Object status | Format-Table Name,Count
Import-Csv reports/gset/full-sdp-batch/sdp-vs-cuda.csv | Where-Object {$_.sdp_best_cut -ne ''} | Format-Table graph,graph_identity,sdp_best_cut,cuda_best_cut,sdp_minus_cuda_cut,paper_cut,sdp_status
```

The comparison remains a preliminary descriptive contrast: full SDP
used **20 rounding restarts per method** and the CUDA baseline used
**5 restarts**, with a different relaxation and optimization budget.
A proper scientific comparison must match rounding budgets and verify
same graph identity, report per-method distributions, solver convergence
and timings, and distinguish proven optima from best-known cuts.

For a single n=2000 pilot *after reviewing the summary*:

```powershell
maxkcut-sdp-batch --graphs G32 --max-n 2000 --allow-large --seconds-per-graph 1200 --total-seconds 1300 --cpu-threads 8 --restarts 20 --output-dir reports/gset/full-sdp-g32-pilot
```

This is still the original **CPU convex SDP**. It may time out or
consume more RAM than the raw Gram matrix. It is not a CUDA solve.

## Continuing the original completed SDP batch with NO wall-clock limit

The first real Windows run successfully solved **30 canonical convex SDPs**
(G1–G21, G43–G47, G51–G54; n=800–1000), with **41 graphs
skipped by n<=1000**, in 952.579 seconds. All saved archives and
`summary.csv` are preserved in
`reports/gset/full-sdp-batch/`. The optional CUDA comparison is
already working, and all 30 matched graph identities.

The operator requested letting remaining graph computations run
**without any artificial wall-clock deadline**. The batch runner now
supports `--no-wall-limit`; it removes both per-graph and total
subprocess timeouts. It **does not** remove the solver's own
`--max-iters` stopping criterion (default 3000), the max-n safety
guard, available RAM restrictions, or the possibility of numerical
failure.

The same output directory can now resume when only the wall-time
policy or maximum accepted n changes; the old solver configuration,
rounding restarts, seed, thread count, input directory and archived
numerical evidence are unchanged. Existing complete cases are checked
against their saved graph SHA and independently rescored partitions.
Both the old and new execution policies are stored in
`plan.json`; each new graph case records its own `runtime_policy`.

**Stage 2: automatically skip the 30 completed graphs, and attempt
only newly eligible 2000-vertex cases with no timeouts:**

```powershell
git pull --ff-only origin main
conda activate maxkcut-cuda
python -m pip install -e .
python -m pytest -q
maxkcut-sdp-batch --max-n 2000 --allow-large --no-wall-limit --cpu-threads 8 --restarts 20 --output-dir reports/gset/full-sdp-batch
```

**Stage 3** after checking the medium-sized logs, advance to n<=3000
with the SAME output directory:

```powershell
maxkcut-sdp-batch --max-n 3000 --allow-large --no-wall-limit --cpu-threads 8 --restarts 20 --output-dir reports/gset/full-sdp-batch
```

**Do not blindly advance to n=20000 on a 64 GB RAM machine.**
Allowing a solver to run indefinitely cannot prevent memory exhaustion
or numerical failure. If attempting a single larger graph deliberately,
both `--allow-large` and `--acknowledge-huge-risk` are required
for n>3000, and the batch still processes one graph per subprocess.
For example, a 5000-vertex G55 full convex SDP is an explicit,
high-risk trial, **not** a guaranteed result; large PSD cone
projections may need many times the 0.186 GiB single-Gram floor:

```powershell
maxkcut-sdp-batch --graphs G55 --max-n 5000 --allow-large --acknowledge-huge-risk --no-wall-limit --cpu-threads 8 --restarts 20 --output-dir reports/gset/full-sdp-batch
```

If any run returns `optimal_inaccurate`, report the status and
feasibility diagnostics honestly; numerical SDP status does **not**
prove integer Max-Cut optimality. The `sdp-vs-cuda.csv` is generated
from saved partitions after each normal batch completion.

For stopping a no-wall batch, use **Ctrl+C** in PowerShell. Already
completed per-graph archives remain stored; an interrupted in-progress
graph can be attempted again on a later invocation. The operation is
synchronous, requiring the local Python/PowerShell process to stay open.

## See batch and solver progress without disturbing the running job

Your existing n=5,000+ convex SDP batch launches SCS without verbose
logging. **You cannot know a precise solve percentage** for G56 or
another already-running graph. CPU-time increases only demonstrate
computation, not SDP convergence.

In **another PowerShell terminal**, from the `maxkcut` repository,
inspect case progress without launching any optimizer:

```powershell
maxkcut-sdp-batch --status --output-dir reports/gset/full-sdp-batch
```

Optionally print updated statuses every 15 seconds; Ctrl+C terminates
the **status monitor**, not the separately running solver:

```powershell
maxkcut-sdp-batch --watch-progress --output-dir reports/gset/full-sdp-batch
```

For **future graphs that have not yet started**, opt in to SCS verbose
per-iteration numerical diagnostics:

```powershell
maxkcut-sdp-batch --graphs ALL --max-n 20000 --allow-large --acknowledge-huge-risk --no-wall-limit --cpu-threads 8 --restarts 20 --solver-progress --output-dir reports/gset/full-sdp-batch
```

This changes **logging only** (no SDP numerical settings) and preserves
completed graphs in the same output directory. Logs are stored in
`runs/Gxx/attempt-NNN/solver.log`. Use a second terminal to follow
the newest G56 log:

```powershell
Get-ChildItem .\reports\gset\full-sdp-batch\runs\G56\attempt-*\solver.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 25 -Wait
```

Important limitations:

- `--solver-progress` is **not retroactive**. Do not stop a healthy
  ongoing solver merely to add verbose logging.
- SCS normally reports numerical iteration index, primal residual,
  dual residual and duality gap every few hundred iterations. These
  metrics help diagnose convergence; **iteration/max-iters is NOT
  percentage completion**, and `max-iters` remains 3000 unless changed
  explicitly (changing it requires a fresh experiment output directory).
- Some native solver prints can be buffered. Unchanged log modification
  time **does not prove that the solver has stopped**.
- The `--status` command reads saved case JSON. A `started` record
  may be left by a hard crash and does not prove a live process.
  Use Task Manager or `Get-Process python*` for liveness.
- This adds **no numerical checkpointing** of a partial SCS solve:
  restart of the current graph usually redoes its SDP iterations.
- Progress remains on **original CPU convex CVXPY/SCS SDP**, never CUDA
  low-rank substitution; no automatic attempt to override memory limits.
