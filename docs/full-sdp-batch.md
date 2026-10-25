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
