# Revisit the seven G-set SDPs excluded in the thesis

## Research goal

Ángel Eduardo Rodríguez Fernández (2021, Chapter 6, page 56) reported
that CVX and Mathematica could not complete seven G-set Max-Cut **SDP
relaxation** instances: G55, G56, G60, G61, G70, G77 and G81.

We now have feasible cuts on all 71 canonical G-set graphs from the
different, nonconvex `maxkcut-gpu` low-rank method. That **does not
demonstrate that we solved the seven full convex SDPs**.

Our research target is a controlled, four-way comparison:

1. Original **convex SDP** solved numerically with CVXPY/SCS, then GW,
   Ángel's K-Means2N and K-MeansNM roundings, 20 restarts per method.
2. A separately labeled **nonconvex CUDA low-rank factor optimizer**,
   followed by exactly the same GW and clustering procedures.
3. Published feasible cut values / known upper bounds, with proven
   optimum labels only when feasible cut equals valid upper bound.
4. Failure modes, time and memory use for instances that cannot be
   processed by the original full SDP on present hardware.

**Do not call the approximate primal SDP objective a certified upper bound.
Do not call a rounded cut a proof of optimality.**

## First: try exact full SDP on canonical G11 (800 vertices)

Run **only the commands** within PowerShell code blocks (not headings).

```powershell
git pull --ff-only origin main
conda activate maxkcut-cuda
python -m pip install -e .
python -m pytest -q
maxkcut --list-solvers
maxkcut --graphs G11 --gset-dir data --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --cpu-threads 8 --restarts 20 --seed 2026 --max-n 1000 --output reports/gset/g11-fullsdp.jsonl
```

This solves the **full convex SDP** and applies the original three
rounding algorithms on the *same resulting full-SDP Gram factors*. SCS
may return `optimal_inaccurate`, which is **not a certified optimum**.
The code rejects grossly non-PSD matrices before rounding.

The GPU G11 strong-run output used rank=64, 500 epochs, 4 optimizer starts,
20 rounding restarts, best cut 536. Compare it:

```powershell
maxkcut-sdp-compare --sdp reports/gset/g11-fullsdp.jsonl --cuda reports/gset/g11-cuda-strong.jsonl --output reports/gset/g11-sdp-vs-cuda.csv
Import-Csv reports/gset/g11-sdp-vs-cuda.csv | Format-Table graph,graph_identity,sdp_best_cut,cuda_best_cut,sdp_minus_cuda_cut,paper_cut,sdp_status
```

The comparison checks **actual independently archived normalized signed
edges**, not merely graph IDs or raw-text-versus-MAT source extensions.
It reports the three methods' separate best scores, best-method name and
mean, solver status, original (not cache-hit) optimization time and ratio,
and literature table 1 references. It explicitly refuses numerical
cut deltas when archives are unavailable or graphs do not match the
canonical dataset. Review `graph_identity` before interpreting gaps.
The seven excluded graphs have **not** been solved by this command.

For an initial follow-up, G12 and G13 are also 800 vertices:

```powershell
maxkcut --graphs G12 G13 --gset-dir data --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --cpu-threads 8 --restarts 20 --seed 2026 --max-n 1000 --output reports/gset/g12-g13-fullsdp.jsonl
maxkcut-sdp-compare --sdp reports/gset/g11-fullsdp.jsonl reports/gset/g12-g13-fullsdp.jsonl --cuda reports/gset/all71-cuda-baseline.jsonl --output reports/gset/sdp-vs-all71-cuda.csv
```

In the 71-row report, graphs with no new successful SDP solve remain
`NOT_RUN` rather than being assigned invented results. Results of
different CUDA budgets should be analyzed in separate comparisons.

## Inspect the seven excluded SDPs before attempting them

```powershell
maxkcut-sdp-compare --preflight-seven
```

This only reads the input header and reports n, edges, and **the memory
for one dense float64 Gram matrix**. It does not solve an SDP or predict
true peak RAM. For n=20,000, a single X is
20,000² × 8 bytes ≈ **2.98 GiB**, but CVXPY canonicalization, SCS's
semidefinite cone, factorization, dense eigendecompositions, temporary
copies, and the solver's linear system may require **many times more**.
The cost of full eigendecomposition scales cubically with n. A 64 GiB
machine can exhaust memory even though the raw X appears to fit.

The `maxkcut` default safety cap `--max-n 1000` remains unchanged.
An explicit n>1000 experiment is a significant resource allocation; do
not launch the seven as a batch or set `--max-n 20000` casually.

If you decide to attempt a carefully bounded bigger convex instance,
choose a single graph and use SCS indirect KKT as an alternative to test
for memory behavior. **Indirect SCS does NOT eliminate the PSD cone
eigenprojections and may still be computationally prohibitive.**

For example, after inspecting the preflight and confirming resources,
an experimental 2000-vertex convex SDP G32 command would be:

```powershell
maxkcut --graphs G32 --gset-dir data --k 2 --solver SCS --eps 1e-3 --max-iters 500 --scs-indirect --cpu-threads 8 --restarts 5 --seed 2026 --max-n 2000 --output reports/gset/g32-fullsdp-indirect-pilot.jsonl
```

This is a budgeted numerical **pilot**, not a guaranteed convergent SDP.
If its solver status is `optimal_inaccurate` or its repaired Gram matrix
fails feasibility, do not report it as a solved/optimal relaxation.

## What the comparison can establish

- Same graph identity: independent normalized signed graph snapshots
  equal canonical `data/Gxx.txt`.
- **Clustering benefit:** compare GW vs K-Means2N/K-MeansNM
  **on the same exact convex SDP factor**, and separately on the
  CUDA low-rank factor. K-MeansNM can preserve its paired initial
  GW cut, but signed weights do not inherit every nonnegative-case
  approximation theorem.
- **Different optimizers:** compare full convex SDP versus CUDA nonconvex
  low-rank factor rounding, keeping seeds, independent optimizer starts
  and rounding restarts clearly reported.
- **Literature:** only the 17 toroidal graphs in Glass–Feder Table 1
  have references in our built-in table. Missing references for other
  graphs must stay missing.
- Runtime: original solver/optimizer duration is reported separately
  from cache use; total wall time includes roundings and preparation.
  A fast GPU value does not prove that the full SDP was solved.

### Important resource limitation

The current `maxkcut` implementation is a **dense CVXPY full-PSD
variable**. It is appropriate for initial 800-vertex tests and maybe
selected medium instances, but a comprehensive comparison for the seven
large excluded cases will likely require a **different scalable numerical
SDP solver** that still optimizes the entire convex relaxation, and
independent verification of its primal/dual residuals. A low-rank
Burer–Monteiro nonconvex solver alone cannot demonstrate full SDP
optimality. See [CUDA low-rank distinction](cuda-lowrank.md).
