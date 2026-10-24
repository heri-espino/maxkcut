# Efficient and reproducible G-set solves

The library is designed for research reproducibility; it is **not** guaranteed to be the fastest solver for all graphs. These are tuning mechanisms, not benchmark outcomes.

## Where the time goes

The G11 Max-Cut SDP creates a symmetric positive-semidefinite matrix of size **800 × 800**. At double precision this dense matrix alone occupies about **4.9 MiB**, but the semidefinite cone, canonicalized sparse constraints, eigenprojections, linear solver and factorization can require much more memory. The graph adjacency is sparse and comparatively cheap.

Stages:

1. `load_graph`: read MATLAB v5 data with SciPy; validate symmetry.
2. `solve_sdp`: CVXPY model and conic solver (normally dominates).
3. `factorize`: one dense symmetric eigendecomposition for Gram vectors, minimum-eigenvalue diagnostics and controlled repair of small numerical PSD violations.
4. `round_fj` / `round_spherical_kmeans`: fast NumPy projection and clustering.
5. `cut_weight`: exact scoring on the original sparse graph edges.

The benchmark now **skips the second eigendecomposition** that used to be performed inside `solve_sdp` when the next operation is `factorize`.

All multi-line commands on this page use **Bash** (backslash line continuation). If running in **Windows PowerShell**, use the corresponding one-line commands or a backtick as the line continuation; see the [PowerShell runbook](windows-powershell.md).

## Cache an expensive solve

```bash
# First run: performs SDP, saves fingerprinted reports/gset/sdp_cache/*.npz
maxkcut --graphs G11 --k 2 --restarts 10 --seed 2026 \
  --output reports/gset/g11-first.jsonl

# Second run: uses the same SDP, explores 100 additional independent rounding seeds
maxkcut --graphs G11 --k 2 --restarts 100 --seed 2027 \
  --output reports/gset/g11-more-rounding.jsonl
```

The cache key includes the **SHA256 of the original .mat bytes**, $k$, solver choice/version, CVXPY version, solver tolerances and SCS linear solver mode. It never silently reuses the Gram matrix for a different file/settings. Raw matrices are saved with `numpy.savez` and loaded with `allow_pickle=False`. Reads revalidate shape and finiteness; **the numerical output still requires PSD/feasibility checks**.

A cached solution does *not* imply an exact solution, certified dual upper bound, or convergence. Small negative eigenvalues from an approximate SDP may be clipped for **rounding only**. The original defect and projection metrics are logged; `--psd-repair-rel-tol 0` requests a strict rejection check. See [PowerShell troubleshooting](windows-powershell.md) for the G11 example. Inspect `solver_status`, `min_eigenvalue` and `max_diagonal_error`.

Opt out:

```bash
maxkcut --graphs G11 --k 2 --no-sdp-cache \
  --output reports/gset/g11-fresh-solve.jsonl
```

**Use a new JSONL output path on each invocation**: results files append, so rerunning with the same file would duplicate samples.

## Compare two SCS linear system strategies

The default continues to use SCS's **direct** linear solver. Its **indirect** conjugate-gradient mode can reduce KKT factorization demands on certain large sparse problems, but is not automatically faster for SDP matrices:

```bash
# Default direct KKT solve
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-3 \
  --max-iters 3000 --restarts 10 --seed 2026 \
  --output reports/gset/g11-direct.jsonl

# Sparse indirect KKT solve: distinct cache identity
maxkcut --graphs G11 --k 2 --solver SCS --scs-indirect \
  --eps 1e-3 --max-iters 3000 --restarts 10 --seed 2026 \
  --output reports/gset/g11-indirect.jsonl
```

For a **fair comparison**, separately report setup time, SCS iteration count, solve time, feasibility residuals, and whether a numerically feasible matrix was obtained. Both solve strategies still require SDP cone projections, including eigendecompositions. SCS documentation recommends trying the indirect system solver for large sparse problems, but the benchmark itself must determine what works best for G11.

Check your available solvers:

```bash
maxkcut --list-solvers
```

`CLARABEL`, `MOSEK` and `SDPA` can be selected **only when the corresponding backend is installed**. Optional solvers have different licensing and memory characteristics; non-SCS tuning parameters must be configured per solver. See the official [CVXPY solver matrix](https://www.cvxpy.org/tutorial/solvers/) and [SCS guidance](https://www.cvxgrp.org/scs/help/).

## Interpret the performance fields

Each per-restart JSONL record includes:

| Field | Interpretation |
| --- | --- |
| `cache_hit` | Whether the optimized Gram matrix was read from disk |
| `cache_path` | The fingerprinted `.npz` file (or null when disabled) |
| `sdp_seconds` | **This invocation's** solver time, zero for a cache hit |
| `sdp_original_solve_seconds` | Solver time recorded when the cached solution was first created |
| `sdp_prepare_seconds` | Wall time including cache lookup and writing or solver call |
| `factorization_seconds` | Gram spectral decomposition and unit-vector normalization |
| `rounding_seconds` | Wall time for one rounding restart |
| `solver_iterations` | Number of solver iterations (when exposed by the backend) |
| `solver_internal_seconds` | Solver-reported solve time, may omit Python canonicalization |
| `min_eigenvalue` | Min eigenvalue of the **original** numerical primal solution |

Do not multiply `sdp_seconds` by the number of rounding restarts; it is reported redundantly in each row so that self-contained records retain provenance.

## CPU/GPU resource controls

The command-line tool exposes `--hardware-info`, `--cpu-threads N`, and
`--scs-cudss`. The first reports detected BLAS and GPU visibility, the
second sets threadpoolctl limits around the SDP and factorization, and the
third requires a nonstandard SCS build with NVIDIA cuDSS. None guarantees
a faster SDP solve. See the [CPU/GPU Windows guide](gpu-cpu-performance.md)
before attempting a CUDA backend. Comparisons with differing CPU-thread
counts or GPU configuration have independent cache identities; use
`--no-sdp-cache` when measuring actual solver runtime.

## Libraries and algorithmic tradeoffs

- **NumPy:** fast dense BLAS operations, RNGs and spherical centroid updates; a general-purpose data-science dataframe is unnecessary for core rounding.
- **SciPy:** sparse graph matrices, MATLAB v5 format and linear algebra. A sparse adjacency **does not** make the SDP cone sparse.
- **CVXPY:** readable modeling and solver interchange. Its overhead is justified for an exact *convex formulation*, although it may become costly for large SDPs.
- **SCS:** open-source first-order conic solver. `CLARABEL` may have different convergence/time tradeoffs; `MOSEK` and `SDPA` are optional depending on installed packages and license/access.
- **scikit-learn:** not needed just to run this specialized spherical $k$-means. Ordinary Euclidean `KMeans` is **not** identical to the normalized-centroid method.
- **PyTorch / JAX / GPU:** not automatic wins. GPU benefits depend on where time is spent; matrix cone projections can dominate even with a fast linear solver.

If a full $n=2000$ SDP is infeasible, explore low-rank **Burer–Monteiro**-type vector factorizations as a **separate, nonconvex algorithm** and label their theoretical guarantees differently. Do **not** silently replace the SDP in a thesis reproduction.

## What has actually been benchmarked?

The source documentation describes setup and comparison *methods*. As of this change, **no runtime G11 timing, speedup, or success rate has been measured** by the authoring environment. Only a timed experiment on the target machine can establish whether direct/indirect or another solver is best.
