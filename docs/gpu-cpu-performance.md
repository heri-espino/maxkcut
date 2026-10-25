# Using CPU threads and an NVIDIA GPU for G11

The existing G11 run finished the **loose-tolerance** SCS SDP and produced
20-restart best cuts GW=516, K-Means2N=546 and K-MeansNM=546.
That run used a slightly indefinite numerical Gram matrix repaired **only for
rounding**, not as a certified SDP upper bound. A subsequent stricter SCS solve
(`--eps 1e-5 --max-iters 20000 --no-sdp-cache`) was reported slow on a Windows
system with 32 logical processors and a GPU.

**Critical distinction:** neither 10% CPU utilization nor the presence of a
GPU proves a bad implementation. Most of the runtime may occur inside the
linear system solver or PSD cone eigenprojections, which have their own
threading and GPU support. CVXPY is only the modeling interface; do not claim
that changing `numpy` arrays to CuPy automatically moves SCS's SDP onto GPU.

## 1. Inspect the machine (no optimization)

From **Windows PowerShell** with Conda activated:

```powershell
git pull origin main
conda env update -f environment.yml --prune
conda activate maxkcut
python -m pytest -q
maxkcut --hardware-info
maxkcut --list-solvers
nvidia-smi
```

The `--hardware-info` report lists logical CPUs, detected BLAS thread
pools and any NVIDIA GPU reported by `nvidia-smi`. **It does not mean the
installed SCS binary includes GPU support**. If `nvidia-smi` is not present,
the GPU might be AMD, Intel, or not configured for CUDA; do not guess from the
user's 32 logical CPUs.

## 2. Control CPU BLAS threads

Use a new output filename for each run:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-5 --max-iters 20000 --no-sdp-cache --cpu-threads 8 --restarts 20 --output reports/gset/g11-cpu8.jsonl
```

This limits thread pools **recognized by threadpoolctl** around the SDP call and
the NumPy symmetric eigendecomposition. **For `--solver SDPA` only**, the
same `--cpu-threads N` value is additionally passed to SDPA's native
`numThreads=N` option. CVXPY documents that SDPA otherwise defaults to
`multiprocessing.cpu_count()`, which is 32 on this workstation. It may speed up Eigen/LAPACK operations
if the linked BLAS supports multithreading, but **it cannot force the entire
SCS implementation to use eight CPU cores**. Some SCS kernels and
configurations can remain single-threaded. Larger thread counts may be *slower*
because of memory bandwidth, kernel overhead and oversubscription.

For fair solver *timing* comparisons, use identical `--eps`, iterations and
graph settings and separate output files. 4, 8 and 16 threads are useful
experiments. Do not launch many large SDP solves concurrently: the same RAM and
memory bandwidth would be contested.

## 3. Compare Clarabel when CPU SCS is slow

Your `--list-solvers` output already includes `CLARABEL` and `SCS`.
Try a separate G11 experiment if you have sufficient RAM:

```powershell
maxkcut --graphs G11 --k 2 --solver CLARABEL --no-sdp-cache --cpu-threads 8 --restarts 20 --output reports/gset/g11-clarabel.jsonl
```

**Important:** the CLI's `--eps` and `--max-iters` flags configure SCS
only; Clarabel currently uses its own default tolerances. Thus this command is
an exploration of feasibility/memory, **not a matched-accuracy benchmark**.
Interior-point solvers can converge more accurately but also exhaust memory
much earlier.

## 4. GPU: SCS with cuDSS (advanced, opt-in)

CVXPY documents the SCS keyword `cudss=True`. SCS must be **compiled and
linked against NVIDIA cuDSS** and CUDA; a regular conda-forge SCS package need
not support that feature. The new CLI has an explicit flag:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --scs-cudss --eps 1e-3 --max-iters 3000 --no-sdp-cache --restarts 10 --output reports/gset/g11-cudss.jsonl
```

**Do not run that command unless you have installed a cuDSS-enabled SCS**.
The code will fail clearly if this optional backend is unavailable. No
automatic CPU fallback is used, because that would silently falsify GPU
timings. The `--scs-cudss` and `--scs-indirect` flags are incompatible.

A cuDSS-backed SCS uses a GPU for parts of its **sparse linear system solve**.
It does not automatically move every part of the semidefinite cone projection
to the GPU. The SCS project's SDP benchmarks report that GPU sparse-solver
acceleration **may not help SDP runtimes** when cone eigenprojections dominate.

CUDA/cuDSS installation and GPU-enabled SCS building are outside the default
cross-platform Conda environment. The exact steps depend on the GPU's vendor,
VRAM, driver, CUDA version and host platform. The SCS documentation gives
specific instructions for building the cuDSS backend; **a native Windows
Conda package supporting the necessary interface is not established here**.
A separate WSL2/Linux Conda environment can be considered if the native
Windows build is unavailable.

## 5. What *not* to infer

- A high overall CPU percentage is **not** itself a performance goal; elapsed
  time, residuals and quality of the SDP solution are what matter.
- GPU acceleration does not imply a faster large cone projection.
- Using a GPU-enabled but low-rank **Burer-Monteiro** optimizer is a **different
  nonconvex algorithm**, not a reproduction of the full convex SDP. If explored,
  put it in its own experiment and do not label its output an SDP optimum.
- An `optimal_inaccurate` SCS status or a small clipped Gram eigenvalue
  is *not* a certified SDP optimum or upper bound.
- A cached SDP's original solve time is **not** the elapsed time of a cache-hit
  run. Use `--no-sdp-cache` when comparing compute backends.

## Primary references

- [CVXPY solver options](https://www.cvxpy.org/tutorial/solvers/) — the SCS
  `cudss` and `use_indirect` options and solver backend contracts.
- [SCS BLAS/LAPACK](https://www.cvxgrp.org/scs/blas_lapack/index.html) —
  eigen-decomposition for semidefinite cone projections.
- [SCS Python installation](https://www.cvxgrp.org/scs/install/python.html) —
  cuDSS-enabled GPU SCS builds and NVIDIA prerequisites.
- [SCS SDP benchmarks](https://www.cvxgrp.org/scs/benchmarks/index.html) —
  GPU acceleration can be limited by PSD cone eigenprojections.

## SDPA memory allocation errors, even with 64 GB RAM

A native Windows G11 run of `sdpa-python 0.2.3` produced
`Memory Exhausted (bad_alloc) :: line 776 in sdpa_struct.cpp`.
The error indicates **a failed native memory allocation**; it does not prove
that 64 GB of physical memory was all used or that 64 GB is intrinsically too
small for G11. It could be SDPA's internal formulation, allocation request,
Windows process memory/commit limit, or a backend-specific issue.

Our previously exposed `--cpu-threads N` only restricted BLAS. It now also
passes SDPA's documented `numThreads=N`, but **thread reduction does not
guarantee memory success**. The SDPA Python wheel uses a prebuilt OpenBLAS
backend, separate from NumPy's detected MKL library.

In **PowerShell**, check actual available memory and the Windows system-wide
commit charge and limit (before or during an attempt):

```powershell
Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' | Select-Object -ExpandProperty CounterSamples | Select-Object Path,@{Name='GiB';Expression={[math]::Round($_.CookedValue/1GB,1)}}
```

This is a non-invasive read-only measurement. The commit limit depends on
physical RAM and page files; low committed memory makes a system-wide
commit-limit explanation less likely, but still cannot exclude a failed
solver-specific allocation.

If you decide to try the native SDPA backend **once** with fewer threads,
use a new results file. No SDP is run automatically:

```powershell
maxkcut --graphs G11 --k 2 --solver SDPA --cpu-threads 4 --no-sdp-cache --verbose --restarts 20 --output reports/gset/g11-sdpa-4threads.jsonl
```

If the same `bad_alloc` occurs, **do not repeat it in a loop** or conclude
more RAM alone is the remedy. Prefer the already working SCS path, examine
SDPA's canonicalization, and consider a genuinely more memory-efficient
SDP formulation only as a separate research project. Keep the machine
responsive and close other memory-intensive processes for experiments.

Official sources: [CVXPY SDPA options](https://www.cvxpy.org/tutorial/solvers/),
[SDPA Python options](https://sdpa-python.github.io/docs/usage/params.html),
[Microsoft system commit limit and page files](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows).
