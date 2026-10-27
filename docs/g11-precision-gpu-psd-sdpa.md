# G11 precision study: CPU SCS/SDPA vs GPU PSD eigendecomposition

## Objective and hardware

Research question: can float32 or float64 on an **RTX 4080 SUPER (16GB
VRAM, 128GB host RAM)** accelerate a full PSD cone projection or a
convex SDP solver relative to traditional CPU SCS/SDPA?

The alternative VM (RTX 4500 Ada, 24GB VRAM but 64GB host RAM) may be
useful later for **larger GPU matrices**. The 4080 SUPER has greater
nominal FP32 throughput and memory bandwidth; its FP64 eigendecomposition
need not be fast.

Use canonical **G11, n=800**, before escalating to G22 (n=2000).
Neither FP32 or GPU availability changes the *mathematical problem*:

```
maximize Tr(LX)/4   subject to   diag(X)=1, X symmetric PSD.
```

Our actual backends are **not** simply one solver with a dtype switch:

| Backend | Algorithm and arithmetic | Mathematical status |
| --- | --- | --- |
| `cpu-scs64` | CVXPY/SCS double precision | Full convex SDP |
| `cpu-sdpa64` | CVXPY/SDPA double precision | Full convex SDP |
| `gpu-admm32` | PyTorch CUDA full-matrix PSD-ADMM, float32 | Same convex SDP, distinct first-order numerical algorithm |
| `gpu-admm64` | Same full PSD-ADMM, float64 | Same convex SDP, distinct numerical algorithm |
| `cpu-eigh64` | One CPU full PSD eigenprojection (double) | Kernel diagnostic, not an SDP solve |
| `gpu-eigh32` | One CUDA full PSD eigenprojection (single) | Kernel diagnostic, not an SDP solve |
| `gpu-eigh64` | One CUDA full PSD eigenprojection (double) | Kernel diagnostic, not an SDP solve |

The full-matrix ADMM implementation does **not** impose a low-rank
Burer-Monteiro factorization. It is a first-order method applied to
the original convex constraints, but numerical convergence depends
on rho, tolerance and iteration budget. Its *correctly named*
`admm_iteration_limit` outcome is never a certified optimum.
The backend is **not SCS moved to GPU**.

## Fresh 4080 SUPER VM setup (Linux Bash)

The VM has Git and Conda. Clone and set up Python 3.12 from the repo:

```bash
git clone https://github.com/heri-espino/maxkcut.git
cd maxkcut
eval "$(conda shell.bash hook)"
conda env create -f environment.yml
conda activate maxkcut
python -m pip install -e .
python -m pip install sdpa-python
# Install the CUDA-enabled torch wheel matching the VM's driver from
# https://pytorch.org/get-started/locally/
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c 'import cvxpy as cp,torch; print("solvers",cp.installed_solvers()); print("cuda",torch.cuda.is_available(),torch.version.cuda,torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
python -m pytest tests/test_precision_psd_benchmark.py -q
python -m pytest -q
maxkcut-sdp-batch --graphs G11 --dry-run
python scripts/precision_psd_benchmark.py --graph G11 --cpu-threads 8 --restarts 20 --admm-iters 120 --eigh-repeats 3
```

**Important**: CUDA package selection differs by driver and platform.
If the cu128 install isn't compatible with this VM, use the current
recommended command printed by the PyTorch selector instead; do not
silently run CPU-only torch and report GPU measurements.
The study refuses to fall back from CUDA.

If SDPA fails with a native memory/allocation exception, its isolated
subprocess is recorded as **failed** and the remaining comparisons
continue. Do not interpret failure as a measured slower runtime.

SDPA's official pip distribution is `sdpa-python` and supports Linux
CPython 3.12; the environment file intentionally keeps it optional
because SDPA is a comparative backend. All results should record the
versions detected on the actual VM.

## Optional small smoke run (before expensive G11)

```bash
python scripts/precision_psd_benchmark.py --graph G11 --backends gpu-eigh32 gpu-eigh64 --eigh-repeats 1 --timeout-seconds 600
```

This is **not** an SDP solve; it isolates symmetric PSD projection
performance. For a fair full comparison run the default all-backend
command afterwards and use those reported measurements.

## Publish results and failures to GitHub

When the default run completes, its raw case JSON and full logs live
under the gitignored `reports/gset/precision-psd-benchmark/` directory.
The command also exports sanitized, source-grounded results and
individual rounding trials to:

`results/gset/precision-psd-benchmark/`

Review the exported files and publish them from the repo root:

```bash
git add results/gset/precision-psd-benchmark
git commit -m "results: benchmark G11 CPU SCS SDPA versus GPU PSD float32 and float64"
git push origin HEAD:main
```

The source case JSON includes durations, graph SHA, original numerical
primal, minimum eigenvalue, diagonal errors, original solver status,
ADMM primal/dual residuals and convergence thresholds, torch/CUDA
versions, full iteration history and every cut restart. A separate
`comparison.csv` retains failures. The `rounding-trials.csv` exposes
all FJ, K-Means2N and K-MeansNM samples for method-level comparisons.

This experiment does **not** alter or replace earlier Windows G-set
results; the fresh study has its own raw and publication directories.

## Interpretation

- Same convex SDP mathematics does **not** imply identical floating
  results, convergence or cut partitions across different solvers.
- Use the same graph checksum, original primal objective, PSD defect,
  diagonal error, ADMM primal/dual residuals and rounded cut quality.
- A primal objective is **not** a certified upper bound.
- An SCS `optimal_inaccurate` or ADMM `iteration_limit` is not a
  verified high-accuracy solution.
- GPU eigendecomposition **kernel time** omits solver setup and
  CVXPY canonicalization; compare end-to-end GPU time to CPU
  end-to-end eigenprojection and **full solver time only to other
  full solver times**.
- RTX 4080 SUPER and RTX 4500 Ada both have consumer/workstation Ada
  FP64 throughput constraints; do not expect GPU float64 to beat CPU
  LAPACK without measurement.
- GPU float32 may improve speed and lower memory, but may worsen
  convergence, PSD eigenvalue accuracy and cut results.

No actual VM measurements are claimed by this code-only benchmark.
