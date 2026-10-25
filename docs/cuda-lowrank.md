# Optional Windows CUDA acceleration: low-rank Max-Cut

This is a **new, separate, nonconvex optimizer**, not a GPU switch that
makes the existing CVXPY/SCS full SDP automatically use CUDA.

We preserve the original thesis experiment exactly as its own baseline:
`maxkcut` + full CVXPY SDP + GW/FJ / K-Means2N / K-MeansNM.
The new `maxkcut-gpu` command produces rank-restricted vectors on an
NVIDIA GPU using optional PyTorch, then applies the **same established
GW/K-Means rounding functions** (rounding currently runs on CPU).

## Mathematical formulation

For **Max-Cut only (k=2)** and sparse weighted undirected graph
$G=(V,E,w)$, write $X=VV^T$ with $V\in\mathbb R^{n\times r}$, $r\ll n$,
and normalize every vertex vector $\|V_i\|=1$:

$$
\max_{\|V_i\|=1}\quad
F(V) = \frac{1}{2}\sum_{(i,j)\in E}
w_{ij}(1-\langle V_i,V_j\rangle).
$$

The negative-objective gradient (before the row normalization) is

$$
\nabla_V(-F(V))=\frac12 A V,
$$

with $A$ the original symmetric signed weighted adjacency matrix.
The optimizer uses sparse GPU matrix-by-vector-factor products followed by
Adam and normalization onto the unit spheres. Every step is $O(|E|r)$
in arithmetic and no dense $n\times n$ Gram matrix is needed.

**Limitations:** The rank constraint makes this nonconvex; iterations can
converge to different local solutions. Its objective is a **primal
rank-restricted value, NOT a valid global SDP upper bound or proof of
optimality**. Selecting different hyperparameters and GPU kernels changes
the optimization dynamics. For $k>2$ the SDP imposes additional
$X_{ij}\ge -1/(k-1)$ bounds and this implementation must **not** be used
to claim an Frieze–Jerrum SDP solve. It deliberately accepts only k=2.

The output vectors feed the existing three thesis-inspired rounders:
GW (one random hyperplane), K-Means2N (independent spherical two-centroid
initialization), K-MeansNM (paired antipodal GW start and best-cut retention).
Output method names carry the `_torchlowrank` suffix, ensuring they are
**not confused with rounders after the full SDP**.

## Your Windows hardware and installation

User-reported workstation: **NVIDIA RTX 4500 Ada 24 GB**, driver 596.86,
`nvidia-smi` lists **CUDA 13.2**, 64 GB system RAM, 32 logical CPUs.
Neither Windows administrator permissions nor WSL nor the standalone
`nvcc` compiler is required to run official precompiled PyTorch CUDA
wheels. A CUDA-enabled wheel is **not** automatically selected by
`pip install torch` on every installation, so verify the CUDA runtime.

Create a **separate Conda environment** so the existing SCS environment
continues to work and can reproduce old results. Run commands from the
repository's root directory using **Windows PowerShell**:

```powershell
git pull --ff-only origin main
conda env create -f environment.yml -n maxkcut-cuda
conda activate maxkcut-cuda
python -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
python -m pytest -q
maxkcut-gpu --device-info
```

The **specific wheel index/version** above is based on the officially
documented Windows/Linux CUDA 13.2 wheel for PyTorch 2.13.0 as of Oct 2026:
https://pytorch.org/get-started/previous-versions/
If the wheel is no longer available for your platform, choose an available
compatible Windows CUDA wheel from
https://pytorch.org/get-started/locally/ . Installing a GPU wheel may
require several GB of disk space; no network work occurs at module import.

`maxkcut-gpu --device-info` **fails clearly** when PyTorch is missing
or `torch.cuda.is_available()` is false: the command never silently
switches to CPU after CUDA was explicitly requested. Run
`maxkcut-gpu --device cpu --device-info` if you want to inspect/use an
explicit CPU-only fallback for tests.

## First G11 CUDA experiment

Use a new filename to avoid mixing it with any previous full SDP or
local-search result:

```powershell
maxkcut-gpu --graphs G11 --device cuda --rank 32 --epochs 200 --optimizer-starts 2 --restarts 20 --seed 2026 --max-n 1000 --output reports/gset/g11-torch-cuda.jsonl
```

This runs **two full nonconvex optimizer starts** on the GPU to find a good
factor, then 20 randomized roundings with each of GW, K-Means2N, K-MeansNM
using the **same factor**, with the same appropriate paired seeds as our
full SDP comparison. The 20 roundings do not require 20 GPU optimizations.

Inspect the saved cuts and diagnostics:

```powershell
Get-Content reports/gset/g11-torch-cuda.jsonl -Tail 1 | ConvertFrom-Json | Select-Object backend,device,gpu_name,method,cut_weight,rank,optimizer_seconds,original_optimizer_seconds,factor_cache_hit
```

Note that JSONL is newline-delimited JSON, not CSV; use the second
`Get-Content | ConvertFrom-Json` command for these files.

To **reuse the optimized factor without repeating GPU optimization**
for a different rounding budget/seed, retain the original `--seed`
(which controls the optimizer and its cache key), change
`--rounding-seed` or `--restarts`, and use a new JSONL filename:

```powershell
maxkcut-gpu --graphs G11 --device cuda --rank 32 --epochs 200 --optimizer-starts 2 --seed 2026 --rounding-seed 2027 --restarts 40 --output reports/gset/g11-cuda-extra-rounds.jsonl
```

The factor is loaded from cache; only the new roundings are computed.
If you change `--seed`, the optimizer produces (and caches) an independent
factor run.

Each experiment archives:
- exact canonical graph SHA256, normalized sparse signed weights;
- original optimizer settings, torch/CUDA versions and detected GPU;
- all per-start convergence history and actual vector objective;
- **rank-r float32 factors**, stored once, not a dense $n\times n$ X;
- every rounded partition and cut score, bit-packed for Max-Cut;
- factor cache-hit status, optimizer duration and GPU peak allocation.

The factor is stored under
`reports/gset/lowrank_cache/G11-k2-lowrank-<hash>.npz` by default,
and separately inside the per-run archive for auditability. The
`--no-factor-cache` flag disables cross-run factor reuse (but the
factor remains inside the archive unless `--no-artifacts` is chosen).

## Larger canonical G-set graphs

A 20,000-vertex graph at rank 32 requires only about **2.4 MiB** for the
factor matrix alone (float32), but optimizer scratch buffers, sparse
weights and PyTorch libraries use additional GPU memory.

Avoid launching all 71 large runs until we have measured G11/G32/G81.
For example, you can explicitly test G81 with a smaller GPU budget:

```powershell
maxkcut-gpu --graphs G81 --device cuda --rank 32 --epochs 100 --optimizer-starts 1 --restarts 5 --max-n 20000 --seed 2026 --output reports/gset/g81-torch-cuda.jsonl
```

The default `--max-n 1000` remains a safety guard. A full-corpus run
requires both `--graphs ALL --allow-all` **and** raising `--max-n`
to the largest graph. Neither command is scheduled automatically in CI.

## Comparing solution quality responsibly

Do **not** use its lowrank_vector_objective as an SDP-certified upper bound.
Compare the rounded **exact weighted cuts** with our original
CVXPY/SCS + GW/K-Means methods and with literature values **only on
identical canonical graphs**. For standard G11, Glass–Feder (2026),
Table 1, proves 564 optimal:
https://arxiv.org/abs/2606.28478

Report **both** optimization wall time (cache misses only) and separately
rounding time, n, number of edges, rank, epochs, optimizer starts, GPU type
and max memory allocated. Cross-machine or cross-solver speedups should
be measured, not assumed. A lowrank optimizer is a different algorithm,
so matching accuracy is not guaranteed.

## Troubleshooting

- **CUDA requested but unavailable:** inspect
  `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
  and `nvidia-smi`; the optimizer refuses CPU fallback.
- **CUDA OOM:** reduce `--rank`, `--optimizer-starts` (impacts runtime)
  or run one graph at a time. GPU peak allocated memory is included in
  optimizer metadata.
- **Slow rounding despite fast GPU optimization:** the thesis rounding
  functions still use NumPy/CPU. Profile them separately; a future
  CUDA rounding implementation must be validated against their semantics.
- **Old results aren't reused:** factor caching keys the exact input
  file, Torch/CUDA backend, hyperparameters, rank and optimizer seed. A
  full-SDP Gram cache is intentionally incompatible.
- **No torch in ordinary environment:** normal package imports, CVXPY
  benchmark and unit tests still work. Only using this optional backend
  imports PyTorch.

## References

- Official PyTorch Windows/CUDA wheel selection:
  https://pytorch.org/get-started/locally/
- Official PyTorch versioned CUDA wheel commands:
  https://pytorch.org/get-started/previous-versions/
- PyTorch CUDA availability:
  https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available
- Existing architecture and performance limitations:
  [CPU/GPU guide](gpu-cpu-performance.md)
