# G77 GPU full convex SDP feasibility on Windows 11 RTX 4080 SUPER

## Target

The **second-largest graph** checked into `data/` by vertex count
is `data/G77.txt` with **14,000 vertices and 28,000 edges**.
Largest is `G81.txt` with 20,000 vertices / 40,000 edges.
We preserve the original `max Tr(LX)/4`, `diag(X)=1`, `X PSD`
convex relaxation. **We are not using the low-rank Burer–Monteiro
CUDA optimizer.**

**Important capacity risks:** for G77, the dense 14,000×14,000
SDP matrix alone is about 748 MiB in float32 or 1.46 GiB
in float64. A full PSD eigenprojector requires many simultaneous
matrix/eigenvector arrays and vendor-dependent workspaces. NVIDIA
WDDM reserves some of the RTX 4080 SUPER's **16 GB VRAM** for the
desktop (observed `2.7 GB used at one earlier point). Its 128 GB
host RAM is **not interchangeable with GPU VRAM**. Full rank
eigh scales roughly cubically with matrix dimension per step;
moving from G11 n=800 to G77 n=14000 is **not** a modest step.

Do not start 3000 full GPU eigenprojections blindly; the
single-step experiment below is a necessary preflight, not
proof that a multi-hour solve will finish, achieve convergence,
or survive WDDM GPU timeouts.

## Step 1: Update on Windows PowerShell and verify

```powershell
cd C:\Users\175199\maxkcut
conda activate maxkcut
git pull --ff-only origin main
python -m pip install -e .
python -m pytest -q
python -c "import torch; print('GPU', torch.cuda.get_device_name(0), 'CUDA', torch.cuda.is_available()); print('free,total bytes',torch.cuda.mem_get_info())"
git push --dry-run origin HEAD:main
```

## Step 2: One full G77 PSD-projection iteration ONLY

This uses the original full-rank 14,000×14,000 PSD
matrix. The new **`--skip-rounding`** skips a separate CPU
14k×14k eigenfactorization and twenty randomized cut trials.
We cannot infer a meaningful final Max-Cut score or a converged
SDP from this diagnostic-only run.

```powershell
python scripts/precision_psd_benchmark.py --graph G77 --backends gpu-admm32-adaptive --rho 1 --admm-iters 1 --admm-abs 1e-5 --admm-rel 1e-4 --log-every 1 --restarts 1 --skip-rounding --timeout-seconds 1200 --output-dir reports/gset/g77-gpu-full-psd-1iter --publish-dir results/gset/g77-gpu-full-psd-1iter --push
```

This probe:
- loads G77's original signed edge weights;
- allocates and performs **one** GPU eigenprojection of
  the full 14k×14k SDP matrix;
- records compute time and diagnostics, including GPU
  reported peak PyTorch allocation, graph SHA,
  partial numerical objective and exception traceback;
- **does not** report an optimal SDP, partition or cut;
- automatically attempts to publish the result or failure
  to GitHub after the isolated process finishes.

Do not infer that smaller allocated PyTorch bytes imply all
CUDA libraries' external workspace fits: cuSOLVER and other
allocations may use separate pools.

## Step 3: Only if the 1-iteration probe succeeds

We need to check elapsed time, VRAM/headroom and WDDM stability
**before** deciding whether a 14k full-rank SDP is feasible
on this particular GPU. If feasible, try a bounded 10-step
diagnostic (still **not a converged SDP**):

```powershell
python scripts/precision_psd_benchmark.py --graph G77 --backends gpu-admm32-adaptive --rho 1 --admm-iters 10 --admm-abs 1e-5 --admm-rel 1e-4 --log-every 1 --restarts 1 --skip-rounding --timeout-seconds 1800 --output-dir reports/gset/g77-gpu-full-psd-10iter --publish-dir results/gset/g77-gpu-full-psd-10iter --push
```

If either run reports CUDA OOM, device loss/reset,
timeout or Windows TDR, do **not** repeat it with higher
iterations. Use the saved logs/results to assess
a different SDP algorithm, distributed hardware or
a larger-VRAM GPU such as the 24 GB RTX 4500 Ada
(not guaranteed sufficient).

## Published evidence

- `results/gset/g77-gpu-full-psd-1iter/`
- `results/gset/g77-gpu-full-psd-10iter/`

The full SDP and low-rank CUDA implementations are
**distinct numerical methods** and must not be merged in a
single undifferentiated benchmark.
