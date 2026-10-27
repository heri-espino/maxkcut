# G11 GPU full-SDP ADMM: fixed vs adaptive penalty rho

**Research experiment, 2026-10-27.** This targets **the same convex
Goemans–Williamson Max-Cut SDP** for k=2, not a nonconvex
Burer–Monteiro rank restriction, simulated bifurcation or GPU-SCS port.

The RTX 4080 SUPER Windows 11 VM previously measured:
- CPU SCS `eps=1e-3`: 41.25 s, primal 627.4429,
  cut 546, solver reports `optimal`.
- CPU SCS `eps=1e-5`: 2136.03 s, primal 629.0750,
  cut 534, solver reports `optimal_inaccurate` (10,000 iterations).
- GPU full-matrix fixed-rho ADMM float32, 3000 iterations,
  eps_abs=1e-5, eps_rel=1e-4: 20.45 s,
  primal 628.1184, cut 546, **did not satisfy dual tolerance**.
- GPU full-matrix fixed-rho ADMM float64, same target:
  66.87 s, primal 628.1184, cut 546, **did not converge**.

Those are existing GitHub results, not values to overwrite.
They motivated experimenting with **residual-balanced rho adaptation**.

## Numerical algorithm

Using scaled ADMM, split `X=Z` in

```
maximize Tr(LX)/4 subject to diag(X)=1, X symmetric PSD
```

The full n×n matrices are updated by:

```
X_(t+1) = Z_t - U_t + (L/4)/rho_t;
diag(X_(t+1)) = 1
Z_(t+1) = Proj_PSD(X_(t+1)+U_t)
U_(t+1) = U_t + X_(t+1) - Z_(t+1).
```

Primal residual `r = ||X-Z||_F`, dual residual
`s = rho ||Z-Z_old||_F`, and scalar penalty `rho>0`.

Every `rho-update-interval` steps (default 25),
after evaluating convergence for the current iteration:

```
if r > mu*s:       rho_new = min(scale*rho, rho_max)
else if s > mu*r:  rho_new = max(rho/scale, rho_min)
else:             rho_new = rho
```

Default `mu=10`, `scale=2`, `rho_min=0.0625`,
`rho_max=16`. **Critically**, if `rho` changes,
rescale the *scaled* dual `U <- (rho_old/rho_new) U`,
so the unscaled dual `Y=rho*U` is unchanged.
The logged residual and tolerance use **rho for the
current, already completed iteration**.

The fixed-rho backend is completely unchanged by default:
`adaptive_rho=False`; no rho updates occur and
the original initial rho is retained.

Updates can help or hurt convergence; this is an **empirical
hypothesis, not a pre-established speedup**.

## Windows 11 PowerShell: same-conditions four-way comparison

Stay in the **existing** `(maxkcut)` Conda environment
on the user's RTX 4080 SUPER VM:

```powershell
cd C:\Users\175199\maxkcut
conda activate maxkcut
git pull --ff-only origin main
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q
git push --dry-run origin HEAD:main
```

Then run **four separate solver processes**, all on canonical G11
with identical numerical tolerances, iterations, rho start
and seed:

```powershell
python scripts/precision_psd_benchmark.py --graph G11 --backends gpu-admm32 gpu-admm32-adaptive gpu-admm64 gpu-admm64-adaptive --rho 1 --admm-iters 3000 --admm-abs 1e-5 --admm-rel 1e-4 --rho-update-interval 25 --rho-balance-mu 10 --rho-scale 2 --rho-min 0.0625 --rho-max 16 --cpu-threads 8 --restarts 20 --timeout-seconds 1800 --output-dir reports/gset/g11-adaptive-rho-strict --publish-dir results/gset/g11-adaptive-rho-strict --push
```

- **gpu-admm32** — fixed rho=1, float32; exact reproduction
  target for the prior strict 3000-iteration run.
- **gpu-admm32-adaptive** — adaptive rho starting from 1, float32.
- **gpu-admm64** — fixed rho=1, float64.
- **gpu-admm64-adaptive** — adaptive rho starting from 1, float64.

The original 3000-iteration SCS and GPU studies are **not**
overwritten. After each backend, `--push` commits and pushes
only this experiment's public GitHub folder.

If one worker fails, the orchestrator records the failure and
continues. Full logs are retained locally and bounded log tails,
results, method rounding trials and metadata are published.
Before tearing down the VM, confirm the GitHub commits.

## What to compare

Open
`results/gset/g11-adaptive-rho-strict/comparison.csv`
or in PowerShell:

```powershell
Import-Csv results/gset/g11-adaptive-rho-strict/comparison.csv | Format-Table backend,precision,solve_seconds,continuous_primal_uncertified,admm_converged,primal_residual,dual_residual,rho_initial,rho_final,rho_updates,cut_best -AutoSize
```

The GitHub `cases/*.json` contain **every rho update**, its iteration,
old/new rho, primal/dual residuals, and scaled-dual multiplier,
along with the complete convergence history, graph hash,
solver statuses, timings, PSD defect, diagonal error,
twenty GW/FJ / K-Means2N / K-MeansNM rounding trials per method,
Torch/driver metadata and error outcomes.

Interpretation hierarchy:
1. Verify the original graph SHA matches the prior G11 studies.
2. Check numerical convergence vs both the **primal AND dual
   tolerances**, which may differ across solvers.
3. Check PSD defect, diagonal defect, continuous primal objective
   and rounding distribution **before** comparing wall time.
4. Compare elapsed time only for **comparable achieved accuracy**.
5. SCS `optimal_inaccurate` is not a certified optimum,
   and the GPU ADMM primal alone is not a certified upper bound.

If the default adaptive schedule oscillates or stalls, test
different `rho-update-interval`, `rho-balance-mu` or
`rho-scale` in **new output/publish directories**. The study
plan fingerprints all these settings and refuses to mix
incompatible experiment configurations. A later optimized
experiment should not retroactively replace the fixed-rho
baseline or conceal non-convergence.

## Engineering validation

- Pure scalar tests verify increase/decrease, clipping,
  invalid controls and unscaled-dual invariance.
- True float32/float64 numerical two-vertex tests of adaptive
  ADMM plus backward-compatible fixed rho are executed
  in **torch-first subprocesses** to avoid the observed
  Windows Torch `shm.dll` import-order problem.
- This code was not executed on the user's VM by the
  GitHub editing agent. Only published case files
  from a new VM run can support runtime or quality claims.
