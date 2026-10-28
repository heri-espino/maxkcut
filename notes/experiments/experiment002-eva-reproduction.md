# experiment002 — EVA factors for reproducible Max-Cut rounding

**Status:** implementation prepared; GPU/CPU G-set benchmarks not yet run.
**Branch:** `experiment002-eva-factor-reproduction` (separate from unmerged
experiment001 GPU ADMM work).
**Primary reference:** Gonzalez-Torres et al. (2026), *EVA: A fast greedy
optimization algorithm for the Max-Cut Vector Relaxation Problem*,
Results in Control and Optimization 24, 100759.
DOI: https://doi.org/10.1016/j.rico.2026.100759
Original MATLAB: https://github.com/b3r8/EVA (Apache-2.0).

## Research questions

1. Does independent sequential double-precision EVA reproduce the
   published G-set vector-relaxation objectives at matching graph identity,
   rank, convergence tolerance and initialization protocol?
2. How does its numerical vector-relaxation objective and runtime compare
   with our full convex CVXPY/SCS baseline, without claiming numerical
   certification for either?
3. Using EVA's saved factor, how do 20× GW, K-Means2N and K-MeansNM
   signed-cut scores compare to the three SCS factor rounders?
4. Does smaller rank improve time while hurting objective/cut quality?
   Treat low rank as a distinct, explicitly nonconvex setting.

## Implementation

- `src/maxkcut/eva.py`: greedy sequential vector updates, row-unit
  validation, monotonic continuous score, angular stopping, atomic
  checkpoint including full vectors and RNG state.
- `src/maxkcut/eva_cli.py`: synthetic/real G-set experiment CLI,
  graph-identity-checked comparison, all candidate partition archives,
  optional opt-in periodic result commits and non-force pushes.
- `tests/test_eva.py`: small graphs with signed edges, monotonicity,
  resume/invalidation and full partition verification.
- `docs/eva-reproduction.md`: mathematical differences, caveats,
  PowerShell commands and complete artifact manifest.

## Prespecified protocol

Start G11 with `p=41`, tolerance `1e-3`, seed `2026`, 10,000
maximum sweeps and 20 restarts per rounding method. Validate against
both the original MATLAB implementation and our previous SCS run,
matching original canonical G-set input hashes. Only then scale to G77,
where `p=168`, and compare against any successfully archived full
convex G77 SDP case. Execute timing experiments independently.

Keep **all** factor files, iteration histories, seeds, signed cut
assignments, failures and numerical diagnostics in tracked
`results/gset/experiment002-eva/`, not solely ignored `reports/`.
For experiment progress on temporary VMs, use the opt-in
`--push --push-branch experiment002-eva-factor-reproduction
--push-every 250`.

**No benchmark figures or accuracy claims until the actual VM runs.**
