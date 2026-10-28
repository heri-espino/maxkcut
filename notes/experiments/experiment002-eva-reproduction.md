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


## Extension: full 71-instance G-set sweep

`maxkcut-eva --graphs ALL --max-n 20000 --continue-on-error`
runs all 71 canonical G-set graphs sequentially and resumes verified
EVA factors and rounded partition archives. The experiment compares
against the first independently archive-verified full convex SDP case
with a matching graph checksum, searching in the declared order:

1. `reports/gset/full-sdp-batch` (previous smaller graph experiments)
2. `reports/gset/full-sdp-threads32` (subsequent large graph experiments)

The user reported the following **rounded cuts** from successful
full convex SCS runs: G60=13663, G61=5270, G62=4628; G63 was interrupted.
These values are user-observed solver output, **not** measured EVA
objectives or cut scores, and need the on-machine saved archives for
independent verification.

The tracked `results/gset/experiment002-eva/summary.csv` separates
uncertified continuous primal objectives, rounded cut values for
all three methods, runtime, convergence status, and missing references.
Actual large EVA experiments have not yet been run by this authoring
session. Preserve `factors/`, `logs/`, `trials/`, `archives/`,
`cases/`, `errors/` and `summary.csv` and commit/push before VM removal.

Full PowerShell commands: `docs/eva-reproduction.md`.
