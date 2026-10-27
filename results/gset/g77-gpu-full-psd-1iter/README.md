# G77 — full convex SDP GPU precision and penalty-parameter study

This is a **new numerical comparison**, not a claim of improved quality or runtime until observed measurements are present.

## Approaches

- CPU SCS (float64), original full convex SDP reference.
- CPU SDPA (float64), alternative full convex solver.
- GPU full-rank PSD ADMM (float32 and float64), fixed or residual-balanced adaptive rho: the **same convex SDP formulation but a different first-order algorithm**. Its iteration limit is not an optimality certificate.
- Independent CPU/GPU PSD eigenprojection microbenchmark: **kernel time is NOT SDP solver time**.

## Saved results

[comparison.csv](comparison.csv) — solver status, timings, objective, feasibility and cut quality.

[rounding-trials.csv](rounding-trials.csv) — all rounded cuts.

[cases/](cases/) — complete backend JSON diagnostics.
[logs/](logs/) — sanitized bounded execution-log tails.

| Backend | Status | Seconds | SDP objective¹ | Best cut | ADMM residual converged | rho start→end | Updates |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| gpu-admm32-adaptive | computed | 4.330889099999979 | 3604.0052066743374 | — | no | 1 → 1 | 0 |

¹ The displayed SDP numerical primal is **not** a certified upper bound.

## Scientific caveats

- GPU ADMM is a new experimental solver, **not** a patch to SCS.
- Adaptive rho changes the numerical penalty only, not the original convex SDP; the *scaled* dual is rescaled whenever rho changes.
- Residual-balanced convergence does not certify SDP optimality or a numerical upper bound. Compare against fixed rho at the same absolute/relative tolerances, graph SHA and iteration budgets.
- Same convex feasible region, but different first-order numerical iterations, residual criteria and finite-precision effects.
- float32 trades precision for potentially faster matrix operations; do not assume it reproduces float64 solver outputs.
- An SDP primal objective is **not** a certified upper bound.
- A solver-only `--skip-rounding` run is a GPU feasibility/timing probe only: no CPU PSD verification, no partition and no cut score.
- All rounded cuts are re-evaluated against original signed edges.
- An ADMM iteration-limit result is approximate/unconverged.
- Never compare eigenprojection kernel time to total SCS solve time.
- Failed cases are retained and shown, not silently omitted.
