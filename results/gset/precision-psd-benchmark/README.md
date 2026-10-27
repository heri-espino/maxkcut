# G11 — float32, GPU PSD eigendecomposition and SDPA study

This is a **new numerical comparison**, not a claim of improved quality or runtime until observed measurements are present.

## Approaches

- CPU SCS (float64), original full convex SDP reference.
- CPU SDPA (float64), alternative full convex solver.
- GPU full-rank PSD ADMM (float32 and float64): the **same convex SDP formulation but a different first-order algorithm**. Its iteration limit is not an optimality certificate.
- Independent CPU/GPU PSD eigenprojection microbenchmark: **kernel time is NOT SDP solver time**.

## Saved results

[comparison.csv](comparison.csv) — solver status, timings, objective, feasibility and cut quality.

[rounding-trials.csv](rounding-trials.csv) — all rounded cuts.

[cases/](cases/) — complete backend JSON diagnostics.
[logs/](logs/) — sanitized bounded execution-log tails.

| Backend | Status | Solver status | Seconds | Best cut |
| --- | --- | --- | ---: | ---: |
| cpu-eigh64 | computed | not_an_SDP_solve | 0.027923399989958853 | — |
| cpu-scs64 | computed | optimal | 41.25 | 546.0 |
| cpu-sdpa64 | failed | failed | — | — |
| gpu-admm32 | computed | admm_iteration_limit | 0.9272986000287347 | 546.0 |
| gpu-admm64 | computed | admm_iteration_limit | 2.730229099979624 | 544.0 |
| gpu-eigh32 | computed | not_an_SDP_solve | 0.010399800026789308 | — |
| gpu-eigh64 | computed | not_an_SDP_solve | 0.02593539998633787 | — |

## Scientific caveats

- GPU ADMM is a new experimental solver, **not** a patch to SCS.
- Same convex feasible region, but different first-order numerical iterations, residual criteria and finite-precision effects.
- float32 trades precision for potentially faster matrix operations; do not assume it reproduces float64 solver outputs.
- An SDP primal objective is **not** a certified upper bound.
- All rounded cuts are re-evaluated against original signed edges.
- An ADMM iteration-limit result is approximate/unconverged.
- Never compare eigenprojection kernel time to total SCS solve time.
- Failed cases are retained and shown, not silently omitted.
