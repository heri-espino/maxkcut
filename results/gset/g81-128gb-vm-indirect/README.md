# G-set — Full convex SDP results

This directory is a **snapshot of completed, archive-verified** CVXPY/SCS Max-Cut SDPs. Updated by the local publication command. No GPU low-rank substitute or new SDP solve is run here.

**Verified complete:** 0 / 71 canonical graphs.

## Files

- [summary.csv](summary.csv): per-graph saved status and best feasible cut.
- [method-summary.csv](method-summary.csv): GW/FJ, K-Means2N, K-MeansNM distributions (20 roundings per method in the main study).
- [rounding-trials.csv](rounding-trials.csv): every independently recorded rounding score and its method/restart.
- [provenance.csv](provenance.csv): graph SHA, solver options, versions, iteration count and source commit.

## Status counts

| Saved status | Graphs |
| --- | ---: |
| started | 1 |

## Numerical SDP solver statuses

| CVXPY/SCS status | Completed graphs |
| --- | ---: |

Runs marked `optimal_inaccurate` reached only the solver's weaker numerical termination criteria. Review feasibility and duality diagnostics before drawing conclusions.

## Interpretation and limitations

- A rounded cut is a **feasible Max-Cut lower bound**, not a proof of the integer optimum.
- An SCS `optimal` status is numerical termination of the **continuous relaxation**, not proof of the best integer cut.
- The SDP primal objective is **not** asserted to be a certified upper bound, and a `started` status may be stale after a crash.
- Graphs marked `unverified_complete` have no published cut.
- Results use the canonical graphs in the repository's `data/` directory; original compact graph/partition archives and huge Gram matrix caches are retained locally under ignored `reports/`.
- Paths identifying the local workstation, solver logs and multi-GB matrices are intentionally excluded.

Project: [maxkcut](../../../README.md).
