# Conducting reproducible experiments

The distinction between **source results**, **reproductions**, and **new experiments** is mandatory.

## Study sequence

1. Validate the checked-in MATLAB graph files with `maxkcut --graphs G11 G1 G22 --inspect-only`.
2. Validate known small mathematical cases with unit tests; separately perform the optional tiny conic-solver smoke test.
3. Run a *single* G-set graph with explicit solver tolerances, memory budget and random seed.
4. Compare **GW hyperplane**, **K-Means2N**, and **K-MeansNM** using a shared $k=2$ SDP matrix.
5. Only after establishing $k=2$ feasibility, explore $k=3,4,5$ with Frieze–Jerrum and spherical $k$-means. These are **new G-set experiments**, not reproductions of Chapter 7's A/B/C/D random graphs.
6. Log failed solves and statuses rather than dropping them.

For details, see [G-set reproduction protocol](https://github.com/heri-espino/maxkcut/blob/main/notes/experiments/gset-cvxpy-reproduction.md) and [background sources](https://github.com/heri-espino/maxkcut/tree/main/notes/background).

## Numerical validity

For $k\ge2$, solve

$$
\max_{X\succeq0,\,X_{ii}=1}
\frac{k-1}{2k}\operatorname{Tr}(LX),\qquad
X_{ij}\ge -\frac{1}{k-1}\; (k>2).
$$

All discrete partitions are feasible in the relaxation, giving (in exact arithmetic) $F(c)\le F^*\le S_{\mathrm{SDP}}$. **However**, the primal score returned by an approximate maximization solve is not an automatically certified upper bound. Check residuals, eigenvalues, and preferably a validated dual upper bound before making approximation claims.

Keep separate columns for **cut weight**, **SDP primal objective**, **solver-reported/verified upper bound**, **clustering inertia**, **wall time**, **seeds**, and **number of nonempty groups**.

## Seed discipline

- Fix independent graph seeds and rounding seeds. Record the exact seed, solver/version/tolerances and environment.
- K-MeansNM at $k=2$ starts at a partition corresponding to GW's random hyperplane and preserves the best cut visited. This retains the paired initial cut, **not** a new universal approximation factor.
- Ensure fair compute budgets (same SDP, comparable restarts, report wall-clock time).
- In statistical comparisons use graphs as the paired experimental units; multiple restarts on a single graph are not independent graph instances.

## Artifacts

Store uncommitted raw outputs under `reports/` (ignored by Git). Each research note in `notes/experiments/` should identify the exact command, commit, inputs, status, environment and any paths to saved artifacts, together with mathematical definitions of plotted quantities. Link figures from `notes/` if curated for a PR and reasonably sized.

No fabricated trial counts, timings, p-values, solvers, or claimed 'optimal' values.

## GitHub Actions policy

Heavy optimizations and figure/PDF generation require **manual `workflow_dispatch`**. Small tests may run automatically when added later, but do not introduce expensive push-triggered workflows. Do not open many concurrent G-set SDP jobs on a shared host.

## Efficient repeated rounding

Use the SDP cache instead of recomputing the same 800×800 cone problem when
comparing many random initializations. New cache entries are keyed by graph
checksum, $k$, solver/version and numerical settings; they do not change the
mathematical experiment. Save each batch into its own JSONL output file and
log whether the SDP came from cache. See [performance](performance.md).
