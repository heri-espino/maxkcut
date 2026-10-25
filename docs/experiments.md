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

Store uncommitted raw outputs under `reports/` (ignored by Git).
By default, the library also writes a graph/k case archive with the original
signed edge list, graph SHA256, run parameters/versions/commit, **every**
rounded partition and score. The SDP Gram is reused from the separate
cache, not stored per restart. A `started` or `sdp_ready` archive is
incomplete, not a successful experiment. Use `maxkcut --verify-archive`
to independently rescore every archived partition. See
[experiment archive format](experiment-archives.md).
For published optimality labels, see
[proven versus best-known G-set cuts](gset-reference-values.md). Each research note in `notes/experiments/` should identify the exact command, commit, inputs, status, environment and any paths to saved artifacts, together with mathematical definitions of plotted quantities. Link figures from `notes/` if curated for a PR and reasonably sized.

No fabricated trial counts, timings, p-values, solvers, or claimed 'optimal' values.

## GitHub Actions policy

Heavy optimizations and figure/PDF generation require **manual `workflow_dispatch`**. Small tests may run automatically when added later, but do not introduce expensive push-triggered workflows. Do not open many concurrent G-set SDP jobs on a shared host.

## Efficient repeated rounding

Use the SDP cache instead of recomputing the same 800×800 cone problem when
comparing many random initializations. New cache entries are keyed by graph
checksum, $k$, solver/version and numerical settings; they do not change the
mathematical experiment. Save each batch into its own JSONL output file and
log whether the SDP came from cache. See [performance](performance.md).

## Numerical SDP postprocessing policy

First-order solvers can return a small negative Gram eigenvalue even when
termination criteria were satisfied. The SDP solution is stored in its raw
solver-produced form; rounding factorization may clip small negative eigenvalues
and rescale vector rows to unit norm. This is **not** a dual upper-bound
certificate and is not Higham's exact nearest-correlation-matrix algorithm.

Default acceptance requires
`max(0,-lambda_min) / max(1,lambda_max) <= 1e-3`. The operation logs
`psd_projection_applied`, `psd_relative_negative_eigenvalue`,
`psd_negative_eigenvalue_count` and `psd_negative_eigenvalue_sum` for
every resulting cut, and always retains `min_eigenvalue` from the original
matrix. More substantial violations remain fatal; do not increase this
threshold merely to obtain a cut without considering solver accuracy.

When producing a scientific comparison, rerun with stricter SCS tolerances
and record whether rounding scores are stable. For $k>2$, vector
renormalization may violate the Frieze–Jerrum lower bounds, although labels
still constitute a valid discrete partition.

## Two-stage G-set research workflow

The entire checked-in corpus (G1–G71) can be examined and subjected to
an inexpensive, bounded **sparse one-flip local-search baseline** using
`maxkcut-sweep`. It saves each graph's cut assignments and scores and is
fully resumable. It **must not** be reported as an SDP, GW, or spherical
K-Means experiment; it uses no SDP and produces no certified upper bound.
See [full-corpus first-pass protocol](full-gset-sweep.md).

Only after that baseline should we spend solver time on a manageable
SDP subset, retaining the existing `maxkcut` results and caching each
Gram matrix. Do not assume that trial counts on the initial heuristic
baseline supply statistical significance for SDP rounding algorithms.

## Canonical graph formats and reference comparisons

The canonical complete G-set input is now **`data/G*.txt`** (71 files:
G1–G67, G70, G72, G77, G81). `maxkcut` and `maxkcut-sweep` use that
folder by default. The legacy `G-Set/*.mat` graphs remain available
with `--gset-dir G-Set`, and `maxkcut-sweep --compare-inputs` checks
all weighted edges, not just labels, between the directories.

The `comparison.csv` produced by the sweep joins the **17 toroidal
instances** in Glass and Feder (2026), Table 1, reporting the paper's
best cut, certified upper bound and whether they agree. Absent entries
are explicitly unreferenced; never assume the other 54 graph optima have
been proved. See [comparison protocol](canonical-data-comparison.md).

## Optional CUDA low-rank factor benchmark

A distinct `maxkcut-gpu` entry point optimizes rank-r unit vectors using
the original sparse signed graph, with optional PyTorch CUDA acceleration,
and then evaluates GW, K-Means2N, K-MeansNM on the resulting factors.
This is **not a reproduction of the full SDP solution** and is not a
certified SDP upper bound. Distinct method labels carry the
`_torchlowrank` suffix to prevent conflating algorithms.

CUDA optimizer starts and rounding restarts are distinct compute budgets.
Factor cache keys include graph SHA256, rank, epochs, learning rate,
optimizer-start count, Torch/CUDA versions, device and optimizer seed.
`--rounding-seed` changes only GW/K-Means random starts and does **not**
invalidate that cached factor.

All results retain exact weighted cuts, rank-r factors, optimizer histories,
timings, GPU identity and every rounded partition. Compare them to the
full convex SDP/rounding baseline only on canonical identical graphs and
at disclosed compute budgets. See [CUDA low-rank guide](cuda-lowrank.md).
