# Experiment005 — Exact MST rounding in the original EVA ambient space

## Objective

Can the minimum spanning tree (MST) of the *full, unprojected*
EVA factor improve Max-Cut rounding? Compare L1, L2 (Euclidean) and
L-infinity metrics, and compare the resulting cuts with the existing
GW, K-Means2N and K-MeansNM roundings.

This is explicitly **NOT PCA**. Every row of the saved EVA n-by-p factor
is a point in the original p-dimensional embedding. For G81, p=201.

## Method

For all pairs of factor rows, define conceptual complete-graph lengths:

- L1: d(x,y) = sum over coordinates of abs(x_a-y_a)
- L2: d(x,y) = sqrt(sum over coordinates of (x_a-y_a)^2)
- Linf: d(x,y) = max over coordinates of abs(x_a-y_a)

Compute the exact complete-graph MST using a Numba-compiled Prim loop.
In the L2 case the kernel compares squared distances, which preserve
Euclidean distance order. Remove exactly the **one** longest MST edge.
Because a tree contains n-1 edges, removing one edge makes exactly two
components. Label their vertices 0/1, and evaluate the ORIGINAL
signed-weight G-set Max-Cut objective using the ORIGINAL graph weights.
Those signed graph weights never define the geometric MST lengths.

This is deterministic for each factor and metric, with vertex-index tie
breaking. There are no meaningful 1,000 random restarts for MST on a
fixed factor. A singleton outlier split may perform badly on Max-Cut;
always report component sizes and evaluate the actual weighted cut.

## Computational cost

The algorithm is exact, not approximate kNN: O(n^2 p) time and O(n p)
space plus the original graph. The full n-by-n distance matrix is never
materialized. G81 requires about 40 billion coordinate-pair comparisons
for each metric. Pilot on G63 before G77/G81; the geometric MST can
cost much more than the already-fast Numba EVA optimization.

## Windows PowerShell validation

Run only after any active EVA process finishes.

    git fetch origin
    git switch --track origin/experiment005-eva-mst-single-linkage
    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_mst.py -q

Pilot on one Numba saved factor:

    python -m maxkcut.eva_mst_cli --graphs G63 --metrics l1 l2 linf --max-n 10000 --source-dir results/gset/experiment003-eva-numba --output-dir results/gset/experiment005-eva-mst

Run over nine large saved Numba factors. This script does not optimize
EVA again. The --push flag commits and pushes only ONCE at the end.

    python -m maxkcut.eva_mst_cli --graphs G63 G64 G65 G66 G67 G70 G72 G77 G81 --metrics l1 l2 linf --max-n 20000 --continue-on-error --source-dir results/gset/experiment003-eva-numba --output-dir results/gset/experiment005-eva-mst --push --push-branch experiment005-eva-mst-single-linkage

For other graphs, use --source-dir results/gset/experiment002-eva with
a distinct destination (e.g. results/gset/experiment005-mst-numpy).
A source directory containing only nine cases cannot serve --graphs ALL.

## Permanent artifacts and verification

- Frozen factor bytes and SHA256 in factor-snapshots/.
- Complete MST parent arrays, metric edge lengths and longest edge in trees/.
- Graph snapshot and exact bit-packed partition in archives/.
- Individual metric cases in cases/Gxx-factorhash-mst-metric.json.
- One full original weighted cut per metric, with component sizes.
- Summary.csv and per-graph errors in errors/.

Archives are independently rescored. Reruns verify and skip completed
cases; new factors create different factor-SHA case names.
When using manual sync, commit all meaningful code/results:

    git add .
    git commit -m "experiment005: original-ambient EVA MST results"
    git push

The test suite compares MST weights against an independent SciPy
dense-distance Kruskal reference on small graphs, checks L1/L2/Linf,
full-dimensional no-PCA behavior, deterministic ties, and validates
signed-weight saved partition scores.

No numerical G-set result is claimed before the local experiment
has run and its outputs have been committed to GitHub.
