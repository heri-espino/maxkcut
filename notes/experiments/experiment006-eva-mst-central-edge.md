# Experiment006 — Central-edge choices for the EVA MST

## What is the "center edge" of a tree?

An MST has several different meaningful centrality notions. We compare
four partitions, each obtained by deleting EXACTLY ONE edge from an
exact full-dimensional EVA embedding MST for L1, L2 or Linf.

1. Longest MST edge: experiment005 baseline, maximizing geometric edge length.
   This can isolate outliers and does not guarantee a good Max-Cut.
2. Balanced edge: delete e yielding components S and V minus S and maximize
   B(e)=|S|*(n-|S|). This minimizes split-size imbalance and maximizes
   edge betweenness in the tree.
3. Weighted diameter-center edge: find a diameter of the tree weighted
   by the selected geometric metric and remove the edge containing
   its length midpoint. The mathematical tree center is actually a
   point. If the midpoint lies exactly on a vertex rather than in
   the interior of an edge, there is no unique center edge. We choose
   an incident diameter edge deterministically.
4. Best-cut edge: choose the edge with the maximum ACTUAL original
   signed weighted G-set Max-Cut score. This is a supervised,
   objective-informed benchmark, NOT the geometric tree center.

For an original graph edge (u,v,w), removing an MST edge e separates
u and v iff e belongs to the unique MST path between them. Therefore

    cut(e) = sum_{original (u,v,w)} w * 1[e in MST-path(u,v)].

An O((n+m)log n) LCA path-addition algorithm evaluates ALL n-1 tree
edge cuts simultaneously, including signed weights, then selects the
best. The balanced edge is O(n), as is the diameter-center choice.

## Computing without PCA

All MSTs use the entire n-by-p SAVED EVA factor Y in the original
ambient space. We never project PCA coordinates or construct an
n-by-n pairwise distance matrix. There is no SDP or EVA re-solve,
and each metric/edge-selection rule produces exactly ONE
deterministic partition for each saved factor.

## Windows PowerShell

Do not switch Git branches during an active EVA run.

    git fetch origin
    git switch --track origin/experiment006-eva-mst-central-edge
    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_mst.py tests/test_eva_mst_center.py -q

Pilot on G63:

    python -m maxkcut.eva_mst_center_cli --graphs G63 --max-n 10000 --metrics l1 l2 linf --rules balanced diameter_center bestcut --source-dir results/gset/experiment003-eva-numba --output-dir results/gset/experiment005-eva-mst

Then all 71 canonical instances, using only saved Numba or NumPy
factors; a missing MST is computed once by exact Prim, and a
missing factor is reported in center-errors. Results push ONCE
at the end, not once per tree edge:

    python -m maxkcut.eva_mst_center_cli --graphs ALL --max-n 20000 --metrics l1 l2 linf --rules balanced diameter_center bestcut --source-dir results/gset/experiment003-eva-numba --fallback-source-dir results/gset/experiment002-eva --output-dir results/gset/experiment005-eva-mst --continue-on-error --push --push-branch experiment006-eva-mst-central-edge

## Saved artifacts

- Original MST tree parents, edge lengths and longest-edge archives
  remain under results/gset/experiment005-eva-mst/.
- center-cases/ stores one complete metric/rule/factor-hash record.
- center-trials/ stores each exact weighted score.
- center-archives/ stores bit-packed graph partitions and verified scores.
- center-summary.csv compares center choices with longest-edge scores.
- center-errors/ retains failed graph reasons.

A best-cut tree edge is optimal only AMONG the n-1 single-MST-edge
partitions, not among all possible 2^(n-1) Max-Cut partitions. MST
edge-selection does not necessarily beat GW or spherical k-means.
The tests independently rescore all tree-edge cuts on random signed
small graphs, validate balanced and diameter-center selections and
audit saved reproducible partition archives.

Manual Git persistence, if automatic push is disabled:

    git add .
    git commit -m "experiment006: central MST cuts and comparisons"
    git push

## Interpreting e*: geometry, balance and signed path load

For each metric-specific exact MST T, the code now computes and
commits all n-1 candidate edge statistics rather than recording only
the highest-scoring edge:

    C(e) = sum_original_(u,v,w) w * 1[e belongs to unique T path(u,v)]
    B(e) = s_e * (n-s_e)
    C(e) = B(e) * mean_cross_pair_weight(e)

Here s_e is the cardinality of a component after removing e, and
the cross-pair mean treats *absent* original graph edges as zero. Thus:

- **Geometric gap:** the chosen edge's original-ambient metric distance,
  midrank length percentile (0 = shortest, 1 = longest), and whether
  e* ties for the longest.
- **Tree centrality:** s_e, n-s_e, B(e), split-size imbalance, midrank
  balance percentile, and whether e* ties for the most-balanced edge.
  For a complete uniform-positive original graph, C(e)=B(e) exactly
  and an optimal tree-edge cut is necessarily most balanced.
- **Metric tree center:** whether the best Max-Cut edge is the chosen
  weighted-diameter midpoint edge (not generally the same as balanced).
- **Signed edge demand:** the sum of positive and negative original
  crossing weights, their edge counts, the net cut weight and the
  average weight per potential separated vertex pair.
- **Within-tree relationships:** Spearman rank correlations of
  C(e) with metric length, C(e) with B(e), and metric length with
  B(e), across all n-1 candidate tree edges. Undefined
  correlations (constant quantities) are represented as null.

The winning edge maximizes the net signed demand routed over a tree
edge. It need not be longest, central or balanced; a negative crossing
weight *reduces* the Max-Cut objective. The length and balance
percentiles describe it, but do not confer a global Max-Cut guarantee.

Interpretation cautions: (i) e* is selected using C(e), so its score
percentile is trivially high; do not report that as evidence of a
discovery. (ii) Candidate tree edges within a graph are not
independent samples. (iii) A high score-length correlation may follow
the EVA factor's construction from the same original weighted graph;
report this as an association, not causal evidence. (iv) Benchmark
the three distance metrics on each graph separately, then aggregate
per-graph summaries rather than treating all edges as independent
graph replicates.

### Saved new outputs

- `edge-profiles/Gxx-<factorhash>-mst-<metric>.npz` preserves **every**
  MST edge's child, parent, original-ambient length, signed original
  Max-Cut, subtree size, B(e) and normalized cross-pair mean.
- `edge-profile-reports/Gxx-<factorhash>-mst-<metric>.json`
  preserves e* with all length, balance, score and signed-demand
  interpretability diagnostics and exact factor/graph identity.
- `edge-profile-summary.csv` has one row per graph and metric,
  including e* length and balance percentiles and the rank
  correlations for cross-instance analysis.

The experiment006 CLI generates these files automatically while
reusing saved exact MSTs. No new EVA optimization or PCA is needed.
