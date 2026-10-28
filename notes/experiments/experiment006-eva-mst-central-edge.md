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
