# Experiment007 — Two-stage Max-Cut optimization: MST-BestCut + flips

## Hypothesis

The one-edge MST constraint limits partition quality, even when the
best of all n-1 tree edges has been selected. Start with the *saved*
best-single-MST-edge partition for each of L1, Euclidean L2 and Linf,
then perform a second discrete Max-Cut optimization by flipping
individual vertex labels.

This second stage is a **warm-started unrestricted binary local search**
on the original signed-weight G-set graph, not an improved MST edge
selector. After flips, the two color classes need NOT be connected
subtrees of the original MST. Call the result `MST-BestCut-Flip`, not
`MST-BestCut` or a new global Max-Cut optimum.

## Mathematical derivation

For a binary assignment s_i in {-1,+1}, the Max-Cut objective is

    C(s) = (1/4) sum_{i,j} w_ij (1 - s_i*s_j)

for symmetric undirected graph weights with zero diagonal. Flipping
one vertex i gives

    Delta_i = s_i * sum_j w_ij * s_j.

The existing `src/maxkcut/local_search.py` implements this exact
signed-weight gain formula. A flip is accepted only when
`Delta_i > 1e-10`, so the discrete cut never decreases (up to
floating-point rescoring tolerance). Neighbor gains update
incrementally in sparse CSR format, avoiding a dense distance or SDP
matrix. It stops on a sweep with no improving flips, or after the
declared max-sweep limit. An iteration limit is NOT a certificate of
local optimality. A completed empty sweep means a 1-flip local optimum;
it is not a global optimum.

For G63's already reported full-dimensional saved factor and L2 MST:

    stage1 e* cut = 21,055
    stage2 MST-BestCut-Flip >= 21,055 (by monotone construction)
    prior best EVA randomized rounder = 26,055

Whether the new method exceeds 26,055 remains untested until the
user runs the command. EVA itself optimized the continuous vectors
earlier, so the entire pipeline comprises EVA, MST edge selection,
and local search; no EVA rerun is needed now.

## Protocol

- Exactly one saved original-ambient MST per metric, from experiment005.
- Stage 1 from experiment006: evaluate ALL n-1 tree-edge partitions to
  obtain the maximum signed weighted cut, already stored in
  `center-cases` plus an independently verified partition archive.
- Stage 2: copy that exact original partition and run the repository's
  sparse positive-gain vertex-flip optimizer.
- Default `--flip-sweeps 20`; `--flip-seed 2026` determines shuffled
  vertex visitation order. Unlike the MST rule, the final flip local
  optimum can depend on the visitation seed. The default is one
  seeded trial, not a claim of 1,000 independent restarts.
- Archive the original weighted graph and the revised 0/1 partition;
  independently score it and prove `C_after >= C_before` numerically.
- Preserve original graph SHA256, factor SHA256, archived stage-1
  partition pointer, stage-1 score, stage-2 score, number of flips,
  number of completed sweeps, status, timing and seed.
- Never recompute EVA or a geometric MST from this program.

## PowerShell runbook

Before switching away from the branch that currently holds original
MST results, **save and push those results**:

    git add .
    git commit -m "experiment006: save original MST partitions and diagnostics"
    git push

Switch to the implementation branch and merge the latest original
MST results so stage 2 can read them in the working directory:

    git fetch origin
    git switch --track origin/experiment007-eva-mst-bestcut-flips
    git merge origin/experiment006-eva-mst-central-edge

If `experiment007-eva-mst-bestcut-flips` already exists locally,
use `git switch experiment007-eva-mst-bestcut-flips` instead of the
track command. If `git commit` reports nothing to commit, the
originals may already have been committed; verify their existence
in the selected branch before running the second stage.

Test before performing G-set runs:

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_mst_refine.py -q

G63 pilot reusing existing experiment006 archives:

    python -m maxkcut.eva_mst_refine_cli --graphs G63 --metrics l1 l2 linf --flip-sweeps 20 --flip-seed 2026 --mst-dir results/gset/experiment005-eva-mst --output-dir results/gset/experiment007-eva-mst-flips

For all graphs with saved best-edge partitions (missing cases reported):

    python -m maxkcut.eva_mst_refine_cli --graphs ALL --metrics l1 l2 linf --flip-sweeps 20 --flip-seed 2026 --mst-dir results/gset/experiment005-eva-mst --output-dir results/gset/experiment007-eva-mst-flips --continue-on-error --push --push-branch experiment007-eva-mst-bestcut-flips

The `--push` flag triggers a **single push after the whole batch**.
If an existing experiment006 directory contains multiple factor
versions for one graph/metric, specify `--factor-sha-prefix` to
avoid arbitrary selection. Distinct seeds and maximum-sweep settings
produce append-only distinct case records and separate archives.

## Output

- `cases/`: one SHA-bound result per graph, metric, seed, sweep cap.
- `archives/`: original weighted graph snapshot and compressed
  refined partition, independently re-scored and verified.
- `trials/`: exact initial/refined score and search termination
  metadata, one candidate per seeded trial.
- `summary.csv`: before/after scores, improvements, sweeps and status.
- `errors/`: any missing source archived MST results or failures.

## Interpretation and controls

A substantial increase over MST-BestCut means the **single-tree-edge
connectivity restriction** was a major limitation. It does not show
that MST geometry outperforms standard EVA rounding. Compare against
the original GW, K-Means2N and K-MeansNM results **with and without
the same flip local-search refinement**, on identical signed input
graphs and transparent compute budgets. If the MST warm start is
still much worse after flips, the initial geometry/labels may also
be important. All outcomes are empirical until validated.
