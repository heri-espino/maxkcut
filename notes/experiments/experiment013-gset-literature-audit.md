# experiment013 — Source-attributed G-set literature benchmark

## Goal and status

Archive a dated 71-graph G-set reference snapshot alongside existing
experiment004 Numba EVA results. No solver, rounding algorithm, or
new graph optimization was run for this analysis.

The published values are *feasible cuts* (lower bounds on a maximum);
they are **not** all proven optima. Reference score gaps are not
primal-dual gaps or certified optimality bounds.

## Sources and precedence

1. [OpenIsing G-set benchmark snapshot](https://github.com/KULeuven-MICAS/openising/blob/5d22c7d5e3752ac7cdea8ababd361cdf46ba44bd/ising/benchmarks/G/README.md):
   all 71 historical scores, signed-weight flags, vertex/edge counts and
   graph families. Its K2000 extra instance is not part of the canonical
   G-set and is excluded from our 71-graph table.
2. [Glass and Feder, 2026, arXiv:2606.28478v1, Table 1](https://arxiv.org/html/2606.28478):
   17 toroidal instances have newer feasible cuts and independently
   published upper bounds. Nine cases have equal cut and upper bound:
   G11, G48, G49, G57, G62, G65, G72, G77 and G81.
   The other eight remain without an optimum proof from this paper.
3. [Liu et al., Enhanced Open-Source Scatter Search, 2025, Table 4](https://leria-info.univ-angers.fr/~jinkao.hao/papers/LiuetalCOR2025.pdf):
   newer cuts for G55, G56, G58, G59, G60, G61, G64 and G70.
4. [Khan and Shukla, 2025, arXiv:2510.21105](https://arxiv.org/abs/2510.21105):
   feasible cut 27047 for G63, without a matching optimum certificate.

Precedence is Glass/Feder for their 17 rows; Khan/Shukla for G63;
Liu et al. for their eight rows; OpenIsing for the remaining rows.
The historical OpenIsing column is retained for independent comparison.
Each selected row has an exact paper URL and an explicit proof status.
This is a **source-attributed snapshot**, not an assertion that all
selected scores are the latest best-known values as of every future date.

## Existing EVA data (no rerun)

Input: the committed experiment004 CSV at revision
ce876a35353348bd7a6ea7900998ad00ce66d222:
https://github.com/heri-espino/maxkcut/blob/ce876a35353348bd7a6ea7900998ad00ce66d222/results/gset/experiment004-eva-precision/summary.csv

The original run used Numba EVA, angular tolerance 1e-6, and at most
100000 sweeps. All 71 graphs have saved FJ, K-Means2N and K-MeansNM
rounded cut scores; 61 converged and 10 reached the iteration limit.
The compact input CSV keeps all 71 graph SHA256 values, three best
scores, iteration count, convergence status and sizes.

Files:
- data/gset_literature_reference_71.csv — literature table
- results/gset/literature-comparison/experiment004-eva-precision-source.csv
  — copied numeric observations for reproducibility
- results/gset/literature-comparison/experiment004-eva-precision-comparison.csv
  — derived comparison across all 71 graphs
- src/maxkcut/gset_literature.py — reusable offline comparator
- scripts/gset_literature_compare.py — CLI
- tests/test_gset_literature.py — offline validation

## Reproduce in PowerShell (no optimization)

~~~powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"
python -m pytest tests/test_gset_literature.py -q
python -m maxkcut.gset_literature --eva-summary results/gset/literature-comparison/experiment004-eva-precision-source.csv --references data/gset_literature_reference_71.csv --output results/gset/literature-comparison/experiment004-eva-precision-comparison.csv
~~~

The script accepts any later EVA summary with the same column scheme
using a different --eva-summary and --output. It checks graph names,
vertex/edge counts, source URLs, valid proof labels, nonfinite scores
and bound consistency. It never solves a new SDP or reruns EVA.

Graph identifier, size, edge count and the SHA256 of the *local* graph
do not independently verify equivalence with the **published external**
weighted-edge set. The report thus labels the identity check only
"size_and_edge_count_match_only"; use exact signed-edge comparison
against an external canonical source before transferring proof claims.

## Observed comparison

| Graph | Our experiment004 EVA best | Published reference | Score gap |
| --- | ---: | ---: | ---: |
| G63 | 26060 | 27047 | 987 |
| G77 | 9016 | 9940 | 924 |
| G81 | 12744 | 14060 | 1316 |

The high-precision EVA factors for G77 and G81 reached 100000
sweeps without satisfying the strict 1e-6 angular criterion. All
rounded partitions remained feasible candidate cuts. The literature
references for those two graphs match published upper bounds; our
EVA results are not optimality claims.

## Next

Compare future 1000-restart rounding and local-search campaigns to
the same dated table, retaining full per-graph distributions and
method-specific budgets. Update future literature citations by
adding source-attributed rows or a new version of the dataset.
