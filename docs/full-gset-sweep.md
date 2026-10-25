# First-pass G-set sweep (all checked-in instances)

**Purpose:** obtain a feasible Max-Cut baseline on every available graph,
without attempting a prohibitively expensive full semidefinite program.

As of 2026-10-24 the repository contains 71 MATLAB files, **G1.mat through
G71.mat**. In particular **G81.mat is not checked in**. This sweep discovers
files at runtime rather than assuming they are present.

## The two separate experiment types

- `maxkcut` (existing): CVXPY/SCS **SDP + GW/FJ and spherical K-Means rounding**. Intended for smaller cases with feasible SDP resource budgets. Caches the original numerical SDP matrix.
- `maxkcut-sweep` (new): sparse **random binary starts + positive-gain single-vertex flip local search**. Does **not** call CVXPY, SCS, SDPA, a GPU or a full SDP. It gives a fast feasible-cut baseline for all graphs. **Do not label these results GW, K-MeansNM or SDP optimal values.**

Mathematically, write a binary partition as spins $s_i\in\{-1,1\}$. If
vertex $i$ is flipped, its objective gain is

$$
\Delta_i = s_i\sum_{j} w_{ij}s_j.
$$

Positive gains are accepted and neighboring gains updated incrementally using
the sparse CSR adjacency. A complete sweep with no accepted move is a
**single-flip local optimum**, *not* a globally optimal cut or an SDP solution.
Negative G-set edge weights are preserved throughout.

## Windows PowerShell commands

**After pulling the new feature**, refresh the editable console-script entry:

```powershell
git pull origin main
conda activate maxkcut
python -m pip install -e .
python -m pytest -q
```

Inspect every graph without optimization:

```powershell
maxkcut-sweep --inspect-only
```

Test the first **two** numerically ordered graph files only:

```powershell
maxkcut-sweep --max-graphs 2 --restarts 5 --max-sweeps 20 --seconds-per-graph 60 --output-dir reports/gset/full-first-pass
```

Then process **all 71 checked-in files** with the **same settings**:

```powershell
maxkcut-sweep --restarts 5 --max-sweeps 20 --seconds-per-graph 60 --output-dir reports/gset/full-first-pass
```

The second command **verifies and skips** already completed graph archives
(G1 and G2 from the smoke run), then continues with the rest. Run the
*same command again* after an interruption; it will skip intact completed
archives, record failures, and continue. It never queues 71 SDPs.

The per-graph 60-second limit is **soft**: it is checked inside the local
search every 128 vertex visits and between restarts. Graph loading,
initialization and writing snapshots are not forcefully terminated by a
separate process. If a deadline is reached, the current feasible cut is
saved as a **budget-limited** result, not mislabeled converged.

Each case is processed **sequentially**; there is no concurrent batch of
memory-intensive solvers.

## Outputs and reproducibility

```text
reports/gset/full-first-pass/
├── plan.json               # immutable configuration for this output directory
├── summary.csv             # one row per graph: status, cuts, times, archive
├── graphs/
│   ├── G1.json             # per-graph status, checksum, settings
│   ├── G2.json
│   └── ...
├── trials/
│   ├── G1.jsonl            # every seed, initial/final cut, flips, time, sweeps
│   └── ...
└── artifacts/
    └── <run-id>/G1-k2/
        ├── manifest.json   # code version, graph fingerprint and metadata
        ├── graph.npz       # original normalized signed edge weights
        └── random_flip_local_search.npz
                             # ALL compact bit-packed partitions and scores
```

An individual graph is **skipped only after** its saved graph checksum and
settings match and its archived partitions are independently re-scored.
An input file changed in place triggers a new run.

`summary.csv` is rewritten safely after each graph, so partial work
survives process interruption. Graph failures remain explicit and do not
silently exclude remaining instances.

In PowerShell, inspect the table:

```powershell
Import-Csv reports/gset/full-first-pass/summary.csv | Format-Table graph,n,m,status,best_cut,total_seconds
```

All output is under the Git-ignored `reports/` directory. To run a
*different* seed, restart count, sweep count, or time budget, choose a
**different output directory**. The runner refuses to combine incompatible
experiments under one `plan.json`.

The library's archive verification CLI also works on the case archive:

```powershell
maxkcut --verify-archive "reports/gset/full-first-pass/artifacts/YOUR-RUN-ID/G1-k2"
```

## Comparing results responsibly

- The best-of-five cut is a preliminary **heuristic lower bound on the
  optimum** for each graph; statistical significance is **not** implied.
- Where a **proved optimum** exists and the local graph identity matches
  canonical G-set, compute cut / optimum as an achieved solution-quality
  ratio. See [reference values](gset-reference-values.md).
- Where literature reports only *best-known* cuts, report the *gap to
  best-known*, not a certified approximation factor.
- For generalization across G-set problems, graphs—not restarts of the
  same graph—are the natural paired experimental units.
- A future *matched-compute-budget* comparison should also run the
  original `maxkcut` SDP/GW/k-means methods on feasible subsets and
  compare their entire cut distributions with the fast baseline.
- We do **not** claim that five starts or 60 seconds is the best possible
  budget, nor that this local-search algorithm scales to every graph
  topology with equal quality.

## Reference

The [Glass–Feder paper (2026)](https://arxiv.org/html/2606.28478),
especially Table 1, proves some toroidal G-set optimal values by matching
a feasible cut with a valid upper bound. Do not apply its certified optimum
to an unverified local graph with a matching filename alone.
