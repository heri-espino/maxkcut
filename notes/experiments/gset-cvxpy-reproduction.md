# G-set SDP reproduction: experimental runbook

**Status:** implementation and synthetic tests are committed; the full G-set SDPs have **not yet been executed or validated against the checked-in graph files**. The initial development environment could not run CVXPY, so no G-set cut scores are claimed here. The package refactor does not itself constitute a benchmark run.

## Why this experiment?

Ángel Eduardo Rodríguez Fernández's 2021 dissertation, *On the Study of Fitness Landscapes and the Max-Cut Problem*, **Chapter 6 §6.3.2, printed p. 56**, reports use of:
- **MATLAB/CVX** on G1–G21;
- Mathematica **SemidefiniteOptimization** on G22–G54, G57–G59, G62–G67, G72;
- failures to obtain the SDP for G55, G56, G60, G61, G70, G77, G81.

The related [2020 journal article](https://doi.org/10.3390/computation8030075), §§2–3, describes the same G-set Max-Cut study and 50 repetitions of randomized rounding/clustering. This runbook aims at **faithfully reproducing the SDP + vector rounding components**, then extending to Max-$k$-Cut as a separate experiment. Note that chapter **7** uses a **different** A/B/C/D random graph benchmark, not G-set.

## Source files

The repository already contains a directory of MATLAB v5 files: `G-Set/G1.mat` through `G-Set/G71.mat` (including G68, G69 and G71, which are not part of the canonical G-set enumeration). The loader in `src/maxkcut/io.py` (exposed through `maxkcut.load_graph` and the backwards-compatible CLI script) uses `scipy.io.loadmat` and can handle a dense/sparse square weighted adjacency or a single 3-column edge list. It refuses mismatched symmetry, self-loops, invalid weights, and ambiguous candidate matrices.

**Important:** the repository's sequence includes G68/G69/G71 but is missing the canonical G72, G77 and G81 entries. These must not be silently considered equivalent. These exact MATLAB files have **not** been decoded and inspected during authoring. Run `--inspect-only` first. Verify, for example, G1 is expected to have $n=800$ and $m=19,176$; G11 $n=800,m=1,600$; G22 $n=2,000,m=19,990$, per [G-set reference metadata](https://github.com/KULeuven-MICAS/openising/blob/main/ising/benchmarks/G/README.md). If inspection differs, stop before benchmarking and investigate the input file conventions.

## Installation

Python 3.11+ is recommended. From the repository root:

```bash
conda env create -f environment.yml
conda activate maxkcut
python -m pytest tests -q
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Or use any virtual environment with `numpy`, `scipy`, `cvxpy`, and a PSD-capable conic solver (the default is `SCS`).

CVXPY is the **modeling language**, not the solver. Available solvers depend on the environment; see [CVXPY solver documentation](https://www.cvxpy.org/tutorial/solvers/).

### Verify the G-set files without solving an SDP

```bash
maxkcut --graphs G11 G1 G22 --inspect-only
python scripts/gset_benchmark.py --graphs ALL --inspect-only
```

This is a fast source-audit command and **does not import CVXPY**.

## The exact SDP

With symmetric adjacency matrix $W=(w_{ij})$, degree $D_{ii}=\sum_jw_{ij}$, and weighted Laplacian $L=D-W$:

$

\begin{aligned}
\max_{X}\quad&\frac{k-1}{2k}\operatorname{Tr}(LX)\\
\text{s.t.}\quad &X\succeq0,\qquad X_{ii}=1,\\
&X_{ij}\ge-\frac{1}{k-1}\quad(k>2).
\end{aligned}

$

For **$k=2$** the lower bound is redundant, and this reduces to the standard Goemans–Williamson SDP in Chapter 6:

$

\max_{X\succeq0,\operatorname{diag}X=\mathbf1}\frac14\operatorname{Tr}(LX).

$

For $k\ge3$, the extra lower-bound inequalities recover the Frieze–Jerrum relaxation in Chapter 7 (thesis Eq. 7.6).

An approximate SDP's **primal** score is not automatically a certified upper bound. Current script logs solver status and PSD/diagonal/$k$-way diagnostics. Solvers that return `optimal_inaccurate` are accepted for exploratory rounding but should be flagged and rerun with stricter tolerances before drawing inference; a certified approximation ratio requires a reliable dual or otherwise valid SDP upper bound.

### Rounding algorithms

- **$k=2$** baseline: Goemans–Williamson with one Gaussian random **hyperplane**; this is not two independently normalized random centroids.
- **$k>2$** baseline: Frieze–Jerrum, choosing the maximum projection onto $k$ independent **unnormalized Gaussian** directions.
- **$k=2$ clustering baselines:** **K-Means2N** uses two independent normalized random centroids; **K-MeansNM** initializes opposite centroids, matches the associated GW initial bipartition, and returns the best cut encountered along the clustering trajectory (including the initial one).
- **$k>2$ alternative:** spherical $k$-means, initialized with $k$ independently sampled and normalized Gaussian centroid directions.
- Both run from the **same factored SDP solution**; the script uses independent recorded seeds and restarts.
- Scores are recomputed using the **original weighted edges**, including signed weights.

The k-means implementation uses conventional normalized centroid updates and handles empty centroids. This is a functional reproduction of the rounding idea, not a guarantee of bitwise identity with the original MATLAB initialization, convergence tolerances, or tie-breaking. Chapter 6 also explores fuzzy c-means, medoids, MST clustering and local search; those variants are **not implemented in this first version**. K-MeansNM and K-Means2N are included. They should be added before claiming a full paper-level reproduction.

## Recommended staged execution

### Stage 0: synthetic tests and data audit

```bash
python -m pytest tests -q
python scripts/gset_benchmark.py --graphs G11 G1 G22 --inspect-only
```

### Stage 1: an 800-vertex Max-Cut instance

```bash
python scripts/gset_benchmark.py \
  --graphs G11 --k 2 --restarts 10 \
  --solver SCS --eps 1e-3 --max-iters 3000 \
  --output reports/gset/g11-k2.jsonl
```

G11 has fewer edges than G1, but **both require an 800-by-800 SDP matrix**. G11 therefore is *not guaranteed* to use much less memory. If G11 succeeds, attempt G1 separately:

```bash
python scripts/gset_benchmark.py --graphs G1 --k 2 \
  --restarts 20 --output reports/gset/g1-k2.jsonl
```

### Stage 2: test $k=3$ only after establishing $k=2$

```bash
python scripts/gset_benchmark.py --graphs G1 --k 3 \
  --restarts 20 --output reports/gset/g1-k3.jsonl
```

This adds $O(n^2)$ pairwise lower-bound constraints and can be much harder than the $k=2$ SDP.

### Stage 3: attempt 2,000 vertices only on a sufficiently large system

```bash
python scripts/gset_benchmark.py --graphs G22 --k 2 \
  --max-n 2000 --restarts 10 \
  --output reports/gset/g22-k2.jsonl
```

The CLI refuses graphs with $n>1000$ by default. The $n=2000$ problem has $2{,}001{,}000$ independent symmetric matrix entries before solver overhead. Avoid opening several SDPs simultaneously. An SDP cone's memory and factorization requirements can far exceed the matrix storage alone, and runtime is solver/hardware-dependent.

### Stage 4: controlled comparisons

Use identical SDP solutions, paired graph instances, documented seeds and solver settings. Compare cut, wall clock, solver residuals and the best-of-restarts trajectory. Compare against *best-known literature cuts* as benchmarks, but do **not** call those values proved optima without a certificate.

The script appends every restart to JSONL (not only maxima). **Use distinct output paths for separate experiments** or remove an output file before rerunning to avoid accidental duplicate records. Rounding labels are not saved in this initial version; best cut scores alone are insufficient to verify a new record.

## Running GitHub Actions

The repository provides a **manual-only** workflow: `.github/workflows/gset-sdp-manual.yml`. Use **Actions → G-set SDP benchmark (manual) → Run workflow**, select one graph and $k$. It runs tests and the requested SDP and uploads a JSONL artifact. The workflow is **not** triggered by pushes or pull requests, honoring the project's policy that heavyweight numerical runs must be explicitly launched.

Default job timeout: 35 minutes. The GitHub-hosted runner might run out of memory, or a large instance might exceed the time cap. Workflow existence does **not** imply a solve has completed, or that it can reliably solve G22.

## What counts as a scientific result?

- Input and edge counts verified against independent metadata.
- PSD solver reports acceptable feasibility diagnostics and statuses.
- Graph cut recomputed exactly from original edge weights.
- Multiple random seeds and meaningful spread, not just a single lucky maximum.
- Shared SDP solution when comparing rounding algorithms.
- Failure reports, timeouts, environment specs, and elapsed time archived.
- Max-Cut $k=2$ replication results kept separate from exploratory Max-$k$-Cut $k>2$.

**Current status:** code, tests, runbook, and opt-in workflow provided; no real G-set SDP run completed during authoring.

## SDP reuse and profiling (code update)

The command-line interface now caches the original solved Gram matrix by
SHA256(input file), $k$, solver and solver-version/tolerance settings. This is
**not** an additional scientific result. Use `--no-sdp-cache` if an independent
solve is desired; use `--scs-indirect` to separately profile SCS's iterative
KKT solver. Each JSONL row includes cache-hit and timing fields. Always use a
**new JSONL output file per run** to avoid mixing old repeats with new data.
See `docs/performance.md` for commands and restrictions.

## October 2026 benchmark protocol update: archiving and certified references

A recent Glass–Feder preprint (arXiv:2606.28478, June 2026) derives valid
toroidal G-set upper bounds. Table 1 confirms that canonical G11's published
feasible 564 equals its upper bound 564: **G11 has a proven optimum**, not
just a heuristic best-known cut. This does not imply all 71 G-set instances
have proven optima; see `docs/gset-reference-values.md`. The local
`G-Set/G11.mat` needs a verified exact graph identity before transporting
the published proof to that file.

Current library experiments use 20 restarts/method by default, preserving
the full score distributions. Every graph/k run additionally archives all
vertex partitions, a signed-weight graph snapshot and a provenance manifest
under `<JSONL-parent>/artifacts/` with unique run IDs. The original SDP
Gram matrix is separately cached by graph SHA256 + solver settings and need
not be regenerated just to examine the previous cuts. Check archive integrity
with `maxkcut --verify-archive PATH`, which rescales no SDP and simply
recomputes the stored cut weights. See `docs/experiment-archives.md`.

## Full checked-in G-set first pass: protocol only (no experiments run here)

Repository inventory in October 2026 contains 71 `G-Set/G*.mat`
files (`G1.mat` through `G71.mat`). To avoid 71 large CVXPY PSD cones,
`maxkcut-sweep` has a separate **sparse one-flip local-search baseline**.
Five independent random starts, 20 sweeps and a soft time budget of
60 seconds per graph are the initial exploratory defaults; this is
**not** SDP rounding and not itself a statistical significance study.
Graph, weighted edge snapshot, all per-seed partitions, objective scores,
elapsed time, and per-graph statuses are saved under
`reports/gset/full-first-pass/`; re-running with identical parameters
verifies and skips completed cases. See `docs/full-gset-sweep.md`.

Run the existing SDP/GW/K-Means pipeline separately on feasible small
G-set instances, preserving matched problem data and distinct method
labels. **No 71-graph benchmark has been executed by the authoring agent.**
