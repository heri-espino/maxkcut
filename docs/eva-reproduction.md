# EVA vector-relaxation reproduction (experiment002)

## Aim

Reproduce **Efficient Vector Ascent (EVA)** from Gonzalez-Torres,
Rodriguez-Fernandez, Menchaca-Mendez, Menchaca-Mendez & Schütze,
*Results in Control and Optimization* 24 (2026), 100759,
DOI [10.1016/j.rico.2026.100759](https://doi.org/10.1016/j.rico.2026.100759).

The [original MATLAB code](https://github.com/b3r8/EVA/blob/main/EVA.m)
(licensed Apache-2.0) implements sequential greedy coordinate updates.
It records objective/time histories but **does not save the final factor**.
Our **independent Python implementation** uses the same mathematical update,
uniform-positive initialization distribution, sequential ordering, default
rank and angular stopping criterion, and saves the complete factor.

Matlab and NumPy RNG streams differ: identical integer seeds do NOT
reproduce the same initial factor, objective trajectory, or final vectors.
Validation against the official MATLAB implementation is still required
before claiming numerical reproduction of a specific table entry.

## Mathematical distinction

The dense convex SDP uses a symmetric n-by-n PSD Gram matrix X.
EVA instead optimizes a factor Y of shape n-by-p directly:

```text
max    F(Y) = (1/2) sum_{i<j} w_ij (1 - y_i dot y_j)
s.t.   ||y_i|| = 1 for each i
```

A sequential EVA update is

```text
h_i = sum_j w_ij y_j
y_i <- -h_i / ||h_i||   (when h_i != 0)
```

Each sweep visits all vertices; stopping is based on the maximum angular
change per vertex. The paper/official MATLAB default is
`p = ceil(sqrt(2*(n+1)))` and angle tolerance 0.001 radians.

For G77 (n=14000), p=168 and a float64 factor occupies about 18 MiB,
versus ~1.46 GiB for *one* dense float64 n-by-n matrix.
This reduces memory, but **the factor optimization is nonconvex**.
The theoretical discussion in the paper is not a numerical certificate
of global SDP optimality for an arbitrary stopped run. The code measures
a continuous primal objective, NOT an independently verified dual bound.

This implementation supports Max-Cut (`k=2`) only; it does not impose
the extra off-diagonal constraint for general Max-k-Cut.

## Windows PowerShell: install and verify

From a checkout of `experiment002-eva-factor-reproduction`:

```powershell
conda activate maxkcut
python -m pip install -e .
python -m pytest tests/test_eva.py -q
python -m pytest -q
```

First, validate on G11:

```powershell
maxkcut-eva --graphs G11 --max-n 1000 --max-sweeps 10000 --tolerance 0.001 --restarts 20 --output-dir results/gset/experiment002-eva --convex-dir reports/gset/full-sdp-batch --push --push-branch experiment002-eva-factor-reproduction --push-every 250
```

Then, only after inspecting the small-graph results, run G77:

```powershell
maxkcut-eva --graphs G77 --max-n 14000 --max-sweeps 10000 --tolerance 0.001 --restarts 20 --output-dir results/gset/experiment002-eva --convex-dir reports/gset/full-sdp-threads32 --push --push-branch experiment002-eva-factor-reproduction --push-every 250
```

**Run EVA and the existing full-SDP batch sequentially**, not concurrently,
when comparing runtime, to avoid confounding CPU contention. The SCS
baseline may fail on large graphs; that must appear as *unavailable*
rather than a manufactured score.

`--push` and `--push-branch` are BOTH explicit opt-ins. `--push-every`
is a multiple of `--checkpoint-every` (default 25), so the factor and
sweep logs are pushed during longer runs. Changes stay on the experiment
branch; no force-push or automatic merge. If Git authentication is absent,
the factor remains in the tracked local `results/` tree, but cannot be
assumed to be backed up remotely.

## Saved artifacts (tracked results tree)

- `factors/Gxx.npz`: full row-unit vector factor, rank, objective, sweep
  count, stopping condition, RNG state and graph/settings fingerprint;
  checkpoints support resuming optimization.
- `logs/Gxx-sweeps.jsonl`: exact factor objective, maximum angle and
  cumulative runtime for **every completed sweep**.
- `cases/Gxx.json`: continuous vector objective and separate best/mean/
  median/std of each rounding method; comparisons to convex SCS only if
  the exact input SHA256 matches.
- `trials/Gxx.jsonl`: **all 20 cut scores per method** and their archive IDs.
- `archives/<run-id>/Gxx-k2/{graph,FJ_EVA,K-Means2N_EVA,K-MeansNM_EVA}.npz`:
  normalized input graph and every bit-packed partition, separately
  checksum-verified by `maxkcut.archive.verify_archive`.
- `archives/<run-id>/Gxx-k2/manifest.json`: source Git revision, input
  hashes, versions, settings and solver status.

The factor is intentionally retained in compact form **instead of**
materializing `Y@Y.T` and storing a redundant huge Gram matrix.
If needed, its Gram entries can be computed on demand.

## Comparison protocol

On precisely matching graph SHA256, report EVA versus SCS continuous
primal values (not guaranteed upper bounds), angular and SCS residual
convergence, runtime (different CPU/thread settings disclosed), and
separately 20×3 signed weighted cut scores.

EVA and convex SCS optimize two *different representations* of the same
Max-Cut SDP objective. They do NOT share the same Gram factor in
general. Thus any cut differences conflate continuous-solver accuracy
with rounding. A clean rounding-method-only comparison must apply
every rounder to the **same** archived factor.

For lower rank `--rank 32`, report a separate factor-restricted
nonconvex experiment, not an exact full convex SDP.

## Current status

Implementation and synthetic tests prepared on a draft PR.
No MATLAB parity run, no G11/G77 EVA numerical result, and no large
benchmark verification on the user's Windows VM yet.


## All 71 canonical G-set graphs

To run every canonical \`data/G*.txt\` instance (including G77 and G81),
use **\`--graphs ALL --max-n 20000\`**, plus per-graph error isolation.
This runs the experimental **EVA vector relaxation**, not 71 convex CVXPY/SCS
SDPs, and does not overwrite \`reports/gset/full-sdp-threads32/\`.

First recover the Python imports if Windows blocked editable reinstall
because a previous \`maxkcut-sdp-batch.exe\` remained open:

\`\`\`powershell
cd C:\Users\175199\maxkcut
conda activate maxkcut
git fetch origin
git switch experiment002-eva-factor-reproduction
git pull --ff-only
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"
python -m pytest tests/test_eva.py -q
\`\`\`

Then run the complete sweep, specifying **both** convex-SDP directories
in the priority order they should be consulted:

\`\`\`powershell
python -m maxkcut.eva_cli --graphs ALL --max-n 20000 --max-sweeps 10000 --tolerance 0.001 --restarts 20 --cpu-threads 1 --continue-on-error --convex-dir reports/gset/full-sdp-batch reports/gset/full-sdp-threads32 --output-dir results/gset/experiment002-eva --push --push-branch experiment002-eva-factor-reproduction --push-every 250
\`\`\`

- Every graph gets a persistent factor checkpoint, up to 60 complete
  bit-packed partitions, diagnostics and provenance, plus a fresh row
  in \`results/gset/experiment002-eva/summary.csv\`.
- The CSV records the EVA continuous objective, rank, iterations,
  best cut by each method, convex solver continuous objective (when
  archive-verified), convex best cuts and EVA–convex differences.
- The first *matching, complete and independently verifiable* convex
  reference in the listed directory order is selected—not whichever
  gives the most favorable comparison. G60/G61/G62 can be found in the
  second folder; earlier small-graph SDPs are in the first.
- Missing, interrupted, mismatched or invalid convex results receive
  a nonnumeric status, **never an invented objective/cut**.
- A rerun verifies factors, graph/settings and partition archives before
  skipping existing complete cases. Interrupted factors resume from
  their latest checkpoint; failed cases are recorded in \`errors/Gxx.json\`
  and the batch continues with \`--continue-on-error\`.
- \`--push-every 250\` uploads a checkpoint every 250 full sweeps and
  pushes each completed/failed case to the experiment branch. Git push
  requires working authentication. Closing the VM before the next push
  may still lose recent local sweeps.
- 20,000 vertices can still require substantial CPU time. Run the
  convex SDP and EVA batches **sequentially** to obtain interpretable
  runtime comparisons. The stated CPU-thread limits can differ; disclose
  hardware/solver budget in the report.
- Source format contains 71 canonical graphs; the EVA article uses a
  67-instance G-set variant. Report this scope difference.
- If a graph is already complete but comparisons were unavailable,
  re-running the same command refreshes comparisons **without solving**.
- If the target environment cannot install the \`maxkcut-eva\` console
  script, use \`python -m maxkcut.eva_cli\` as above.

Quick result inspection:

\`\`\`powershell
Import-Csv results/gset/experiment002-eva/summary.csv | Group-Object status | Format-Table Name,Count
Import-Csv results/gset/experiment002-eva/summary.csv | Where-Object {$_.convex_status -eq 'compared'} | Format-Table graph,eva_KMeansNM_best,convex_KMeansNM_best,best_cut_eva_minus_convex -AutoSize
\`\`\`
