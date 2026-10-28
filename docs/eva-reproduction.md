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
