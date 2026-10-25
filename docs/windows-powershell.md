# Windows PowerShell: run your first G-set experiment

This page uses **PowerShell**, not Bash. The `maxkcut` CLI is the same on Windows, macOS and Linux; **only the shell's line-continuation syntax changes**.

## Start in the repository

```powershell
cd .\maxkcut
git pull origin main
conda activate maxkcut
python -m pytest -q
maxkcut --list-solvers
```

If you have already activated the environment or are already in the project directory, skip those setup commands. Installing fresh uses `conda env create -f environment.yml`; updating an existing Conda environment uses `conda env update -f environment.yml --prune`.

## Graph inspection (does not solve an SDP)

```powershell
maxkcut --graphs G11 --inspect-only
```

Review its reported `n`, number of edges, and edge-weight range before running a solver.

## First G11 Max-Cut SDP

**Copy the entire line**:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --restarts 20 --seed 2026 --output reports/gset/g11-first.jsonl
```

Alternatively, the same command may be written on multiple lines using PowerShell's **backtick**, which must be the very last character on each continued line:

```powershell
maxkcut --graphs G11 --k 2 `
  --solver SCS `
  --eps 1e-3 `
  --max-iters 3000 `
  --restarts 20 `
  --seed 2026 `
  --output reports/gset/g11-first.jsonl
```

**Do not copy the Bash trailing backslash as a line continuation in PowerShell.** It will be passed to `maxkcut` as a literal argument (e.g. `invalid int value: '\'`), and subsequent lines beginning with `--` will cause PowerShell syntax errors.

## Change rounding seeds without re-solving the SDP

If the first SDP solve completes and its matrix is safely cached, reuse it with another random seed and more restarts:

```powershell
maxkcut --graphs G11 --k 2 --restarts 100 --seed 2027 --output reports/gset/g11-more.jsonl
```

The cache is keyed by graph file hash and solver settings. Do not reuse the same output JSONL filename when rerunning, because the experiment runner **appends** records.

To compare SCS's indirect linear system solver, run a separate experiment:

```powershell
maxkcut --graphs G11 --k 2 --scs-indirect --output reports/gset/g11-indirect.jsonl
```

This is a new, separately fingerprinted SDP solve. It is not necessarily faster than the direct solver.

## View completed results

```powershell
Get-Content reports/gset/g11-first.jsonl
```

Each line is one JSON record for one rounding restart. Fields include cut weight, solver status, factorization time, the original SDP solve time, and whether the cached Gram matrix was reused.

## Tests, warnings and solver status

- `14 passed` means the small tests passed; **it does not mean G11 was solved**.
- SciPy's old `loadmat` default sparse type and `sparse.diags` dtype warnings are compatibility warnings, not test errors. Newer code chooses a compatible sparse-array format explicitly where supported and explicitly sets the Laplacian diagonal dtype.
- A returned `optimal_inaccurate` status is an approximate numerical result, **not** a certified integer optimum or validated dual SDP upper bound.
- G11 requires an 800×800 dense PSD matrix; the solver can consume much more memory than that matrix itself.
- If you want solver iteration output, add `--verbose`, but expect substantial terminal logging.

See the [performance guide](performance.md) for solver strategies and the [G-set reproduction runbook](https://github.com/heri-espino/maxkcut/blob/main/notes/experiments/gset-cvxpy-reproduction.md) for scientific interpretation.

## If SCS reports a slightly negative SDP eigenvalue

An SDP solver can return a near-PSD *numerical* Gram matrix even when it reports
`optimal` or `optimal_inaccurate`. A real G11/SCS run reported
`lambda_min = -0.000837374814667048` at `--eps 1e-3`, after which the
previous `factorize` routine aborted before any rounding took place.

The library now clips *small* negative eigenvalues to zero and normalizes
the resulting vectors. This gives well-defined vectors for randomized rounding;
it **does not** certify the raw SDP as feasible, guarantee any approximation
ratio, or validate its primal objective as an upper bound.

Every successful record includes original `min_eigenvalue`, the relative
negative violation, how many eigenvalues were clipped, the clipping sum and
`psd_projection_applied`. The default rejection threshold is `1e-3`
relative to the larger of 1 and the maximum eigenvalue.

Once you have updated `main`, this command can reuse the previously
cached *raw* approximate Gram matrix without another lengthy SDP solve:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --restarts 20 --seed 2026 --output reports/gset/g11-after-repair.jsonl
```

The original matrix is retained in the cache for independent verification; the
postprocessed Gram representation is used only for rounding. If its defect
exceeds the threshold, the command still raises an error. To rerun **SCS**
at tighter accuracy rather than accepting that approximation, use a fresh solve:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-5 --max-iters 20000 --no-sdp-cache --restarts 20 --seed 2026 --output reports/gset/g11-stricter.jsonl
```

A tighter tolerance generally requires more computation and does not guarantee
convergence. You can opt into a strict PSD rejection check using
`--psd-repair-rel-tol 0`, but even roundoff-level negativity can then stop
the run. For (k>2), projection onto PSD and unit-vector rescaling **may not**
preserve the Frieze–Jerrum pairwise lower bounds, so the repaired Gram matrix
must not be used as an SDP feasibility certificate.

## Workstation CPU/GPU options

A machine with 32 logical processors and a GPU will **not** automatically
use them all during SCS's PSD cone projection. Check which BLAS backend is
linked, then try a controlled 8-thread solve:

```powershell
maxkcut --hardware-info
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-5 --max-iters 20000 --no-sdp-cache --cpu-threads 8 --restarts 20 --output reports/gset/g11-cpu8.jsonl
```

If you have an NVIDIA GPU, use `nvidia-smi` to identify it. The separate
`--scs-cudss` flag only works if your SCS Python package is compiled with
cuDSS; normal Conda SCS installations do not guarantee this. It is **not**
necessarily faster for semidefinite programs. See
[CPU/GPU performance research](gpu-cpu-performance.md).

## SDPA failed with `Memory Exhausted (bad_alloc)` despite 64 GB RAM

This is a failure of native SDPA memory allocation, **not** evidence that G11
cannot fit into 64 GB. SDPA's native worker count defaults to the number of
logical processors. Our `--cpu-threads N` now controls SDPA's documented
`numThreads` option, not only external BLAS threads.

Check Windows committed memory (read-only):

```powershell
Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' | Select-Object -ExpandProperty CounterSamples | Select-Object Path,@{Name='GiB';Expression={[math]::Round($_.CookedValue/1GB,1)}}
```

For an **optional one-time** smaller-thread experiment:

```powershell
maxkcut --graphs G11 --k 2 --solver SDPA --cpu-threads 4 --no-sdp-cache --verbose --restarts 20 --output reports/gset/g11-sdpa-4threads.jsonl
```

If it still reports `bad_alloc`, stop retrying the same setup. Your SCS
run already succeeded, and a backend-specific failure does not invalidate
the rounded cut it produced. See [CPU/GPU performance](gpu-cpu-performance.md).

## Archive and independently verify every vertex partition

Running the benchmark now creates a unique archive of the normalized graph,
weighted edges, all rounded partitions, solver/provenance metadata and
per-method statistics **without any additional SDP solve**:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --restarts 20 --seed 2026 --output reports/gset/g11-archive.jsonl
```

Use the case archive path printed at the end of the run:

```powershell
maxkcut --verify-archive "reports/gset/artifacts/YOUR-RUN-ID/G11-k2"
```

The command recomputes all cut scores from the saved signed edge list and
vertex assignments; it does **not** rerun the SDP. See
[experiment archiving](experiment-archives.md) and
[certified G-set optima](gset-reference-values.md).

## Run every checked-in G-set graph with the safe sparse baseline

The 71 checked-in graphs are G1 through G71. The dedicated
`maxkcut-sweep` command uses sparse signed-weight one-vertex-flip local
search, **not** the expensive full SDP/GW/K-Means methods:

```powershell
git pull origin main
python -m pip install -e .
python -m pytest -q
maxkcut-sweep --inspect-only
maxkcut-sweep --max-graphs 2 --output-dir reports/gset/full-first-pass
maxkcut-sweep --output-dir reports/gset/full-first-pass
```

Re-run the last command after an interruption to resume. View
`reports/gset/full-first-pass/summary.csv` with PowerShell `Import-Csv`.
The first-pass budget is **5 restarts, 20 sweeps, soft 60 seconds/graph**.
See [first-pass sweep methodology](full-gset-sweep.md).
