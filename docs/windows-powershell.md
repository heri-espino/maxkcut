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

The complete 71 canonical text graphs are in `data/`: G1–G67, G70, G72, G77, G81. The older MATLAB graphs remain in `G-Set/`. The dedicated
`maxkcut-sweep` command uses sparse signed-weight one-vertex-flip local
search, **not** the expensive full SDP/GW/K-Means methods:

```powershell
git pull origin main
python -m pip install -e .
python -m pytest -q
maxkcut-sweep --inspect-only
maxkcut-sweep --compare-inputs
maxkcut-sweep --max-graphs 2 --output-dir reports/gset/full-first-pass
maxkcut-sweep --output-dir reports/gset/full-first-pass
```

Re-run the last command after an interruption to resume. View
`reports/gset/full-first-pass/summary.csv` with PowerShell `Import-Csv`.
The first-pass budget is **5 restarts, 20 sweeps, soft 60 seconds/graph**.
See [first-pass sweep methodology](full-gset-sweep.md).

## Compare the canonical data against the paper and older graphs

```powershell
maxkcut-sweep --compare-inputs
maxkcut-sweep --compare-only
Import-Csv reports/gset/full-first-pass/comparison.csv | Where-Object {$_.paper_cut -ne ''} | Format-Table graph,our_best_cut,paper_cut,paper_upper_bound,reference_status,assessment
```

Use `--gset-dir G-Set` to explicitly reproduce previous runs based on
legacy `.mat` files. Without the flag, the CLIs now load `data/*.txt`.
The 2026 paper lists 17 toroidal instances, not 71 proven optima.
See [canonical graph and published-cut comparison](canonical-data-comparison.md).

## October 2026: all graph files appeared missing

An earlier `discover_graphs` implementation accidentally used a
**double-escaped regular-expression pattern**. Consequently, valid files
named `G11.txt` or `G11.mat` were silently excluded, causing eight
unit tests to fail and making `maxkcut-sweep` incorrectly report
`No G-set .txt or .mat files in data`. The error did **not** establish
that the local dataset was absent. This is fixed in PR #12.

Copy **only commands** into PowerShell, not descriptive sentences such
as "Check whether..." or "Run two instances...".

```powershell
git pull --ff-only origin main
python -m pytest -q
Get-ChildItem .\data -Filter 'G*.txt' | Measure-Object
maxkcut-sweep --inspect-only
maxkcut-sweep --compare-inputs
maxkcut-sweep --max-graphs 2 --output-dir reports/gset/canonical-first-pass
maxkcut-sweep --output-dir reports/gset/canonical-first-pass
```

There should be **71 canonical G-set TXT files** in the checked-out
`data/` directory. If the `Get-ChildItem` command returns a missing
directory error or a count of zero **after pulling**, inspect Git state
without destroying local changes:

```powershell
git status --short --branch
git ls-files data
git ls-tree -r --name-only HEAD data
```

A failed `--compare-inputs` invocation will not create
`input_comparison.csv`; an immediate `Import-Csv` of that missing
output will then also fail. Only inspect the CSV after the generating
command completes successfully.

## October 2026: test sweep succeeded internally but printed `KeyError: 'complete'`

A second bookkeeping bug caused a successfully completed graph to be
misreported as failed. The run manifest uses `status="complete"`, whereas
the batch summary counter is intentionally called `"completed"`. The
old code did `counts[status] += 1` and raised `KeyError: 'complete'`
after writing the partition archive. The exception handler then
incorrectly marked the case failed. Fixed in PR #13.

After updating, first rerun the **lightweight test suite**; the user's
previous run was **54 passed and 4 failed**, all four due to this issue:

```powershell
git pull --ff-only origin main
python -m pytest -q
```

Your local `data/` already contained **71** `G*.txt` files. A fresh
canonical first pass can now process two graphs, then resume for all 71:

```powershell
maxkcut-sweep --max-graphs 2 --output-dir reports/gset/canonical-first-pass
maxkcut-sweep --output-dir reports/gset/canonical-first-pass
```

Do not delete previous archives automatically. The runner saves each
run separately; when a previous record was marked `failed`, it will
recompute that graph under the same plan. A successful `summary.csv`
and `comparison.csv` appear in the selected output directory. The
latter compares only the 17 toroidal instances documented in
Glass–Feder (2026), not all 71.

## Use NVIDIA CUDA without WSL or administrator privileges

For your RTX 4500 Ada with an NVIDIA driver reporting CUDA 13.2, we added
**an optional PyTorch GPU low-rank optimizer**. It is a *different
nonconvex Max-Cut solver* (k=2 only) that still runs our established
GW/K-Means rounders on the resulting vectors. It is **not** the original
full-rank SDP solved on a GPU.

From the repository root, create a separate CUDA Conda environment and
install the official prebuilt Windows CUDA PyTorch wheel:

```powershell
conda env create -f environment.yml -n maxkcut-cuda
conda activate maxkcut-cuda
python -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu132
python -m pytest -q
maxkcut-gpu --device-info
maxkcut-gpu --graphs G11 --device cuda --rank 32 --epochs 200 --optimizer-starts 2 --restarts 20 --seed 2026 --output reports/gset/g11-torch-cuda.jsonl
```

The CUDA check refuses silent CPU fallback. To change just random rounding
seeds and **reuse the saved factor**, keep `--seed 2026` and add
`--rounding-seed 2027`. See [full CUDA guide](cuda-lowrank.md).

## Windows CUDA verification: G11 ran; PyTorch DLL failure inside pytest

On 2026-10-24, the user's `maxkcut-cuda` environment installed
**torch 2.13.0+cu132**. The separate `maxkcut-gpu --device-info`
command successfully reported `cuda_available: true` and
`NVIDIA RTX 4500 Ada Generation`. The actual G11 GPU low-rank smoke
experiment also **completed** (rank 32, 100 epochs, one optimizer start,
10 rounding starts): GW=442, K-Means2N=506, K-MeansNM=502. These
measurements are **not the full SDP** and should not be conflated with
the previous full-SDP G11 best cut of 546.

The full test suite ran **71 passed, one failed**: `import torch` late
in the in-process pytest session raised WinError 127 for `torch/lib/shm.dll`
(or one of its dependencies), while a later fresh CLI process imported
and used CUDA successfully. This suggests a native DLL load-order or
environment conflict **in that process**, not proof of a broken GPU.

The optional CPU PyTorch numerical test now executes in a clean Python
subprocess, isolating DLL initialization from the other tests. It still
reports an explicit failure with captured stderr if that subprocess
cannot import or run PyTorch. This is test isolation, **not** a repair
of a genuinely missing native runtime.

```powershell
git pull --ff-only origin main
conda activate maxkcut-cuda
python -m pytest -q
maxkcut-gpu --device-info
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `python -c "import torch"` fails in a fresh process with WinError
127, inspect conflicting DLLs and the Microsoft Visual C++ Redistributable
(official [PyTorch Windows FAQ](https://docs.pytorch.org/docs/stable/notes/windows.html)).
Do not assume that re-running pytest or suppressing errors repairs
PyTorch's native installation.

For better G11 cut quality, tune ranks, optimizer epochs, starts and
learning rates in **new, separately archived** experiments. The first
CUDA result is 506/564 (89.72% of the proven G11 optimum), below the
earlier full-SDP K-Means result 546/564 (96.81%). These runs have
different budgets; no speedup or statistical superiority has yet been
measured.

## Full convex SDP versus the existing CUDA low-rank experiment

The `maxkcut` command uses the actual CVXPY **convex SDP**, followed by
the three existing GW/K-Means clustering methods. It does not use the
CUDA low-rank solver. You can directly contrast saved experiments using
`maxkcut-sdp-compare`, which checks normalized graph snapshots against
canonical `data/Gxx.txt` and refuses unverified quality deltas:

```powershell
git pull --ff-only origin main
conda activate maxkcut-cuda
python -m pip install -e .
python -m pytest -q
maxkcut-sdp-compare --preflight-seven
maxkcut --graphs G11 --gset-dir data --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --cpu-threads 8 --restarts 20 --seed 2026 --output reports/gset/g11-fullsdp.jsonl
maxkcut-sdp-compare --sdp reports/gset/g11-fullsdp.jsonl --cuda reports/gset/g11-cuda-strong.jsonl --output reports/gset/g11-sdp-vs-cuda.csv
Import-Csv reports/gset/g11-sdp-vs-cuda.csv | Format-Table graph,graph_identity,sdp_best_cut,cuda_best_cut,sdp_minus_cuda_cut,sdp_status
```

Only attempt n>1000 **full SDP** experiments with a bounded resource
plan. The new `--preflight-seven` only inspects text headers and does not
run any SDP. It reports the memory needed for ONE dense X matrix, **not**
the full peak CVXPY solver RAM. See
[the full SDP study protocol](full-sdp-comparison.md).
