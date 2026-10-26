# Publish full convex SDP results to GitHub

The `maxkcut-sdp-batch` solver stores its numerical output under ignored
`reports/`, so those data never appear in GitHub automatically.
Supervisors who only use GitHub can read **tracked result snapshots**
under `results/gset/full-sdp/`.

**No numerical SDP solves or CUDA optimization are performed when publishing.**
The publisher can run in a **second Windows PowerShell terminal** while
another terminal computes G55, G56, etc. It reads atomically finalized
per-graph `cases/Gxx.json` files and independently validates completed
signed weighted partition archives, graph SHA256 and full rounding count
before reporting a cut as verified. A `started` entry does not count as
complete, and an interrupted large SDP remains unpublished until finished.

## One-time publication

After installing the latest code and tests, run from the repository root:

```powershell
conda activate maxkcut-cuda
python -m pip install -e .
python -m pytest -q
maxkcut-sdp-publish --output-dir reports/gset/full-sdp-batch --push
```

The first run generates a human-readable `README.md`, `summary.csv`,
`method-summary.csv`, `rounding-trials.csv`, `provenance.csv`,
and an optional graph-verified `sdp-vs-cuda.csv`. It then commits and
pushes only that curated folder to the **main** branch. A repeated run
with no content changes creates no new commit.

Git must already be authenticated (via `gh auth login` and
`gh auth setup-git`, if necessary), and the checkout must be on
`main`. If another contributor updated main, the non-force push may
be rejected; reconcile changes normally and never force-push.

## Automatic publication as the batch makes progress

In a **second, persistent** PowerShell terminal, run:

```powershell
cd C:\Users\175199\maxkcut
conda activate maxkcut-cuda
maxkcut-sdp-publish --watch-seconds 900 --push --output-dir reports/gset/full-sdp-batch
```

This checks every 15 minutes, exports only completed validated SDP
cases and pushes a new snapshot **only if contents have changed**.
Your running SDP batch continues separately. Ctrl+C here stops only
the **publication watcher**. Closing the watcher terminal stops GitHub
updates but does not stop the solver. Reopen with the same command.

The publisher **does not** invoke `git pull`, modify or kill numerical
solver subprocesses, modify the ignored `reports/` source, nor
perform a force push. If a push fails because the remote advanced,
the published commit remains local and an explicit user reconciliation
is necessary.

For a lighter offline review without any Git action:

```powershell
maxkcut-sdp-publish --output-dir reports/gset/full-sdp-batch
```

## What's published

| Tracked file | Purpose |
| --- | --- |
| `README.md` | GitHub landing page: verified successes, skipped/started counts and caveats |
| `summary.csv` | One graph per row, case status, best actual weighted cut, method, wall time, SHA256 |
| `method-summary.csv` | GW/FJ, K-Means2N, K-MeansNM cuts: best, worst, mean, median, std and restarts |
| `rounding-trials.csv` | Every per-method rounded cut, restart, numerical status and timing |
| `provenance.csv` | Exact graph SHA, archived run identifier, original Git revision, CPU threads, seed, solver/version and iteration diagnostics |
| `sdp-vs-cuda.csv` | Optional identical-graph-verified full convex SDP vs nonconvex CUDA comparison, including the selected baseline run provenance |

**Not published:** full `n\times n` raw SDP matrices (up to multi-GB),
the solver cache, local log files, Windows usernames/absolute paths,
unverified incomplete cuts or arbitrary files under `reports/`.
The original compressed binary graph and partition archives are kept
locally and remain available for further research/auditing. CSV trial
data permit method distribution and relative-performance analysis,
but are *not* replacements for archived partitions when an independent
cut recomputation is required.

The original SCS `optimal` is numerical continuous-relaxation
termination, **not** a proof of optimal integer Max-Cut. The primal
SDP objective is not automatically a certified upper bound.

## If GitHub doesn't update

Run `git status`, `git branch --show-current`, `git remote -v`,
and `gh auth status`. The publisher deliberately requires branch
`main` when pushing, stages only the small, known publication files,
and never uses force pushes. If a push is rejected, do not delete the
original `reports/` experiment archive or discard the local results.
