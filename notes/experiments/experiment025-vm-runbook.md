**LATEST RESULT (2026-11-03):** The full local-only VM experiment has finished with `1,278/1,278` complete graph-method cells and `20,448/20,448` recorded seed checkpoints; `all_complete=true`. The older partial-coverage and launch instructions in this runbook are historical. Do not launch another full optimization; next work is independent original signed-cut partition audit, matched-seed comparison, verified off-VM raw archive and compact GitHub publication. [Full evidence and provenance](experiment025-full-local-computation-2026-11-03.md).

# Experiment025 — VM execution, storage and recovery (2026-11-02)

**UPDATE 2026-11-03:** This original runbook documents the Git-publishing mode. The user has since demonstrated a **local-only** VM run via `-PublishMode local` and compact preview, with the live/unmerged script-versus-remote distinction documented in [the canonical university VM operations guide](../../docs/university-vm-experiment025-operations.md). The last user-observed local coverage was 755/1,278 completed graph-method cells (59.08%) before a new full launch; see [G44 WinError recovery record](experiment025-windows-local-recovery-2026-11-03.md). Avoid conflicting Actions jobs and do not pull/reset/clean while an experiment is running. Prior Git/LFS-push advice below applies ONLY to explicit Git publication mode, not local-only execution.


This is an execution runbook for the original signed Max-Cut quality-first objective. Frozen prior EVA factors (Experiments021, 003, 002) are loaded read-only. Each of 18 methods receives 16 shared GW starts and each start is independently refined by Flip, KL and ILS. The winner is the maximum VERIFIED cut visited at ANY meaningful iteration, even if a subsequent iteration is worse. Runtime is metadata, not a ranking criterion.

## Four execution issues and their resolution

1. **Outdated queued revision.** The old [run #41426842790](https://github.com/heri-espino/maxkcut/actions/runs/41426842790) is confirmed **cancelled**. The current workflow is manual-only and explicitly checks out main. The VM driver creates a separate experiment025-results branch from origin/main, rather than running the old experiment025-gw16-bestcut-quality branch. Every seed fingerprint includes a deterministic SHA256 of all optimizer Python files, so resumed runs refuse to mix code revisions. Git result-only commits change HEAD without changing the source hash.

2. **GitHub six-hour limit.** The preferred runner is the **persistent Windows VM running PowerShell directly**, not a GitHub Actions job. No method-level or total runtime cutoff is imposed by the script. The optional GitHub Actions workflow, manually dispatched from main, runs **one method** rather than 18 methods per graph; its job still has a platform timeout and is not the recommended path for the largest graphs.

3. **Thousands of tiny commits and recoverability.** The driver defaults to 16 new starts per method per invocation, with one checkpoint commit/push per complete graph-method (maximum 1,278 full graph-method commits, instead of 5,112 four-seed commits). For risky VM sessions, specify -ChunkSize 4 to push after each 4 completed starts. Every iteration event and every strict-best binary partition is saved to the persistent local disk immediately, but the newest events may not reach GitHub before a mid-seed crash. Re-running resumes validated completed seed archives; it does not overwrite prior results.

4. **Large storage.** Git LFS applies ONLY to Experiment025 .npz binary partitions and .jsonl progress streams, via a scoped root .gitattributes. The VM driver checks that git lfs is installed and these rules are active before computing. All other historical data/factors are unchanged. The repository owner's Git LFS storage and bandwidth allowances are finite; check them before a full sweep. Push failures fail closed without claiming archived remote success.

## VM commands (built-in Windows PowerShell 5.1; from the repository root)

The existing maxkcut Conda environment, Git for Windows, Git LFS and Git push credentials must already be configured. Do not execute a second competing writer on the same working directory or experiment025-results branch. Experiment023 may use the separate GitHub runner concurrently, but avoid competing for physical CPU resources.

First update the code and Git LFS locally:

    git switch main
    git pull --ff-only origin main
    git lfs install

Run the pilot, all 18 methods across three short-setting test graphs (G1 random, G14 planar-family, G11 toroidal):

    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase pilot

Inspect the local pilot coverage:

    Get-Content .\results\gset\experiment025-gw16-bestcut-quality\analysis\pilot_coverage.json

Run a full-quality trial on one graph, all methods, all 16 independently refined starts:

    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -Graph G11

Run the full 71-graph set (previously completed G11 seeds are verified and reused). This command does not impose a job time limit:

    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -AllGraphs -SkipTests

When all results are safely published, construct and audit the global completeness report:

    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase coverage -SkipTests

Optional: -Method gw-rr to select one of the 18 methods; -ChunkSize 4 for more frequent Git pushes (default 16). The driver handles Conda PowerShell activation, native DLL directory PATH entries, PYTHONPATH, Numba and BLAS environment variables. It tests dependencies and the archived 71-graph reference table before optimization. The source-tree SHA pin rejects mixing optimizer revisions.

## Data preservation and final publication

- Append-only iteration events are flushed after every meaningful algorithm iteration. Strict improvements get versioned SHA-verified binary snapshots. Round, Flip, KL and best-ever Final stage partitions are independently rescored on the original signed graph.
- Results are written to results/gset/experiment025-gw16-bestcut-quality/ on the **experiment025-results** branch, not main. No force pushes. A Git failure blocks further execution until the state is reconciled.
- Pilot settings use reduced inner iterations only for resource validation and cannot be mixed with the full-quality search.
- The full experiment comprises 71 graphs × 18 method protocols × 16 full seed refinements = 20,448 complete search trajectories. Missing or incomplete seeds remain unranked. Final observed best cuts are compared with GW-RR and archived best-known feasible references. These references are not all certified optima.
- Publish and merge a separate verified results PR into main **after** complete audit. Source files, configurations, statuses, graphs, all numeric histories and saved best partitions must remain accessible to a future independent AI agent.

## Optional GitHub Actions

The manual-only Experiment025 workflow runs one chosen method/graph with a selected 1–16 seed checkpoint size, pinned to a checkout of current main. It uses the same driver and experiment025-results branch. Its individual job remains subject to GitHub's 360-minute platform cap. Prefer the direct VM commands above for prolonged or interrupted graph-method runs.


## Audited execution update — 2026-11-03

The first successful [seed-first pilot run #41570037856](https://github.com/heri-espino/maxkcut/actions/runs/41570037856) completed G1/G14/G11 × 18 methods × **only seed 1**, 54/54 numerical jobs, with all stage partitions, intermediate scores, and strict-best binary snapshots pushed to experiment025-results. Every pilot method remains partially_refined (1/16); **the 71-graph full run has not occurred**. See [the detailed audit and handoff](experiment025-successful-pilot-parallel-vm.md).

For the preferred direct university Windows 11 + existing maxkcut Conda mode, ensure both operational scripts are present **on experiment025-results** and invoke:

    git fetch origin experiment025-results
    git switch experiment025-results
    git pull --ff-only origin experiment025-results
    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -AllGraphs -Workers 32

For lower RAM usage use -Workers 4 or 8; add -ChunkSize 4 to push after every four new seeds per method rather than the default 16. The driver runs independent graph subprocesses, never EVA solves; Git commits and LFS pushes are serialized in one parent process. The existing -Workers 1 serial path is preserved. Pilot has only three graphs and hence at most three simultaneous workers. Never let another VM/Actions writer publish into experiment025-results concurrently. Unsynchronized newest per-iteration events can be local-only until the next Git checkpoint.

**Branch caveat:** If [PR #66](https://github.com/heri-espino/maxkcut/pull/66) is not merged, DO NOT launch the stale version from main: PowerShell loads its parameters before switching to the results branch. The operational scripts have been installed directly on experiment025-results without changing src/maxkcut numerical source files or EVA factors. No 32-worker full numerical performance measurement has yet been made.
