# Follow-up interruption — G56, G54, G67 (2026-11-03, user-observed)

**Evidence scope:** The user shared the final console output of a subsequent full local-only VM run. The command exited after the scheduler reported **three** graph-method subprocess failures, with preserved local checkpoints. **Actual Python traceback bodies for these three failures were not supplied**, so the root causes are unknown. Do not assert they were WinError 5 or that the algorithms failed mathematically.

| Graph | Failed method | Log path (relative to repo) |
|---|---|---|
| G56 | `mst-l2-longest` | `results/gset/experiment025-gw16-bestcut-quality/full/G56/logs/mst-l2-longest-parallel.log` |
| G54 | `mst-l2-multicut` | `results/gset/experiment025-gw16-bestcut-quality/full/G54/logs/mst-l2-multicut-parallel.log` |
| G67 | `cutaware-kmeans` | `results/gset/experiment025-gw16-bestcut-quality/full/G67/logs/cutaware-kmeans-parallel.log` |

Each failed with process exit code 1. The parallel scheduler raised `Graph-parallel execution failed. Locally saved checkpoints remain available; inspect per-graph logs before resuming.` at `scripts/experiment025-vm.ps1:307`. Other workers had already saved successful chunks to the VM, e.g., G63/multigroup-k4 16/16, G64/hyperplane-sweep 16/16, and G57/em-gibbs 16/16, before returning `stopped`. **These messages are not evidence that complete earlier seed archives were discarded.**

**Next safe action:** The process has exited, but first verify no Python child or additional writer remains. Read the exact first invocation line and last 70–120 lines of each of the **three** log files; determine whether the code invoked `scripts/experiment025-windows-io-guard.py` or direct `-m maxkcut.experiment025_bestcut`. Check `status.json` and saved seed indices for each failed method, then recover only missing seeds **after diagnosing the exceptions**, and only when the respective graph has no owner. Recalculate local coverage after stopping writers. No new live VM access, output audit, or failure root-cause confirmation was possible in this GitHub documentation-only update.

**Branch caveat:** GitHub `experiment025-results` currently contains the guarded launcher but not the confirmed VM's `-PublishMode local` changes. [PR #70](https://github.com/heri-espino/maxkcut/pull/70) is still open at the time of review. Never pull or overwrite the VM's working local scripts or raw results to resolve this mismatch during computation.

---

# Experiment025 — Windows VM local-only interruption and recovery (2026-11-03)

## Provenance and status

This is a **user-observed VM terminal transcript** from November 3, 2026, not a GitHub-derived measure of live local state. The university Windows 11 machine uses built-in PowerShell 5.1, Conda environment maxkcut, repo root C:\Users\175199\maxkcut and Conda Python C:\Users\175199\AppData\Local\miniconda3\envs\maxkcut\python.exe. The same physical host is registered as GitHub runner PCAG103A-13.

At the end of the supplied transcript the user had **restarted** the full benchmark and was typing the full-run launcher. Its current process state and later numerical coverage are **not verified** by this note. Do not stop or change the live VM or the result branch to apply these docs.

## What failed

A prior 16-worker local-only full execution logged several successful checkpoints such as:

    LOCAL CHECKPOINT full G50 gw-rr: 16/16 (not uploaded)
    LOCAL CHECKPOINT full G44 multigroup-k4: 16/16 (not uploaded)
    LOCAL CHECKPOINT full G44 multigroup-k8: 0/16 (not uploaded)
    FAILED G44: G44/multigroup-k8 failed (1)

The G44 log lives at:

    results\gset\experiment025-gw16-bestcut-quality\full\G44\logs\multigroup-k8-parallel.log

The **actual invoked command** recorded in that failing log began:

    2026-11-03T17:45:44Z C:\Users\175199\AppData\Local\miniconda3\envs\maxkcut\python.exe -m maxkcut.experiment025_bestcut single --graph G44 --method multigroup-k8 --phase full --max-new-seeds 16

At the first GW seed (internal index 0), while optimizing the KL stage, the iteration observer attempted to write best.json atomically. The stack traversed:

    experiment025_bestcut.py:183 run_method
    experiment025_journal.py:151 refine_with_journal
    eva_elite_search.py:170 optimize_kl_fast
    experiment025_journal.py:98 observe
    experiment023_methods.py:79 atomic_json

The exception was:

    PermissionError: [WinError 5] Access is denied:
    ...\G44\multigroup-k8\progress\0000\best.json.tmp
    -> ...\G44\multigroup-k8\progress\0000\best.json

Exit code 1 was thus a **Windows checkpoint file replacement failure**, *not* demonstrated mathematical/numerical failure of multigroup-k8. The precise OS process holding the file or denying replacement was not identified; **antivirus/indexing interference remains a hypothesis, not a verified root cause**.

After the first failure, the parallel scheduler set a global stop event. Already active workers finished their current method chunks and saved locally; queued graphs reported status=stopped. Do **not** interpret a stopped result as wiping previous archives. No complete graph-method became ineligible merely because the global scheduler stopped.

## Recovery, verified success

The user examined the log with:

    Get-Content ".\results\gset\experiment025-gw16-bestcut-quality\full\G44\logs\multigroup-k8-parallel.log" -Tail 120

Then used **isolated, guarded** PowerShell execution:

    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -OnlyGraph -Graph G44 -Method multigroup-k8 -Seed 1 -Workers 1 -PublishMode local

The VM preflight reported:

- PowerShell version 5.1.26100.7462;
- correct Conda environment;
- NumPy 2.5.4 and SciPy 1.18.1;
- scientific source SHA256 99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad;
- **31 tests passed in 2.52 seconds**.

The seed then finished with:

    graph=G44, method=multigroup-k8, phase=full
    fingerprint=dc109a684e1e6ec0646b0df2
    status=partially_refined
    starts_completed=1 / 16
    completed_seed_indices=[0]
    best_initial=6521.0
    best_seen=best_final=6588.0
    best_last_ils_candidate=6588.0
    wall_seconds_this_invocation=1.5470000002533197

PowerShell printed:

    LOCAL CHECKPOINT: experiment025: full G44 multigroup-k8 matched GW seed 1 (no Git add, commit or push)
    Experiment025 full completed locally. Raw checkpoints remain ONLY on this VM until backed up; none were uploaded.

**Scope:** The isolated G44 *seed* recovered, **not** all 16 G44 seeds, and **not** the complete Experiment025 campaign. The improvement from raw to best saved cut was 6588 − 6521 = 67 signed cut-weight units.

## Provisional aggregate status observed afterward

The user invoked:

    $env:PYTHONPATH = (Resolve-Path .\src).Path
    python -m maxkcut.experiment025_bestcut summary --phase full
    python .\scripts\experiment025-summary-preview.py --phase full

The first command printed:

    planned_cells: 1278
    completed_cells: 755
    planned_independent_full_refinements: 20448
    all_complete: false

Thus at that **local snapshot only**, 755/1278 = 59.08% of graph-method combinations were fully complete, with at least 755×16 = 12,080 complete seed trajectories and 523 incomplete graph-method cells. Some of the 523 may contain 1–15 completed seeds, so 523×16 is **not** necessarily the true number of remaining trajectories. Report path:

    results\gset\experiment025-gw16-bestcut-quality\analysis\full_preview.md

This snapshot **predates later restarted computation** and cannot be used as a continuously live statistic. Because local mode deliberately suppresses Git/LFS uploads, GitHub remote may be far behind.

## Why another agent must inspect the exact launcher revision

The G44 *failing* log invoked the unguarded Python module directly. A newer remote results-branch scheduler examined on 2026-11-03 invokes:

    scripts/experiment025-windows-io-guard.py single ...

That guard retries WinError 5 and 32 for approved JSON/NPZ atomic replacements within the Experiment025 result tree (10 attempts, bounded backoff); it does not edit optimizer source. The successful isolated PowerShell invocation used this guarded launcher. The locally running parallel scheduler may have loaded yet another revision. **Never claim a running process is guarded merely because GitHub shows a guarded script.** Confirm its invocation in the log's first line.

Similarly, the VM has successfully used -PublishMode local and the preview generator, but the remote experiment025-results branch did not yet have this optional mode at the time of inspection. See local-only implementation [PR #70](https://github.com/heri-espino/maxkcut/pull/70). Do not pull/checkout/change an active worktree to close this difference.

## Safe handoff to a future AI

1. First read AGENTS.md, RESEARCH_HANDOFF.md and [the canonical operations guide](../../docs/university-vm-experiment025-operations.md).
2. Determine whether the VM scheduler is still active **from the user or actual VM process list**, not by GitHub commits.
3. While active, do nothing that mutates its scripts, branch, outputs, lock or filesystem.
4. After it stops, preserve all checkpoints and logs; inspect any errors; recover only missing seeds on an idle graph, then resume the full local-only run using the previously tested source hash.
5. Generate a fresh local full coverage snapshot only when writers have stopped. Validate independent partitions, backups and seed pairing before interpreting method rankings.
6. Preserve a verified off-VM backup; a single university VM is not durable storage. Publish only compact summary reports, figures and reproducibility documentation from a **clean separate publication branch**.

The historical successful 54-job seed-1 Actions pilot and its runtime overhead are documented at [the pilot/runner audit](experiment025-successful-pilot-parallel-vm.md). The experiment's mathematical contract is in [Experiment025 protocol](experiment025-gw16-bestcut-quality.md). No EVA/SDP solution was recomputed during this recovery.
