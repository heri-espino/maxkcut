> **Finding the frozen EVA solutions (2026-11-03):** They are not stored together under Experiment021. Refer FIRST to [the exact 71-factor EVA source map and read-only VM verifier](eva-factor-locations-vm.md): G1–G59 in Experiment002, G60–G62 in Experiment021, G63–G67/G70/G72/G77/G81 in Experiment003; always validate against Experiment021's 71-record manifest. The interactive repo root differs from the historical GitHub Actions worktree. Never solve EVA or SDP again merely because a relative file path fails.

> **STATUS UPDATE — COMPLETED COMPUTATIONAL SWEEP (2026-11-03):** The user-provided scheduler output and a separately rerun aggregate both confirm **1,278/1,278 graph-method cells complete, 20,448/20,448 recorded seed checkpoints, 71/71 fully covered graphs; `all_complete=true`**. This *supersedes* the 755/1,278 snapshot and "restarted/live" statements later in this runbook, which remain dated historical context. All raw iterations remain VM-local; independently verified original graph rescoring, off-VM backup, paired-seed statistics and compact GitHub publication are **not yet done**. Do not rerun the entire numerical campaign. See [latest evidence and next-agent plan](../notes/experiments/experiment025-full-local-computation-2026-11-03.md).

# University Windows VM: Experiment025 operations and agent handoff

**Observed on November 3, 2026.** The user reported restarting the full local-only experiment on the university Windows VM. Live process state is **unknown from GitHub**. An agent must never pull, switch branches, run another experiment, reset, clean, change scripts or merge into the active results worktree.

## Quick decision table

| Need | Preferred execution | Notes |
|---|---|---|
| Full 71-graph / 18-method / 16-seed run, resume, single-seed recovery | **User runs PowerShell in the university VM** | Persistent local results; optional local-only mode skips per-chunk Git/LFS uploads |
| Inspect local progress and errors | **User runs VM commands** | Raw VM results can be ahead of GitHub |
| Aggregate CSV/JSON and Markdown preview | **User runs VM commands after writers stop** | No EVA solve; provisional until audited |
| Python tests, docs, code-quality checks | **GitHub hosted Actions / pytest** | Do not use the university machine for simple CI |
| Windows runner PowerShell/Conda preflight | **Manual GitHub Actions only when the direct VM experiment is idle** | Tests Actions runner configuration |
| Seed-round or graph/method numerical Actions job | **Actions fallback only when VM is idle** | Same PC, 360-minute per-job limit, heavy serial Git publishing |
| Publish compact report/figures and documentation | **Separate clean GitHub branch + PR** | Avoid pushing raw local iteration archives |
| Solve EVA/SDP matrices | **Never in Experiment025** | Previously solved immutable EVA factors are reused |

**GitHub Actions and direct commands can occupy the same university computer.** Never run Experiment025 workflows and direct terminal experiments at the same time.

## 1. Verified university machine and paths

| Setting | Value / qualification |
|---|---|
| Repository | https://github.com/heri-espino/maxkcut |
| Registered runner name | PCAG103A-13 |
| Actual runner labels | self-hosted, Windows, X64 (**name is not an Actions label**) |
| OS | Windows 11 |
| Shell | Windows PowerShell 5.1, executable powershell.exe; pwsh (PowerShell 7) was absent from Actions runner PATH |
| Exact user-observed checkout | C:\Users\175199\maxkcut |
| Portable checkout | $env:USERPROFILE\maxkcut |
| Exact user-observed Conda interpreter | C:\Users\175199\AppData\Local\miniconda3\envs\maxkcut\python.exe |
| Portable interpreter | $env:LOCALAPPDATA\miniconda3\envs\maxkcut\python.exe |
| Environment | maxkcut |
| Observed library versions (Nov 3) | NumPy 2.5.4, SciPy 1.18.1, Git LFS 3.7.1 |
| User-reported resources | 32 logical processors; an earlier reading ~4% CPU and ~10% RAM **without proven cause** |
| Numerical results branch | experiment025-results |
| Local-only implementation proposal | [PR #70](https://github.com/heri-espino/maxkcut/pull/70) (branch experiment025-local-summary-only); **check current status** |
| Frozen optimizer SHA256 | 99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad |

These Windows paths were observed in the user's transcript, not portable configuration. Verify via git rev-parse --show-toplevel and python -c "import sys; print(sys.executable)". Do not commit GitHub credentials, university access tokens, internal share URLs or sensitive institutional configurations.

## 2. Scientific experiment (do not change midstream)

- The original **signed Max-Cut objective** is evaluated on 71 canonical G-set graphs: G1–G67 plus G70, G72, G77, G81. It is distinct from thesis Chapter 7 random-graph Max-k-Cut.
- **18 methods:** gw-rr, control-paired, gw-threshold, cutaware-kmeans, hyperplane-sweep, multigroup-k4, multigroup-k8, em-hard, em-cem, em-gibbs, mst-l2-longest, mst-l2-balanced, mst-l2-diameter, mst-l2-bestcut, mst-l2-multicut, mst-l1-bestcut, mst-linf-bestcut, control-tabu.
- **Primary baseline:** original Goemans–Williamson hyperplane rounding (gw-rr), refined under the same downstream search. **Secondary control:** GW-seeded K-MeansNM (control-paired). Report raw GW separately.
- **16 pinned shared GW directions per graph and method.** All 16 candidates get independent initializer → Flip (up to 20 sweeps) → KL (up to 30 passes) → ILS (up to 80 cycles, 8 inner KL passes) refinement. Preserve and select the **highest verified original signed cut encountered anywhere**, even when a later iterate worsens.
- Every meaningful iteration has JSONL diagnostics; every strict-best improvement has a versioned binary partition and SHA. Deterministic MST arms are GW-anchored hybrids; equal starting normals do not imply equal computational budgets.
- Experiment021's usable EVA factors are frozen for all 71 graphs. **Do not recompute SDP/EVA or edit src/maxkcut/*.py.** EVA low-rank factors are not certified SDP optima or upper bounds.
- Full size: **1,278** graph-method cells and **20,448** seed-level refinements. Count a cell complete only for status complete_16_best_seen, starts_completed=16, seed indices 0–15, verified partitions. Never treat incomplete arms as zero.

## 3. Important repository files

| Relative file | Purpose |
|---|---|
| AGENTS.md; RESEARCH_HANDOFF.md | Canonical instructions, research state and historical warnings |
| docs/university-vm-experiment025-operations.md | This operational guide |
| notes/experiments/experiment025-windows-local-recovery-2026-11-03.md | User-observed Windows failure and successful G44 seed recovery |
| notes/experiments/experiment025-gw16-bestcut-quality.md | Immutable design, iteration archives, best-ever score semantics |
| notes/experiments/experiment025-successful-pilot-parallel-vm.md | Audited 54-job Actions pilot and VM handoff |
| notes/experiments/experiment025-vm-runbook.md; experiment025-actions-pcag103a13.md | Earlier VM and Actions runbooks; some Git-publishing defaults superseded by local-only mode |
| scripts/experiment025-vm.ps1 | Windows launcher; Conda discovery, validation, serial/parallel dispatch |
| scripts/experiment025-parallel.py | Graph-level parallel scheduler, STOP propagation, serialized Git publisher in older mode |
| scripts/experiment025-windows-io-guard.py | Retries narrow WinError 5/32 atomic JSON/NPZ replacements |
| scripts/experiment025-summary-preview.py | Compact preview generator, available on user's VM and in PR #70; may be absent from remote result/main |
| src/maxkcut/experiment025_bestcut.py; experiment025_journal.py | Frozen numerical optimizer, summary action, per-iteration journal |
| tests/test_experiment025_bestcut.py; test_experiment025_parallel_driver.py; test_experiment025_vm_preflight.py | Source/launcher regression tests |

**VM raw-results root:**

    C:\Users\175199\maxkcut\results\gset\experiment025-gw16-bestcut-quality

Examples, relative to this result root:

    full\G44\multigroup-k8\status.json
    full\G44\multigroup-k8\0000.json
    full\G44\multigroup-k8\0000.npz
    full\G44\multigroup-k8\progress\0000\events.jsonl
    full\G44\multigroup-k8\progress\0000\best.json
    full\G44\multigroup-k8\progress\0000\best_000001.npz
    full\G44\statuses\multigroup-k8.json
    full\G44\logs\multigroup-k8-parallel.log
    analysis\full_coverage.json
    analysis\full_best_final_by_graph.csv
    analysis\full_preview.md

Logs may be named METHOD-vm.log for single-seed PowerShell recovery. The repo .gitignore excludes *.log; no GitHub reader can see unuploaded local-only output.

## 4. Commands for the user to run on the VM

**First verify no Experiment025 process or child is still running.** These are instructions for an idle machine; **do not interrupt the current reported run**.

### Inspect without changing anything

    Set-Location "$env:USERPROFILE\maxkcut"
    git rev-parse --show-toplevel
    git branch --show-current
    git status --short
    python -c "import sys; print(sys.executable)"
    git lfs version
    Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'experiment025' } | Select-Object ProcessId, ParentProcessId, CommandLine

This process list is not a substitute for checking Task Manager and the PowerShell parent process. Do not kill processes indiscriminately.

### Full local-only experiment — preferred, resumable

**Only when idle and after verifying the local PowerShell/Python scripts support -PublishMode local / --publish-mode local:**

    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -AllGraphs -Workers 16 -ChunkSize 16 -PublishMode local

- -Workers 16 creates up to 16 independent **graph** workers, each capped to one BLAS/Numba thread. 32 logical processors do not guarantee a 32-fold acceleration. Reduce count in a *subsequent run* if measured RAM pressure warrants it.
- -ChunkSize 16 is up to 16 *new seeds per subprocess*, not journal flush frequency. Intermediate iterations and strict-best partitions are recorded during execution.
- Already completed and verified seeds are reused on resume. Local-only mode avoids Git/LFS checkpoints; the user must arrange a separately verified off-VM backup.

### Recover one failed seed when its graph is idle

The following **historically succeeded** for G44 and seed 1; do not blindly rerun it:

    Get-Content ".\results\gset\experiment025-gw16-bestcut-quality\full\G44\logs\multigroup-k8-parallel.log" -Tail 120
    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -OnlyGraph -Graph G44 -Method multigroup-k8 -Seed 1 -Workers 1 -PublishMode local

Human seed 1 is internal 0. Inspect actual failed graph/method/seed and source fingerprint before replay. The isolated PowerShell launcher should invoke the guarded Windows I/O wrapper.

### Generate local coverage and Markdown preview **after writers stop**

    $env:PYTHONPATH = (Resolve-Path .\src).Path
    python -m maxkcut.experiment025_bestcut summary --phase full
    python .\scripts\experiment025-summary-preview.py --phase full
    Get-Content ".\results\gset\experiment025-gw16-bestcut-quality\analysis\full_coverage.json"

These commands do **not** rerun EVA; the aggregate is from statuses, not independent rescoring. A missing preview script means the checkout may predate PR #70; do not pull over an active experiment.

### Run quick preflight tests

    $env:PYTHONPATH = (Resolve-Path .\src).Path
    python -m pytest tests/test_experiment025_bestcut.py tests/test_experiment025_parallel_driver.py tests/test_experiment025_vm_preflight.py -q

The user's isolated G44 recovery printed 31 tests passed; passing tests does not certify complete numerical coverage.

## 5. GitHub Actions workflow selection

Use the repository **Actions** tab. Only workflows with workflow_dispatch can be manually launched through **Run workflow**. Choose the correct branch, inputs and destination. **Never dispatch self-hosted Experiment025 numerical work while the direct VM run is active.**

| Workflow | Use |
|---|---|
| .github/workflows/experiment025-quality-tests.yml | Fast synthetic hosted tests; appropriate for CI |
| experiment025-actions-ci.yml; experiment025-seed-first-ci.yml; experiment025-vm-preflight-ci.yml | Workflow matrix, source, PowerShell parsing and detached-HEAD tests; branch/path-dependent CI |
| experiment025-quality-first.yml | **Manual fallback** pilot, full graph/method or batch on self-hosted Windows; 360-minute job cap, Git/LFS publication |
| experiment025-seed-round.yml plus experiment025-seed-shard.yml | **Manual fallback**, one matched GW seed 1..16, six chained shards across all methods; high overhead |
| gset-sdp-manual.yml and historical EVA workflows | **Do not use for Experiment025** |
| visualization-smoke.yml, Sphinx docs | Docs/visualization checks from separate PR/worktree |

Self-hosted workflows request runs-on [self-hosted, Windows, X64], then **verify RUNNER_NAME=PCAG103A-13**. This machine uses shell: powershell, not pwsh. A GitHub Actions checkout may be detached HEAD; avoid direct .Trim() on empty git branch output. The runner must discover existing Conda maxkcut, native DLL path and CRLF-normalized scientific source hash.

Past [successful run #41570037856](https://github.com/heri-espino/maxkcut/actions/runs/41570037856) ran 54/54 numerical **pilot** graph-method jobs for G1, G11, G14 (18 methods each), **GW seed 1 only**, in 48m34s. G1/GW-RR Python numerical search took ~1.25s compared with ~41s total job. This repeated serial checkout/Conda/Git overhead explains why a direct VM command is preferred for the 20,448-seed study. No one should call this pilot the finished benchmark.

## 6. Windows checkpoint fault, exact evidence and recovery

The earlier full local-only run stopped after G44/multigroup-k8 failed on its first seed. The log's first command was **unguarded** Python module execution:

    python -m maxkcut.experiment025_bestcut single --graph G44 --method multigroup-k8 --phase full --max-new-seeds 16

During a Kernighan–Lin journal callback, atomic_json could not replace progress\0000\best.json.tmp with best.json:

    PermissionError: [WinError 5] Access is denied

This is an **OS file-replacement failure**, not proof of mathematical failure. The actual locking program was not found; antivirus/indexing remains a hypothesis. The remote experiment025-results branch had scripts/experiment025-windows-io-guard.py with bounded retries for known WinError 5/32 and approved JSON/NPZ replacements, but the failing process **did not invoke it**. A later isolated guarded PowerShell run successfully recovered the first seed:

    G44 / multigroup-k8 / seed 1:
    starts_completed=1/16, index=[0]
    best_initial=6521, best_seen=6588
    optimizer_source_sha256=99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad
    31 preflight tests passed; no Git uploads

Parallel scheduler behavior: after a graph failure, STOP prevents new work; already active chunks can finish and save locally. Many graphs reporting stopped does **not** erase their earlier checkpoints. Never delete .git/experiment025-parallel.lock before confirming no owner remains. Read the exact traceback and [incident record](../notes/experiments/experiment025-windows-local-recovery-2026-11-03.md) before recovery.

## 7. VM-local versus remote results divergence

The user successfully ran scripts supporting -PublishMode local and the Markdown preview. At the reviewed GitHub snapshot **origin/experiment025-results did not have the local-mode flags**, although it did contain the guarded wrapper, while main did not yet have the preview. The optional implementation lives in [PR #70](https://github.com/heri-espino/maxkcut/pull/70) pending independent status verification. It is unsafe to infer the running Python scheduler's implementation solely from latest GitHub; a running process does not reload changed scripts.

**Never git pull/reset/checkout/clean, merge main into the active scientific results branch, or stage git add . while local raw data is uncommitted.** .gitignore does not hide *tracked* raw files. A Git push sends all unpublished prior commits, even when the newest commit stages only a CSV.

For compact publication, **after writers stop and backup is verified**, create a separate clean worktree from origin/main and a new publication branch, copy only analysis/full_coverage.json, analysis/full_best_final_by_graph.csv, analysis/full_preview.md and approved summary figures/manifests, then open a PR. This avoids pushing the old raw-results history. Do not silently claim that PR merges happened or that VM changes reached remote.

## 8. Backup, verification, exit criteria and reporting

Local-only storage is **not** durable by itself. Preserve the complete VM archive on an approved independent off-VM location (university persistent share, external drive or object storage), with a manifest of relative paths, sizes, file counts, SHA256, source/graph/factor/direction fingerprints and retrieval instructions. Validate hashes; never publish private credentials or institution-only share paths. Do not use destructive robocopy /MIR or delete the source on an unverified copy.

The last user-observed **local** coverage command, before the restart, reported:

    planned_cells=1278
    completed_cells=755
    planned_independent_full_refinements=20448
    all_complete=false

This is **59.08% complete graph-method cells**, at least 12,080 full seed trajectories, 523 incompletely covered cells. It is a **dated snapshot**, not the live current amount and not a reflection of GitHub remote. Some partial cells may already have saved seeds.

**Final completion requires** 1,278/1,278 complete 16-seed cells; 20,448 independently verified seed partitions; all_complete=true; independent rescoring of original signed cuts and hashes; paired GW comparisons, topology and uncertainty; documented unequal method effort and best-known-versus-proven-optimum limits; verified off-VM backup; compact reviewed results PR.

**Successor agent handoff after every incident:** date/time and timezone, runner host, branch and optimizer source hash, exact command/flags, live versus stopped process, log path, graph/method/seed, failure type/root-cause confidence, recovery outcome, *VM-local* and *GitHub-confirmed* coverage separately, backup state, PR, and next safe command. Do not infer VM status from a stale remote commit.


## 9. How to access the VM / self-hosted runner (without conflating them)

The user must **log into the managed Windows 11 university machine** using the university's authorized method. No RDP address, university account password, scheduled-task name, Windows service installation, or Actions runner installation folder was verified; **do not invent them**.

- For **direct research computation**, open Windows PowerShell 5.1, enter the repository (Set-Location "$env:USERPROFILE\maxkcut"), use the existing maxkcut Conda interpreter, and run the scripts in section 4. **The GitHub runner does not need to be online** for direct terminal computation. The script normally discovers Conda; if needed, activate with conda activate maxkcut in the existing Conda-enabled shell.
- For **GitHub Actions self-hosted jobs**, the runner listener must be **online**. Check https://github.com/heri-espino/maxkcut/settings/actions/runners/21 or Settings → Actions → Runners (requires authorization). The prior setup preferred an **interactive, logged-on-only runner**, but the exact auto-start task and runner folder have **not** been verified by this handoff. Check with:

      Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'Runner\.(Listener|Worker)' } | Select-Object Name, ProcessId, ExecutablePath, CommandLine
      Get-ScheduledTask | Where-Object { $_.TaskName -match 'runner|GitHub' } | Select-Object TaskName, State

  These are **inspection commands only**. If none are found, do not assume the runner is installed incorrectly: it may be idle/offline, installed elsewhere, or the task named differently. Do not install a second runner, enable a Windows service, update runner credentials, or modify university login/startup policy without authorization.
- The current physical PC is shared university infrastructure. Minimize unexpected load, respect the logged-on-only preference, and do not use Actions to bypass institutional access controls.

## 10. Concrete backup and compact-publication procedure (only after writers stop)

**Step 1: choose an approved backup location.** The example below deliberately requires the operator to supply the destination; no actual off-VM share address has been verified:

    Set-Location "$env:USERPROFILE\maxkcut"
    $source = (Resolve-Path ".\results\gset\experiment025-gw16-bestcut-quality").Path
    $destination = Read-Host "Enter the approved OFF-VM persistent backup path"
    if ([string]::IsNullOrWhiteSpace($destination)) { throw "Backup destination required" }
    robocopy $source $destination /E /Z /COPY:DAT /DCOPY:DAT /R:2 /W:2 /XJ
    if ($LASTEXITCODE -ge 8) { throw "Robocopy reported failure ($LASTEXITCODE)" }

**Step 2: verify actual backup contents.** Count files and create a SHA256 source manifest on the backed-up snapshot (hashing *all* files can take time; do this only after computation quiescence):

    $sourceFiles = @(Get-ChildItem -LiteralPath $source -Recurse -File)
    $destinationFiles = @(Get-ChildItem -LiteralPath $destination -Recurse -File)
    "Source files: $($sourceFiles.Count)"
    "Backup files: $($destinationFiles.Count)"
    if ($sourceFiles.Count -ne $destinationFiles.Count) { throw "File count mismatch" }
    $sourceManifest = foreach ($file in $sourceFiles) {
        [pscustomobject]@{
            RelativePath = $file.FullName.Substring($source.Length).TrimStart('\')
            Bytes = $file.Length
            SHA256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
    }
    $sourceManifest | Export-Csv -NoTypeInformation -Encoding UTF8 "$env:USERPROFILE\experiment025-source-manifest.csv"

For strong verification, independently hash destination files too and compare matching relative paths, sizes and hashes before treating the backup as valid. A **file-count comparison alone is not sufficient**. Store the verified manifest, snapshot date, optimizer hash, graph count, backup media/retention policy and authorized retrieval instructions in the research handoff; never commit an internal share address or credentials publicly.

**Step 3: publish only compact files via a separate branch/worktree.** This example does not merge the raw experiment results branch or push its uncommitted bulk history:

    Set-Location "$env:USERPROFILE\maxkcut"
    git fetch origin main
    git worktree add -b experiment025-compact-report "$env:USERPROFILE\maxkcut-compact-report" origin/main
    $src = Join-Path (Get-Location).Path "results\gset\experiment025-gw16-bestcut-quality\analysis"
    $dst = "$env:USERPROFILE\maxkcut-compact-report\results\gset\experiment025-gw16-bestcut-quality\analysis"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item "$src\full_coverage.json" $dst
    Copy-Item "$src\full_best_final_by_graph.csv" $dst
    Copy-Item "$src\full_preview.md" $dst
    Set-Location "$env:USERPROFILE\maxkcut-compact-report"
    git add -- results/gset/experiment025-gw16-bestcut-quality/analysis/full_coverage.json results/gset/experiment025-gw16-bestcut-quality/analysis/full_best_final_by_graph.csv results/gset/experiment025-gw16-bestcut-quality/analysis/full_preview.md
    git diff --cached --stat
    git commit -m "experiment025: publish compact audited results snapshot"
    git push -u origin experiment025-compact-report

Open a PR into main from **experiment025-compact-report** and include the local snapshot timestamp, complete/incomplete coverage and provenance. Use another name if the example branch or worktree already exists. Confirm files are valid and the backup complete before final publication. **Do not merge incomplete numerical results into main.** Documentation-only changes can be maintained separately from numerical results.

## 11. Branch ownership and source-of-truth precedence

For new agents, in descending operational authority:

1. **Currently running actual VM process and local files** for that run (read-only inspection; ask user for evidence).
2. **Frozen optimizer and verified per-seed identities** for mathematical validity and reproducibility.
3. **Current GitHub main documentation and associated reviewed PRs** for procedures.
4. **Remote experiment025-results** for *published* coverage only (not unuploaded VM progress).
5. Prior chat summaries and stale commits (historical context only).

When these conflict, **do not overwrite or reconcile the active VM automatically**. Document the discrepancy, preserve artifacts, verify the exact code/hash, and propose a safe post-run migration. This is especially important for PR #70 and the Windows I/O guard, which differed between VM launcher revisions.
