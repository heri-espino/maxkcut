# Experiment025 — first 32-worker full-VM interruption and forensic recovery

**Audit:** 2026-11-03. **Repository:** `heri-espino/maxkcut`. **Source:** user-supplied Windows PowerShell transcript from the 32-worker direct-VM run, cross-checked against the live `experiment025-results` GitHub branch. **Last verified remote head at audit:** `4f89704820eb089a50d882fe5c00d4f740ebdb04` (`experiment025: full G19 gw-threshold 16/16 parallel checkpoint`). **Numerical work launched in this audit:** none.

## Scope and protected data

This is the already-defined **Experiment025** full phase, using frozen EVA factor matrices from Experiment021 and the unchanged signed Max-Cut methods under `src/maxkcut/`. The intended complete study remains 71 canonical graphs × 18 methods × 16 shared GW directions = 20,448 independently refined trajectories. The original signed cut at every meaningful iteration and each strictly improving feasible binary partition must survive. Do not re-solve EVA or change optimizer source to repair an orchestration problem. Keep the established source SHA256 `99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad` for saved Experiment025 result compatibility.

## What the transcript proves

The parallel dispatcher did execute the **full** (not pilot) phase and published numerous 16/16 checkpoints. It encountered four recorded failures:

| Graph/method | User-visible error | Verified on remote `experiment025-results` |
| --- | --- | --- |
| G21/control-paired | Python subprocess exit 1, reported `0/16` | Committed complete seed files `0000` and `0001` (2 seeds); in-progress journal `progress/0002/` |
| G10/control-paired | Python subprocess exit 1, reported `0/16` | Committed complete seed files `0000`–`0006` (7 seeds); in-progress journal `progress/0007/` |
| G20/gw-threshold | Python subprocess exit 1, reported `0/16` | Committed complete seed files `0000`–`0005` (6 seeds); in-progress journal `progress/0006/` |
| G24/control-paired | `git push` reported `RPC failed; curl 55 Send failure: Connection was aborted`, exit 1 | **All 16 seeds and `status=complete_16_best_seen` are remotely present**; checkpoint commit `fd8fd99d75e5aa57831e33452c174832985d3b4c` |

These are **15 completed and remotely stored seed archives within the three failed method cells**, not zero successful seeds. The `0/16` publisher label comes from `status.json` being absent: `run_method` writes method-level status after finishing an invocation, while it writes and validates each individual seed immediately. The failed current seeds have `best.json.tmp` and other partial journal files; treat these as forensic evidence, not as completed seeds. `run_method` verifies existing finished stage archives and skips them on resume, but an interrupted seed itself may need re-execution. Independently recheck all saved hashes/cuts through the built-in verifier.

For G24 the remote commit exists despite the push client's nonzero exit status; the Git error does **not** mean the 16 G24/control-paired results are absent. The scheduler's global `STOP` event is why later graphs (including G33 onward) reported `stopped` with `methods:0`: their queued work was cancelled after an error, not proven numerically invalid. In-flight graph workers were allowed to finish their current method and publish checkpoints after `STOP`.

## Primary evidence gap: worker logs are ignored

The current root `.gitignore` has a blanket `*.log` rule. The scheduler writes the actual numerical traceback to:

- `results/gset/experiment025-gw16-bestcut-quality/full/G21/logs/control-paired-parallel.log`
- `results/gset/experiment025-gw16-bestcut-quality/full/G10/logs/control-paired-parallel.log`
- `results/gset/experiment025-gw16-bestcut-quality/full/G20/logs/gw-threshold-parallel.log`

These logs existed according to the transcript, but are **not** visible on the remote results branch because they were ignored during `git add -A`. The true cause of each Python exit code 1 is **not yet established**. Leftover `best.json.tmp` files are consistent with interruption during checkpoint writing, but cannot alone distinguish transient Windows file locks, interrupted processes, low disk space, Python exceptions, or other failures. Do not claim any specific cause without the local tracebacks.

The separate Git push error is clearly a transport failure as reported by Git (`curl 55`), but its corresponding remote ref update apparently succeeded; a more robust future publisher should check the remote SHA after an ambiguous push error before classifying a checkpoint as failed.

## Immediate non-destructive diagnostics on the Windows VM

From the existing repository worktree in **Windows PowerShell**:

```powershell
git branch --show-current
git status --short
git fetch origin experiment025-results
git rev-list --left-right --count HEAD...origin/experiment025-results
git lfs status

$base = 'results\gset\experiment025-gw16-bestcut-quality\full'
@(
  "$base\G21\logs\control-paired-parallel.log",
  "$base\G10\logs\control-paired-parallel.log",
  "$base\G20\logs\gw-threshold-parallel.log"
) | ForEach-Object {
  Write-Host "`n===== $_ ====="
  if (Test-Path $_) { Get-Content -LiteralPath $_ -Tail 100 }
  else { Write-Warning "Missing local log: $_" }
}
```

**Before any `git pull`, switching branches, or rerun:** inspect `git status --short` and divergence. Never discard local uncommitted results, force-push, reset the branch, or rebase a concurrent author's result writes. An ambiguous push may have succeeded remotely; compare local and remote refs. A repository checkout or automated cleanup must not remove ignored local `*.log` files.

After confirming no competing writer, preserve the three small diagnostic logs in the existing results branch with `git add -f -- <exact local log paths>`, inspect the staged diff, then commit and push; do not bulk-force-add unrelated ignored files. A future operations-only fix should change the scoped logging policy so `*-parallel.log` in Experiment025 can be uploaded automatically. Do not launch full computation merely to recover diagnostic traces.

## Safe continuation — only after investigating tracebacks

- Resolve any fatal numerical or filesystem error *without changing `src/maxkcut/`* when possible. If a genuine optimizer bug requires a source change, isolate the new scientific protocol/experiment and do not mix old fingerprints.
- Test the one failed seed of G21/control-paired (user-facing seed **3**, internal index `2`) with a **single** worker and the existing Windows driver, first confirming that all local uncommitted artifacts are safe and that no other process writes results:
  `powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -OnlyGraph -Graph G21 -Method control-paired -Seed 3 -Workers 1`.
  Equivalent failing user-facing seeds: G10/control-paired **8**, G20/gw-threshold **7**.
- The full script `-Phase full -AllGraphs -Workers 32 -ChunkSize 8` should only be resumed after the failures are understood; use a smaller worker count if there is evidence of resource contention.
- The scheduler currently serializes all Git publications, but every graph-method completion still creates a commit/push; the full campaign can produce many heavy Git LFS transactions. A future transport fix should use bounded retries plus `git ls-remote` ref verification, without force-push and without concurrent Git writers.
- The `full_coverage.json` aggregate was not present at this forensic snapshot. Do not declare the experiment complete until independent rescore of all 1,278 fully populated method cells and the `all_complete=true` gate.

## Agent handoff

Relevant files: `AGENTS.md`, `RESEARCH_HANDOFF.md`, `notes/experiments/experiment025-successful-pilot-parallel-vm.md`, `notes/experiments/experiment025-vm-runbook.md`, `scripts/experiment025-parallel.py`, `scripts/experiment025-vm.ps1`, `src/maxkcut/experiment025_bestcut.py`, `src/maxkcut/experiment025_journal.py`, and `tests/test_experiment025_parallel_driver.py`.

This note records evidence and an investigation plan. It is **not** a claim that the Python failures have been repaired or the full sweep has been restarted.
