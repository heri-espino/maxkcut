# Experiment025 — local full archive; compact GitHub publication

Status: optional storage-mode change developed 2026-11-03. **Never apply code edits, switch branches, or pull while the currently running full-quality VM scheduler is active.** This branch adds no changes under `src/maxkcut/` and makes no new EVA or SDP solves.

## Goal and scope

The 71 × 18 × 16 full-quality design is unchanged. Persist every meaningful initializer/Flip/KL/ILS iteration, best-ever binary partitions, seed identities, graph/factor/source hashes, original signed cut scores and logs to the existing local Experiment025 results directory. Avoid thousands of Git commits and LFS uploads while computing. Publish only a small coverage CSV/JSON plus a provisional Markdown preview **after the scheduler is stopped**.

## Local mode

Run from the checked-out `experiment025-results` branch in the existing Windows 11 / Conda `maxkcut` environment after incorporating the operational changes and ensuring no other active process uses the output directory:

```powershell
powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -AllGraphs -Workers 16 -ChunkSize 16 -PublishMode local
```

- `-PublishMode local`: keeps all full data locally and disables both the Python scheduler's serialized `git add/commit/push` and the PowerShell wrapper's `Publish-Checkpoint`. Checkpoint frequency, numerical seed ordering and source fingerprint are unchanged. A failed seed still leaves its local iteration archive and diagnostic log.
- `-PublishMode git` (default): backwards-compatible full Git/LFS publication.
- `-ChunkSize` governs **computational subprocess chunk size**, not local checkpoint durability. Journal events and record-bests are written during computation. `-Workers` selects up to 32 **independent graphs**; each worker's BLAS threads are capped at one.
- Local-only mode requires the `experiment025-results` branch and permits dirty files only inside `results/gset/experiment025-gw16-bestcut-quality/`. It refuses to switch branches and does not pull. **Never run `git clean`, `git reset --hard`, or `git add .` while local raw outputs are uncommitted.** The Git index will still report already tracked raw files as modified: `.gitignore` cannot hide tracked files.
- If the current live scheduler was launched without this mode, changing the script or publishing the PR has no effect on that already-running Python process. Wait for it to stop at a safe checkpoint; inspect active child Python processes and ensure the checkpoint journal has flushed before resuming.

## Provisional summary publication

After **stopping all writers**:

```powershell
python -m maxkcut.experiment025_bestcut summary --phase full
python .\scripts\experiment025-summary-preview.py --phase full

# Stage ONLY these three small files, never the complete experiment directory.
$base = 'results/gset/experiment025-gw16-bestcut-quality/analysis'
git add -- "$base/full_coverage.json" "$base/full_best_final_by_graph.csv" "$base/full_preview.md"
git diff --cached --stat
git commit -m "experiment025: publish compact quality and coverage snapshot"
git push origin HEAD:experiment025-results
```

The preview includes design, coverage, finished-seed counts, method-level completion, and small example results **only for graphs where all methods are complete**. It never ranks incomplete methods. If push is non-fast-forward, **stop** and reconcile with the other publisher; do not force-push or reset local data. The summary-only branch publication can be integrated into main later following an independent original-weight rescore and final audit.

## Data preservation and backup

**A VM-only copy is not a sufficiently durable research archive.** Preserve full data in a separately named off-VM backup (university share, external drive, or object storage) with checksum manifest, access controls, retention plan, and a documented retrieval path for future agents. Document the exact saved backup location, snapshot date, graph count, file count, total bytes, and verified hashes in a run manifest. Do not commit institutional credentials or private paths. Git retains optimizer code, protocol, provenance, compact summaries, reports, figures and instructions.

## Acceptance tests

```powershell
python -m pytest tests/test_experiment025_parallel_driver.py tests/test_experiment025_local_publication.py tests/test_experiment025_vm_preflight.py -q
```

No numerical seed, EVA factor, or SDP is recomputed by these tests.
