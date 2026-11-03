# Experiment025 — WinError 5 checkpoint replacement recovery (2026-11-03)

**Status:** Root *immediate* failure mechanism confirmed by three Windows 11 VM tracebacks supplied by the operator. The identity of any third-party file locker is **unknown**. This is an operational fault, not evidence of an algorithmic flaw. No EVA recomputation and no numerical source edits are needed.

## Root error and evidence

All failing subprocesses exited 1 at `src/maxkcut/experiment023_methods.py:79` inside `atomic_json`, called from `src/maxkcut/experiment025_journal.py:98`. The failing operation was:

```python
os.replace(
    "results/gset/experiment025-gw16-bestcut-quality/full/G21/control-paired/progress/0002/best.json.tmp",
    "results/gset/experiment025-gw16-bestcut-quality/full/G21/control-paired/progress/0002/best.json",
)
```

Windows raised `PermissionError: [WinError 5] Access is denied` for all three cases:

| Case | Failure stage | Completed remote seeds before failure | In-progress seed |
| --- | --- | ---: | --- |
| G21 / control-paired | KMeansNM, best checkpoint | 0–1 (2) | index 2 (human seed 3) |
| G10 / control-paired | KMeansNM, best checkpoint | 0–6 (7) | index 7 (human seed 8) |
| G20 / gw-threshold | Flip, best checkpoint | 0–5 (6) | index 6 (human seed 7) |

The previously reported G24/control-paired `git push` `curl 55` is separate: the remote has the complete, saved 16/16 commit, despite the local Git command returning failure. See `notes/experiments/experiment025-full-windows-parallel-interruption.md` in documentation PR #67 for the first-pass status audit.

**Do not conclude** that antivirus, Git LFS, the OS indexer, OneDrive, or any particular Windows service caused the lock: these are possible explanations only. `os.replace` can fail if either file is temporarily unavailable to the caller. Permanent filesystem ACL problems would not be solved by retries.

## Operational-only mitigation

Implementation in **`scripts/`** rather than `src/maxkcut/`:

1. `scripts/experiment025-windows-io-guard.py` invokes the *unchanged* `maxkcut.experiment025_bestcut` module, intercepting only atomic JSON/NPZ replacements within `results/gset/experiment025-gw16-bestcut-quality/`.
2. Only Windows `PermissionError.winerror` **5** (access denied) or **32** (sharing violation) is retried; at most ten attempts with bounded backoff. Every other exception immediately propagates; prolonged lock remains an explicit failure. No unlink, delete, copy, or non-atomic write fallback.
3. `scripts/experiment025-parallel.py` and the two serial `single` call sites in `scripts/experiment025-vm.ps1` use this launcher. Existing optimizer source tree, seed streams, graph factor files, objectives and stage budgets remain untouched, preserving recorded source SHA256 `99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad`.
4. The publisher force-stages only the current graph/method `*-parallel.log` alongside the result checkpoint. Global `*.log` ignoring remains unchanged so previously ignored logs do not unexpectedly cause the PowerShell clean-worktree preflight to reject a resumed run.
5. A `git push` client failure triggers a remote SHA receipt check via `git ls-remote`; if the remote already points at the exact local HEAD it is confirmed successful. Otherwise only a bounded set of transport errors is retried; non-fast-forward errors are never force-pushed, rebased, or automatically overwritten.
6. New mock-only tests `tests/test_experiment025_windows_io_guard.py` and additions to `tests/test_experiment025_parallel_driver.py`; hosted CI includes the retry test. **This does not constitute a successful real Windows VM smoke test** until explicitly run on PCAG103A-13.

The partial in-progress journal contains prior events and a possibly stale `best.json.tmp`. The existing journal verifier validates the last committed `best.json` and the completed `000N.json/.npz` stage archives before continuation. The in-progress *seed* can be re-executed deterministically and the JSONL journal may then contain events from more than one attempt. Do not silently deduplicate these; treat `attempt`/reruns as provenance and validate the eventual best cut against original signed graph edges.

## Safe sequence on the university VM

**Never begin by resetting, cleaning or switching away from local uncommitted research results.** Stop all competing writes and inspect:

```powershell
git branch --show-current
git status --short
git fetch origin experiment025-results
git rev-list --left-right --count HEAD...origin/experiment025-results
git lfs status
```

Do not `git reset --hard`, `git clean`, `git push --force`, or discard ignored logs. If the local and remote refs diverge, reconcile intact committed snapshots before proceeding.

After the operational fix PR is reviewed, tested, and **merged to experiment025-results** (the current PowerShell script automatically switches to that branch), fetch the fast-forwarded results branch while no process is writing to it. Test a **single failed seed** on the same existing VM, avoiding any new EVA solve:

```powershell
powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -OnlyGraph -Graph G21 -Method control-paired -Seed 3 -Workers 1
```

The analogous failure index for G10/control-paired is `-Seed 8`, and G20/gw-threshold is `-Seed 7`. Once the originally failed trajectories succeed and the remote checkpoint is verified, resume the full sweep. Start with `-Workers 8 -ChunkSize 4` if the physical cause of file contention remains uncertain, observe `[experiment025-io]` retry messages, then increase parallelism only with evidence of stable filesystem throughput and memory headroom.

The full `71 × 18 × 16` experiment is **still incomplete**. The operational patch is not a claim that all saved partial JSONL logs or stage binaries have been scientifically audited. The final `full_coverage.json` must report all 1,278 method cells complete and every selected partition must be rescored on the original signed graph.

## Reproduction and limitations

Mock-only regression command from the repository root (existing Python test environment):

```powershell
python -m pytest tests/test_experiment025_windows_io_guard.py tests/test_experiment025_parallel_driver.py -q
```

Even a passing hosted test cannot replicate your university filesystem's antivirus/indexer state or all 32 simultaneous workers. A permanent WinError 5 after all retries is an intentional stop, not silently ignored. Do not change file permissions or disable institutional security tools without authorization.
