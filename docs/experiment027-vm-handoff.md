> **NEW PREFERRED NO-LFS GITHUB PUBLICATION (Nov 7, 2026):** User needs complete VM numerical results preserved in the repository even if Git LFS storage is exhausted. Follow [GitHub Releases publisher and checksummed regular-Git index](experiment027-release-no-lfs.md); use `scripts/experiment027_publish_release.py` from the idle Experiment027 VM worktree. The older LFS runbook remains historical. Publishing is not complete until the user executes it and `published/experiment027/latest_release.json` and corresponding server-verified Release assets appear remotely. No EVA recompute, no repeated Experiment025, no merge.

> **DIRECT REPOSITORY PUBLICATION (2026-11-05):** The user cannot transfer VM raw files through ChatGPT. Follow [Experiment027 LFS publication instructions](experiment027-publication.md). The new `scripts/experiment027_publish.py` audits, archives and pushes directly from the idle VM. Graph ZIPs go into `published/experiment027/archives/` (Git LFS) and analyses/manifests into normal Git. This is prepared tooling, **not evidence that the user's numerical files have been pushed**. Preserve Experiment025 raw outputs and verify remote Git LFS payloads before claiming backup.

# Experiment027 — Windows VM file map, successful-pilot handoff and safe resume

## Read-only post-run audit and paired comparison (2026-11-05)

The user subsequently reported that the numerical experiments **ran successfully**. The available conversation does not establish whether this means only the corrected 3-graph pilot or all 71 graphs; there are no copied VM numerical status files in GitHub. **Do not assert 213/213 cells until checked.**

New **`scripts/experiment027_analyze.py`** is deliberately outside `src/maxkcut/` and does not change the content-hashed Experiment027 numerical runner. It independently reads the original signed graph and the exact SHA-pinned EVA factor, checks 16 GW normals are byte-equal across old/new experiments, verifies all 16 seeds for GW-RR, historical Hyperplane Sweep and each of the three new methods, validates per-stage NPZ SHA256 and original signed-cut scores, strict-best checkpoint SHA and score, counts every JSONL event, and independently scores each accepted high-dimensional normal from `accepted_normals.npz`. It fails closed on missing/mismatched input; no missing data is treated as zero. It then generates per-seed paired differences, graphwise best-of-16 comparisons, graph-cluster bootstrap intervals and a method leaderboard.

From an **idle** Experiment027 worktree, in `conda activate maxkcut`, choose **one** command based on what actually ran:

```powershell
# After the successful G1/G11/G14 pilot, audit exactly 3 graphs / 9 new cells:
python scripts/experiment027_analyze.py --graphs G1 G11 G14

# If ALL 71 graphs have been run, audit full 71 graph / 213 new cells:
python scripts/experiment027_analyze.py --graphs ALL
```

Analysis outputs are written under `$env:USERPROFILE\maxkcut-exp027\results\gset\experiment027-angular-greedy\analysis\`:
`audited_comparison.json`, `audited_seed_scores.csv`, `audited_graph_comparisons.csv`, `audited_method_leaderboard.csv`, and `audited_paired_seed_deltas.csv`.

Expected `audited_method_seed_archives` includes two historical comparator methods: **80 per graph**, so **240** for three graphs or **5,680** for 71; of those, new method-seed archives are **48 per graph** (144 or 3,408). The audit script does not rerun any optimization, does not edit historical raw output, and writes only the new study's `analysis/` directory. Share the small `audited_comparison.json` and leaderboard CSV for interpretation. Before calling any result a new best-known cut, perform independent provenance and reference verification. Back up *both* raw archives off-VM with verified checksums; do not leave unique experimental data only on the Windows computer.


**Dated:** 2026-11-05. **Branch:** `experiment027-angular-greedy` ([draft PR #80](https://github.com/heri-espino/maxkcut/pull/80)). **This document is the current Experiment027 operational entrypoint.**

## Status and evidence (do not overstate)

- The user **reported that the corrected G1/G11/G14 full-phase pilot succeeded** after executing `scripts/experiment027_from_archives.py` on the university Windows 11 VM. The exact console completion summary, nine status files, numerical scores, per-seed audit and off-VM backup have **not** been supplied to this conversation or independently verified remotely. Record this as **user-reported successful execution**, **NOT independently certified 9/9 completed cells or 144/144 seed archives**.
- An earlier direct call to `scripts/experiment027_angular.py` failed because it supplied **one** Experiment021 factor folder for G1, G11 and G14, while `cases/G14.json` was absent in that folder. The failure was in input selection, not evidence that EVA or the angular algorithm failed. The first parallel call might have persisted G1/G11 results; never erase them.
- The repaired factor-selector launcher, `scripts/experiment027_from_archives.py`, was committed without changing the content-hashed numerical runner `scripts/experiment027_angular.py`. Hosted workflow [#42117365600](https://github.com/heri-espino/maxkcut/actions/runs/42117365600) passed **12/12** small-graph tests and confirmed frozen Experiment025 source SHA256 `99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad`. This is a code check, **not** a production-data audit.
- Original Experiment025 computations were reported complete locally: **71 graphs × 18 historical methods × 16 starts = 20,448 recorded refinements**. Their full raw trajectories remain on the university VM unless separately backed up. Do **not** recompute them, and do **not** rerun the EVA/SDP solves.
- Full 71-graph Experiment027 results have **not** been independently confirmed. The three added algorithms are `angular-margin-greedy`, `angular-gain-greedy`, and `angular-pair-greedy`, each seeded by exactly the same 16 GW directions used in Experiment025 (seed indices **0–15**).

## Machine and paths (last user-observed; verify at runtime)

Windows 11, PowerShell 5.1, self-hosted Actions runner `PCAG103A-13`, existing Conda environment `maxkcut`. University VM username/path examples below reflect the observed machine and may not be portable.

| Role | Path on the university VM | Mutation policy |
| --- | --- | --- |
| **Original checkout and historical source data** | `C:\Users\175199\maxkcut` | **READ ONLY** for Experiment027; never reset, clean or checkout over unpublished raw output |
| **New Experiment027 worktree** | `C:\Users\175199\maxkcut-exp027` | Current execution and **new** files only |
| Canonical signed G-set inputs | `C:\Users\175199\maxkcut\data\G*.txt` | Read only; verify original bytes/checksum |
| **Historical Experiment025 results and GW normals** | `C:\Users\175199\maxkcut\results\gset\experiment025-gw16-bestcut-quality` | Read only; local-only full archive may be newer than GitHub |
| First eligible EVA factor archive | `C:\Users\175199\maxkcut\results\gset\experiment021-complete-71-gset\eva-numba` | Read only; not necessarily available for every graph |
| Second eligible EVA factor archive | `C:\Users\175199\maxkcut\results\gset\experiment003-eva-numba` | Read only |
| Third eligible EVA factor archive | `C:\Users\175199\maxkcut\results\gset\experiment002-eva` | Read only |
| **New Experiment027 results** | `C:\Users\175199\maxkcut-exp027\results\gset\experiment027-angular-greedy` | Append new verified results; never erase already completed seeds |
| Conda interpreter, previously observed | `C:\Users\175199\AppData\Local\miniconda3\envs\maxkcut\python.exe` | Verify which interpreter is running |

**Important:** Do **not** run the old multi-graph `--factor-dir ...experiment021...` command. Experiment025 chose the factor **per graph**. The new launcher insists on each factor's exact SHA256 matching the completed Experiment025 GW baseline and checks the graph file bytes before executing *any* new method. If an archive is absent, stop and locate the preserved original; **never substitute another factor or solve EVA again**.

The old traceback also showed `maxkcut` being imported from `maxkcut-exp026\src\maxkcut` due to the editable Python environment. The new selector prepends the **Experiment027 worktree's** `src` to the launched subprocess's `PYTHONPATH`. Prefer the selector to direct ad hoc `python scripts/experiment027_angular.py...` calls. Experiment026's separate [PR #77](https://github.com/heri-espino/maxkcut/pull/77) must remain untouched.

## Source-code and documentation index

| Repository-relative file | Why it matters |
| --- | --- |
| `AGENTS.md` | Canonical AI-agent rules and current experiment status |
| `RESEARCH_HANDOFF.md` | Cross-experiment context, preservation and open tasks |
| `notes/experiments/experiment027-angular-greedy.md` | Exact scientific method, parameters, experimental design, statistical comparison and recovery |
| **`docs/experiment027-vm-handoff.md`** | **This file: VM path map, past error, pilot status, archival layout, validation and safe next steps** |
| `scripts/experiment027_from_archives.py` | **Correct launcher**; resolves the Experiment025-pinned EVA factor separately for every graph, verifies all inputs before starting, runs one graph per subprocess and writes launcher logs |
| `scripts/experiment027_angular.py` | **Frozen content-hashed numerical implementation**; 3 angular methods, exact signed-cut sweeps, 16 GW normals, refiner reuse, checkpoint/resume, summaries |
| `tests/test_experiment027_angular.py` | Geometry, deterministic signed-cut, numerical invariants, checkpoint replay |
| `tests/test_experiment027_factor_sources.py` | Missing-factor, hash, source-resolution and no-unwanted-recompute regressions |
| `.github/workflows/experiment027-angular-ci.yml` | Hosted-only small tests; **does not** launch large numerical computations on the university machine |
| `src/maxkcut/experiment024_paired_gw16.py` | Deterministic 16 matched GW starting normals and their archive |
| `src/maxkcut/experiment025_bestcut.py` | Frozen 16-start settings and original quality-first experiment |
| `src/maxkcut/experiment025_journal.py` | Per-iteration JSONL and strictly improving immutable partition snapshots |
| `src/maxkcut/experiment023_methods.py` | Historical factor archive priority, original signed-cut verification and per-seed NPZ checks |
| `docs/university-vm-experiment025-operations.md` | Original Experiment025 VM/raw-data protection and backup hazards |

**Do not edit `src/maxkcut/*.py`** for an Experiment027 documentation or launcher fix: Experiment025's saved records are tied to a frozen all-source fingerprint. **Do not change `scripts/experiment027_angular.py` lightly either**: Experiment027's seed fingerprints contain the runner's source SHA, so changing it can intentionally invalidate existing seed reuse. Prefer separate tooling and docs for operational changes.

## Archived file layout: where to look

Define PowerShell roots:

```powershell
$Original = "$env:USERPROFILE\maxkcut"
$Worktree = "$env:USERPROFILE\maxkcut-exp027"
$OldRoot  = Join-Path $Original 'results\gset\experiment025-gw16-bestcut-quality'
$NewRoot  = Join-Path $Worktree 'results\gset\experiment027-angular-greedy'
```

**Experiment025 historical full archive**, example G11:

```text
$OldRoot\full\G11\gw16_common.npz
$OldRoot\full\G11\gw16_common.json
$OldRoot\full\G11\gw-rr\status.json
$OldRoot\full\G11\hyperplane-sweep\status.json
$OldRoot\full\G11\hyperplane-sweep\0000.json
$OldRoot\full\G11\hyperplane-sweep\0000.npz
$OldRoot\full\G11\hyperplane-sweep\progress\0000\events.jsonl
$OldRoot\analysis\full_best_final_by_graph.csv
$OldRoot\analysis\full_coverage.json
```

**Experiment027 new full archive**, example G11 + `angular-gain-greedy` + seed index zero:

```text
$NewRoot\full\G11\gw16_common.npz
$NewRoot\full\G11\gw16_common.json
$NewRoot\full\G11\angular-gain-greedy\status.json
$NewRoot\full\G11\angular-gain-greedy\0000.json
$NewRoot\full\G11\angular-gain-greedy\0000.npz
$NewRoot\full\G11\angular-gain-greedy\progress\0000\events.jsonl
$NewRoot\full\G11\angular-gain-greedy\progress\0000\best.json
$NewRoot\full\G11\angular-gain-greedy\progress\0000\best_000001.npz
$NewRoot\full\G11\angular-gain-greedy\progress\0000\accepted_normals.npz
$NewRoot\launcher-logs\G11.log
$NewRoot\analysis\full_method_summary.csv
```

Interpretation: `0000` denotes **seed index 0**, not a graph ID; sixteen seeds are `0000` through `0015`. Each stage NPZ holds `Round`, `Flip`, `KL`, `Final` partitions. The per-seed JSON contains signed original graph scores, start data and SHA-linked provenance. `events.jsonl` contains **each attempted candidate angular sweep and each refinement step**, whereas `accepted_normals.npz` records the shorter accepted hyperplane trajectory. `best_*.npz` versions preserve strict-new-best binary partitions. A completed seed is reused only after hash, fingerprint, graph and cut checks; an unfinished journal stops rather than being wiped.

## Correct verification and resume commands (PowerShell, from the NEW worktree only)

**Before any new computation:** ensure the old Experiment025 and any other experiments are idle; do not run concurrent local numerical writers or self-hosted Actions. The VM's original history should have a verified off-VM backup before a large 71-graph campaign. The commands below do not remove or rewrite the old worktree.

```powershell
Set-Location "$env:USERPROFILE\maxkcut-exp027"
git status --short
git pull --ff-only origin experiment027-angular-greedy
conda activate maxkcut
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'

# Tests only; no EVA or long numerical jobs:
python -m pytest -q tests/test_experiment027_angular.py tests/test_experiment027_factor_sources.py

# Input preflight only: checks exact signed graphs and Experiment025-pinned factors:
python scripts/experiment027_from_archives.py --graphs G1 G11 G14 --workers 3 --dry-run

# Resume/verify the already user-reported successful THREE-graph full pilot.
# Completed verified new seeds are skipped; no old methods run:
python scripts/experiment027_from_archives.py --graphs G1 G11 G14 --workers 3

# Summarize files already on disk; does not optimize:
python scripts/experiment027_angular.py summary --phase full
```

**Do not assume the user-reported pilot automatically means 9/9 complete status files.** Run a read-only status check:

```powershell
$NewRoot = "$env:USERPROFILE\maxkcut-exp027\results\gset\experiment027-angular-greedy"
$methods = 'angular-margin-greedy','angular-gain-greedy','angular-pair-greedy'
& {
    foreach ($g in 'G1','G11','G14') {
        foreach ($m in $methods) {
        $p = Join-Path $NewRoot "full\$g\$m\status.json"
        if (Test-Path $p) {
            $r = Get-Content $p -Raw | ConvertFrom-Json
            [pscustomobject]@{ Graph=$g; Method=$m; Status=$r.status; Starts=$r.starts_completed; Best=$r.best_final; Paths=$r.total_paths }
        } else {
            [pscustomobject]@{ Graph=$g; Method=$m; Status='MISSING'; Starts=0; Best=$null; Paths=$null }
        }
        }
    }
} | Format-Table -AutoSize
```

Do not present missing cells as zero-valued Max-Cut scores.

Target for the three-graph full pilot: **9 graph-method cells × 16 seeds = 144 completed new seed archives**. Require `status == 'complete_16'`, `starts_completed == 16` and independently verify stage/partition rescoring before calling it audited.

**Only after pilot audit and verified off-VM backup:** expand to 71 graphs, three new methods and all sixteen matched starts (3,408 maximum new seed jobs), using graph-parallel workers with one BLAS/Numba thread each:

```powershell
python scripts/experiment027_from_archives.py --graphs ALL --workers 8
python scripts/experiment027_angular.py summary --phase full
```

The launcher preflights all 71 graph/factor pairs **before** scheduling any new graph. It prints a JSON plan, writes `launcher-logs/Gxx.log`, and exits nonzero if any graph process fails. Start with fewer workers when memory is limited. Reusing completed verified seeds is the default. Never create two concurrent launcher processes targeting the same graph.

## Backup, reporting and unresolved work

1. **Verify pilot completeness** on the VM with all nine `status.json` records and all 16 per-seed archives each; independently rescore `Round`, `Flip`, `KL`, `Final` against the original signed graph, confirm hashes and comparisons to matching old baselines. User success alone is not a numerical audit.
2. **Back up both historical Experiment025 and new Experiment027 raw trees off-VM**, with checksum manifests and verified readback before removing or changing any source workspace. GitHub currently cannot be assumed to contain uncommitted VM-local NPZ/JSONL histories. Avoid raw Git commits without LFS or documented external storage.
3. **Publish compact audited outcomes** (statuses, summary CSV, reproducibility settings, SHA manifests and analysis) back into a separate branch/PR without overwriting Experiment025. Raw data should live in documented durable storage with integrity verification.
4. Compute paired comparisons versus **Experiment025 GW-RR and Hyperplane Sweep** using the identical 16 GW seeds and exact factor hashes. Report raw rounds separately from Flip/KL/ILS and report CPU time / number of semicircle evaluations. Do not call feasible new records proven global optima.
5. **No merging** without user authorization; PR #80 remains a draft until the numerical audit and storage situation are resolved.

**Handoff rule:** new agents begin with `AGENTS.md`, `RESEARCH_HANDOFF.md`, **this document**, and `notes/experiments/experiment027-angular-greedy.md`. Query the VM for actual local progress; do not infer it from GitHub's remote branch.
