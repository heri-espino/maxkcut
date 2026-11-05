# Experiment027 — publish verified results to GitHub (Git + Git LFS)

**Status:** Publisher prepared on the separate `experiment027-angular-greedy` branch / draft [PR #80](https://github.com/heri-espino/maxkcut/pull/80). The numerical data remain on the university Windows VM until the publication command runs successfully. Successful optimization does not imply the raw data are already on GitHub.

**Purpose:** Let future agents read verified numerical results directly from the repository, while preserving Experiment027 per-candidate angular-sweep histories, accepted normals, intermediate strict-best partitions and four-stage checkpoints with checksums. Never solve EVA or rerun Experiment025. The PR remains unmerged unless explicitly authorized.

## What gets published

| Destination | Contents | Git storage |
|---|---|---|
| `published/experiment027/analysis/audited_comparison.json` | Scope, leaderboard, uncertainty, sanitized path | Git |
| `published/experiment027/analysis/audited_method_leaderboard.csv` | New methods vs historical GW and Hyperplane Sweep | Git |
| `published/experiment027/analysis/audited_graph_comparisons.csv` | Best-of-16 per graph | Git |
| `published/experiment027/analysis/audited_paired_seed_deltas.csv` | Matched 16-start differences | Git |
| `published/experiment027/analysis/audited_seed_scores.csv` | Each seed and stage, events, CPU time, sweep count | Git |
| `published/experiment027/archives/Gxx-SOURCEHASH.zip` | Entire Experiment027 `full/Gxx/`: new per-seed JSON/NPZ, full iteration JSONL, best-partition versions, accepted-normal trajectories and shared GW normals | **Git LFS** |
| `published/experiment027/manifests/Gxx-SOURCEHASH.json` | Per-file SHA256, source membership and ZIP SHA256 | Git |
| `published/experiment027/index.json` | List of published graphs, archive links and checksums | Git |

Archives are **content-addressed**, so subsequent changed snapshots produce new immutable ZIP files and retain prior history. The index identifies the selected snapshot. `.gitattributes` uses Git LFS only for these ZIP archives, not the analysis or manifests.

**Storage limits:** GitHub LFS storage, bandwidth and object-size quotas depend on the account/repository. The publisher refuses archives exceeding 1,900 MiB each by default; inspect quota and available disk space first. A successful local commit alone does not prove an off-VM backup. The old historical Experiment025 full iteration archive is **not** included automatically; this operation publishes Experiment027 raw results plus old-method scores derived and checked by the audit.

**Privacy:** Raw per-graph result archives exclude standalone `launcher-logs/Gxx.log`, which may contain VM paths and must be reviewed separately if desired. Do not commit credentials, private logs or other unreviewed local files.

## Commands on the university VM (Windows PowerShell)

Only when numerical jobs have stopped, use the new Experiment027 worktree—not the original Experiment025 checkout:

```powershell
Set-Location "$env:USERPROFILE\maxkcut-exp027"
conda activate maxkcut
git status --short
git pull --ff-only origin experiment027-angular-greedy
git lfs version

$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
```

Preview the source volume without creating archives or pushing:

```powershell
# If only the G1/G11/G14 pilot ran:
python scripts/experiment027_publish.py --graphs G1 G11 G14 --dry-run

# If all 71 graphs ran:
python scripts/experiment027_publish.py --graphs ALL --dry-run
```

**Run ONE of the following to independently audit, package, commit and push:**

```powershell
# Pilot only:
python scripts/experiment027_publish.py --graphs G1 G11 G14 --push

# Full 71-graph results:
python scripts/experiment027_publish.py --graphs ALL --push
```

The publisher automatically runs `scripts/experiment027_analyze.py` again against the current saved source bytes for exactly the requested graphs. It verifies all 16 seeds of each new method and two historical comparator methods, independently checks four-stage original signed cut values and SHA fingerprints, strict-best snapshots, every journal event and the accepted-normal trajectory. Missing/corrupt sources stop publication without pretending they are zero. **No numerical optimization, SDP/EVA, or historical experiment is rerun.**

It then constructs and verifies the per-graph ZIPs, writes per-file SHA256 manifests, copies the audited comparison to a stable Git path, stages **only files under `published/experiment027/`**, commits and pushes the existing draft PR's remote branch `origin/experiment027-angular-greedy`. It never merges. If remote push is rejected because another agent updated the branch, **do not force push**; preserve local commits, inspect and integrate changes, then push normally. If LFS quota is insufficient, preserve both local raw and generated LFS archives while arranging storage.

## Verify the remote publication

```powershell
git status --short
git log -1 --oneline
git lfs ls-files
git lfs fsck
git rev-parse HEAD
git ls-remote origin refs/heads/experiment027-angular-greedy
```

The HEAD and remote branch SHA should agree, and the LFS listing should contain the new archived graph ZIPs. For a stronger off-VM check, use a different machine to clone the branch and run `git lfs pull`, then check each ZIP's SHA256 against its recorded manifest. **Do not delete raw VM data until this read-back verification is complete.**

## Expected coverage

| Execution scope | Graphs | New method cells | New 16-start checkpoints | Selected old-method checkpoints |
|---|---:|---:|---:|---:|
| Pilot | 3 | 9 | 144 | 96 |
| Full corpus | 71 | 213 | 3,408 | 2,272 |

The independent audit covers 80 seed archives per graph, or 240 for the pilot and 5,680 for all 71. Do not infer that all 71 graphs completed from the statement “the experiment succeeded”; confirm the index and statuses. The analysis gives **feasible signed Max-Cut scores**, not global-optimality certificates. Report raw rounding apart from Flip/KL/ILS and account for differing CPU/sweep budgets.

## Agent handoff

First inspect `published/experiment027/index.json`, then the leaderboard, paired seed comparisons and manifests, before making any numerical claims. Preserve all historical files and source identities. Related documentation: [VM file paths](experiment027-vm-handoff.md), [Experiment027 research protocol](../notes/experiments/experiment027-angular-greedy.md), [Experiment025 VM protection](university-vm-experiment025-operations.md), [AGENTS.md](../AGENTS.md), and [RESEARCH_HANDOFF.md](../RESEARCH_HANDOFF.md).

**The old Experiment025 raw archives still need a separate Git LFS or durable external-storage publication.** Never overwrite them or assume the Experiment027 ZIPs contain all historical methods.
