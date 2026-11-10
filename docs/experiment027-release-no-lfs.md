> **2026-11-08 — Full Experiment027 archived; draft-release lookup recovery:** The user-supplied Windows VM log shows all 71 graph archives were packed through G81, followed by `RuntimeError: Release creation did not become visible`. The old publisher created a **draft release** and then incorrectly queried the by-tag endpoint, which may not find drafts. The updated `scripts/experiment027_publish_release.py` retrieves the existing draft from the **authenticated, paginated release listing**, resolves by numeric ID, and paginates uploaded asset listings. It reuses SHA-addressed ZIP archives and existing upload assets; it does not rerun EVA, optimization, or historical 18 methods. The draft and Git analysis publication remain **unverified** until the VM run completes. 

**Resume from the correct Experiment027 worktree** (never the protected `maxkcut` checkout):

```powershell
Set-Location "$env:USERPROFILE\maxkcut-exp027"
conda activate maxkcut
git status --short
git fetch origin experiment027-angular-greedy
git merge --ff-only origin/experiment027-angular-greedy
python scripts/experiment027_publish_release.py --graphs ALL
```

Stop and preserve results if the fast-forward merge is rejected. A directory named `trendestimation-PC08_add97svd_20261105_025108` and process filter `experiment011` belong to another project, **not Maxkcut**; the pasted `Set-Location` failure did not leave `maxkcut-exp027`. GitHub documentation: [List releases](https://docs.github.com/en/rest/releases/releases#list-releases) and [Get a release by tag](https://docs.github.com/en/rest/releases/releases#get-a-release-by-tag-name).

> **Nov 8, 2026: local VM branch compatibility.** The 71-graph preflight recognized all **3,408 new seeds** totaling **393,169,236 uncompressed bytes**. The only observed block was an overly strict publisher branch-name check: the VM uses a separate, important **`exp027-local`** branch. The publisher now accepts it safely, but insists the current remote `experiment027-angular-greedy` branch is its ancestor before uploading or committing. Do not switch or rename `exp027-local`. First run the commands below in that same worktree (this changes only the exp027 worktree, never the older `maxkcut` checkout):

```powershell
Set-Location "$env:USERPROFILE\maxkcut-exp027"
conda activate maxkcut
git status --short
git fetch origin experiment027-angular-greedy
git merge --ff-only origin/experiment027-angular-greedy
python scripts/experiment027_publish_release.py --graphs ALL
```

If the fast-forward merge is refused, **stop; do not force, reset, rebase or clean the VM worktree**. The new publisher performs a second fetch/ancestry check before any expensive verification/upload. Its final push is a normal, non-forced push to `origin/experiment027-angular-greedy`. The previous attempt stopped before the audit/Release upload, so it did not publish numerical data.

# Experiment027 publication WITHOUT Git LFS — GitHub Releases

**Current preferred workflow (2026-11-07).** The user wishes to preserve all Experiment027 numerical history in GitHub without relying on Git LFS quota. The repository itself already contains substantial data; storing the raw binary histories as regular Git blobs would permanently enlarge clones and may violate file/push limits. **GitHub Release assets are hosted by the same repository but do not increase ordinary Git history or use Git LFS.** The small audited comparisons, per-file SHA256 manifests and a stable index remain ordinary committed Git files.

## Storage and source-of-truth

- **Experiment027 original VM raw output (immutable input):** `C:\Users\175199\maxkcut-exp027\results\gset\experiment027-angular-greedy\full\Gxx\`. Contains original per-seed NPZ/JSON partitions, strictly improving best partitions, full JSONL attempts and accepted normal trajectories. Do not delete, alter or re-run these archives.
- **Historical Experiment025 baseline and EVA factors (immutable input):** `C:\Users\175199\maxkcut\results\gset\experiment025-gw16-bestcut-quality\` and the per-graph archived EVA factor sources. Do not recompute EVA or any historical refinement.
- **Temporary local ZIP staging:** `C:\Users\175199\maxkcut-exp027\results\gset\experiment027-release-staging\archives\`. Git-ignored; do not treat this as a remote backup. Files are SHA-identified to allow safe resume.
- **GitHub Release:** `https://github.com/heri-espino/maxkcut/releases/tag/experiment027-data-... `. Stores one full immutable ZIP per completed graph, up to 71 assets. **No LFS used.**
- **Ordinary Git:** `published/experiment027/latest_release.json` points to `published/experiment027/releases/<release-tag>/index.json`. Each versioned folder contains `analysis/` (five audited reports) and `manifests/` (SHA-256 for every raw member and remote ZIP). Future agents should read `latest_release.json` first.

The GitHub Releases documentation permits up to **1,000 assets per release**, **each under 2 GiB**, and states no total-size or bandwidth limit. The uploader refuses assets above 1,900 MiB as a safety margin. For a graph exceeding this size, stop and implement a documented split rather than bypassing the limit. GitHub's current standard-Git single-object limit is 100 MiB, with an enforced 2 GiB push limit; storing massive raw archives directly in Git history is not a sustainable backup strategy.

## Run from the Windows 11 VM

Use **PowerShell**, after confirming the numerical solver and all other experiment writers are idle.

```powershell
Set-Location "$env:USERPROFILE\maxkcut-exp027"
conda activate maxkcut

git status --short
git pull --ff-only origin experiment027-angular-greedy

gh --version
gh auth status
```

If `gh` is missing, install the official GitHub CLI with `winget install --id GitHub.cli -e`, reopen PowerShell and run `gh auth login`. The same GitHub account must have permission to create releases and push to this repository. **Never put a personal access token in the repository, chat or command-line arguments.**

**1. Preflight only (no mutation, optimization or upload):**

```powershell
# If the completed study was the G1/G11/G14 pilot:
python scripts/experiment027_publish_release.py --graphs G1 G11 G14 --dry-run

# If ALL 71 graphs were completed:
python scripts/experiment027_publish_release.py --graphs ALL --dry-run
```

**2. Run exactly one upload command:**

```powershell
# Pilot, 3 graphs = 144 new seeds:
python scripts/experiment027_publish_release.py --graphs G1 G11 G14

# Entire study, 71 graphs = 3,408 new seeds:
python scripts/experiment027_publish_release.py --graphs ALL
```

This automatically runs the **independent saved-result audit** before packaging. The audit verifies complete 16-start methods, signed original graph scores, historical GW/Hyperplane Sweep comparators, partition checksums, every journal event and normal trajectory. **It never runs an optimizer or EVA.** If any status or provenance check fails, the publication stops rather than uploading questionable results.

The publisher builds content-addressed ZIPs containing **all** per-seed raw files, creates a draft Release, uploads each asset and independently compares GitHub's *server-computed* asset SHA256 to the local file. Existing assets with identical checksums are reused on rerun; mismatches fail closed. It publishes the release after all assets verify, then commits and pushes only the small index/manifests/analysis files to the Experiment027 draft PR branch. **No merge, no Git LFS and no force push.**

A failure partway through upload is recoverable: rerun the same command. Already uploaded and verified Release assets are skipped. Do not delete source directories or reset the worktree. If the branch push is rejected due to concurrent agent changes, keep the uploaded Release and local commit, fetch and reconcile legitimately before retrying; never force-push.

## After the upload

Open:

- [GitHub Releases](https://github.com/heri-espino/maxkcut/releases) — find the versioned Experiment027 data Release.
- [Stable result index](https://github.com/heri-espino/maxkcut/blob/experiment027-angular-greedy/published/experiment027/latest_release.json) — discover the exact Release tag, graph coverage, and detailed index.
- [Draft PR #80](https://github.com/heri-espino/maxkcut/pull/80) — source code, read-only audit and compact numerical comparisons.

On the VM, verify:

```powershell
git status --short
git log -1 --oneline
git rev-parse HEAD
git ls-remote origin refs/heads/experiment027-angular-greedy
Get-Content .\published\experiment027\latest_release.json
```

For strongest off-VM validation, download a sample asset from another computer with `gh release download <tag> --repo heri-espino/maxkcut --pattern 'G11-*.zip'` and check its SHA256 against the committed manifest. A GitHub Release uploaded and SHA-verified by GitHub is an off-VM copy, but independent download verifies retrieval as well.

**Distinguish scopes:** The user reported successful numerical experiments; no exact local completion/coverage or audited scores have been observed directly through this conversation. If only the pilot exists, publish pilot only. The full historical Experiment025 *all-18-method raw archive* remains separate and needs its own durable backup; Experiment027 releases include new method traces and independently verified comparison scores, not every historical run.

**Historical note:** [The earlier Git LFS publication runbook](experiment027-publication.md) remains for provenance, but **use this GitHub Release workflow when LFS quota is insufficient**. Do not invoke the old `scripts/experiment027_publish.py --push` by mistake.
