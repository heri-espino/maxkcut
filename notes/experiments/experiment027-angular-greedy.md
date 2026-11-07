> **VERSIONED GITHUB RELEASE DATA (Nov 7, 2026):** Prefer [no-LFS publisher](../../docs/experiment027-release-no-lfs.md) `scripts/experiment027_publish_release.py` to publish the complete existing Experiment027 16-seed raw iteration histories to GitHub Releases (one SHA-verified ZIP asset per graph), with audited regular-Git analyses and source SHA manifests. Previous Git LFS publication is optional and may exceed quota. The numerical data remain VM-local until the user runs the script and confirms the remote index; never repeat EVA/old methods.

> **RESULT PUBLICATION (2026-11-05):** Never require the user to paste large VM output into chat. [Run the Git LFS publisher on the VM](../../docs/experiment027-publication.md) to independently rescore all saved partitions, include every new method's per-iteration trace and partition archive as immutable ZIPs, push accessible compact analysis and SHA manifests to PR #80, and verify remote LFS payloads. Actual publication requires user execution; no EVA/old method recomputation and no merged PR.

# Experiment027 — angular-margin and pair-targeted greedy hyperplane sweeps

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


> **CURRENT VM STATUS (Nov 5, 2026):** The user reports that the repaired full-phase G1/G11/G14 numerical pilot succeeded. The status records, all 16 archived starts per method, verified signed-cut scores and off-VM backup have **not** been independently inspected. For precise Windows paths, source-factor directories, Experiment025 archives, where new results/journals are stored, commands to verify 9 method–graph cells, and safe resume, see **[Experiment027 VM handoff and file map](../../docs/experiment027-vm-handoff.md)**. The earlier FileNotFoundError for G14 in the assumed single Experiment021 factor folder was corrected by `scripts/experiment027_from_archives.py`, with successful hosted 12-test regression. This file retains the historical failure narrative for provenance.


## November 5 Windows pilot failure: missing G14 factor in assumed single archive — FIXED IN WRAPPER

The first user-run `full` G1/G11/G14 pilot passed seven numerical regression tests but raised `FileNotFoundError` for `experiment021-complete-71-gset/eva-numba/cases/G14.json`. The prior command erroneously supplied **one factor directory** for all graphs. Experiment025 instead selects saved factors **per graph**, from Experiment021, then Experiment003, then Experiment002, depending on which archives exist. **No EVA failure is implied.** The old command's G1/G11 child processes may have completed or partially saved results; these must be preserved. The original `scripts/experiment027_angular.py` has not been changed, because its content hash is part of every new saved-seed fingerprint.

A separate `scripts/experiment027_from_archives.py` launcher now searches the original VM worktree for each graph independently and **requires an exact SHA256 match to Experiment025's complete 16-start GW baseline** before launching any numerical work. It also verifies the original signed graph file checksum, preflights *every requested graph*, and routes Python imports to this Experiment027 worktree's `src` rather than the previously observed editable `maxkcut-exp026` installation. It never solves EVA or reruns a historical method. All three algorithms still run with 16 matched GW starts and preserve prior completed seed checkpoints. If an exact factor is missing, stop and audit the archive rather than substituting another factor.

With the original worktree at `$env:USERPROFILE\maxkcut` and current directory `$env:USERPROFILE\maxkcut-exp027`, run the read-only source preflight and then the new-method-only three-graph pilot:

```powershell
Set-Location "$env:USERPROFILE\maxkcut-exp027"
git pull --ff-only origin experiment027-angular-greedy
conda activate maxkcut
python -m pytest -q tests/test_experiment027_factor_sources.py tests/test_experiment027_angular.py
python scripts/experiment027_from_archives.py --graphs G1 G11 G14 --workers 3 --dry-run
python scripts/experiment027_from_archives.py --graphs G1 G11 G14 --workers 3
```

The wrapper checks all 16 starts for each new algorithm by default and reuses fully verified old Experiment027 seed archives. After independently verifying the pilot, the all-graph command is `python scripts/experiment027_from_archives.py --graphs ALL --workers 8`; it may be long-running. `--seed 0` is an optional single-start diagnostic. Results remain under the separate Experiment027 result root, with per-graph launcher diagnostics under `launcher-logs/`. Do not run multiple numerical launchers simultaneously or on top of active VM jobs. Before pulling, ensure no numerical process uses the new Experiment027 worktree and preserve its generated results; never pull or clean the old Experiment025 worktree.

---



**Status (2026-11-05):** New algorithms, tests and launcher prepared on a separate branch. **No production numerical run on the university VM has been initiated or claimed.** This is a prospective, append-only extension of Experiment025, not a repeat of its 20,448 completed runs. Experiment026 is separately reserved for the existing long-run sweep PR #77.

## Objective and experimental hypotheses

Given a canonical original signed graph with adjacency W, immutable n-by-p EVA vectors Y, and the exact same 16 GW unit normals from Experiment025, compare three *additional* optimizers of the original signed Max-Cut objective:

C(x) = sum_{i<j} W_ij * 1[x_i != x_j],  x_i = 1[Y_i dot r > 0].

1. **angular-margin-greedy:** rank candidate vertex boundaries by increasing angular margin delta_i = asin(|Y_i dot r|/||Y_i||), construct the shortest tangent to each selected boundary, exactly sweep along each candidate semicircle, then accept the best realized cut.
2. **angular-gain-greedy:** rank vertices by max(0, g_i)/(delta_i + 1e-3), where g_i = s_i sum_j W_ij s_j and s_i=1-2x_i. An improvement must be demonstrated by the full signed-cut sweep, not inferred from g_i. Ties use lower delta then vertex index; if no positive gains, the ranking falls back to smaller margins.
3. **angular-pair-greedy:** rank pairs from a pool of high-priority vertices, construct the projection of r onto the common orthogonal complement of two selected EVA vectors (if defined), exactly sweep in the associated great-circle direction, and accept the best realized cut. Pair targeting is not an unrestricted two-vertex flip or proof of optimality.

All methods are deterministic given the saved GW normal, graph, factor, and settings. Each greedy outer iteration evaluates **up to K** tangent candidates by calling the frozen `optimize_semicircle` routine, retains the best actual raw cut, and updates the normal only after comparing the K candidates. Early termination occurs if no candidate improves the incumbent. The full protocol uses **32 outer rounds × up to 8 candidates = up to 256 exact sweeps** per seed, close in number of sweeps to Experiment025's 250 (but not computationally equal). The small pilot uses 2 rounds × 2 candidates. There is no global or locally optimal hyperplane guarantee.

The original 16 starts, 20 positive-flip sweeps, 30 KL passes and 80 ILS-KL cycles (eight inner KL passes) are inherited **without changing** `src/maxkcut/*.py`. The pilot inherits Experiment025's reduced refiner settings. The highest verified signed cut at *any* iteration wins, not simply the final candidate. `angular-*` is *initializer family*, not a different postprocessor.

## Data, controls, and caveats

- Source of truth: original canonical `data/G*.txt`; same graph/factor checksums, original signed Max-Cut scorer, `SeedSequence([2026,graph_number,restart,tag])` downstream refinement streams and 16 GW normals as Experiment025. New methods use exactly those 16 initial directions (indices 0–15). No new EVA/SDP solve, and no rerun of the prior 18 method protocols.
- Reading Experiment025 is **read-only**. Before any `full` comparison, verify its `full/<graph>/gw-rr/status.json`, factor hashes, graph hashes, and 16 directions; when present, the new runner checks those. Missing old `full` baseline yields `null`, **not zero**. Old full raw per-seed trajectories remain VM-local according to the latest handoff; their existence on GitHub is NOT assumed. A previously saved Experiment025 pilot with only one seed is never a valid full baseline.
- Separate destination `results/gset/experiment027-angular-greedy/`. Each new `(graph,method,seed)` gets the same 4-stage `0000.npz`/`0000.json` archive and `progress/0000/events.jsonl` journal as Experiment025; each strict new best partition is saved as an immutable `best_*.npz`. The complete *accepted normal trajectory*, its cut scores, and iteration numbers are saved in `accepted_normals.npz`. Every attempted candidate path is recorded with its parent iteration, vertex id(s), angular margins, individual gains, selected arc angle, realized cut, and numerical validity. `status.json` tracks progress and best of all verified completed seeds.
- Old code under `src/maxkcut/` is not changed, because Experiment025 fingerprints the entire `src/maxkcut/*.py` set. New code and tests live in `scripts/` and `tests/`.
- A seed is skipped only if its four-stage archive can be independently checked against the original signed graph and fingerprint. **Partial journals fail closed** instead of being overwritten or silently replayed. To recover a partial seed, inspect its journal and saved partitions; preserve a backup before any manual recovery.
- One worker runs one graph at a time; different graph folders may run concurrently. Limit numerical library threads to one in each worker. Running this on the shared university VM must not overlap with any active Experiment025 or Experiment026 process. Neither `git checkout` nor `git clean` is safe in the original VM worktree containing unbacked raw results.

## Validation before production

On an idle Windows 11 VM, in a **separate** clean worktree of the Experiment027 branch, with the existing Conda environment and read-only access to the original graph data, frozen factor folders, and Experiment025 result directory:

```powershell
conda activate maxkcut
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
python -m pytest -q tests/test_experiment027_angular.py
python scripts/experiment027_angular.py run --dry-run --graphs G1 G11 G14 --method all --phase pilot
```

Use `--factor-dir PATH` when the factor archive is in the original VM worktree, and `--data-dir PATH` for read-only canonical `data/`. Use `--baseline-root PATH` to point to the VM's existing Experiment025 raw result root. **Never copy or move the VM's working result tree solely to run a pilot.** A new worktree needs its own `results/gset/experiment027-angular-greedy/` output path.

Perform a small, new-method-only test:

```powershell
python scripts/experiment027_angular.py run --graphs G1 G11 G14 --method all --phase pilot --seed 0 --workers 3 --factor-dir 'C:\path\to\original\saved-factor-folder' --data-dir 'C:\path\to\original\data' --baseline-root 'C:\path\to\original\results\gset\experiment025-gw16-bestcut-quality'
```

The explicit `--factor-dir` is a single factor source folder and must contain matching `cases/Gxx.json` and `factors/Gxx.npz` for **every requested graph**. If source factor archives differ across graphs, either use `select_factor` in the main original repo or group graphs by their source folders. Read-only factor provenance is mandatory.

After the pilot passes independent score and partition audit, and after the original Experiment025 VM archive is backed up, run the new methods only (16 matched starts per new method):

```powershell
python scripts/experiment027_angular.py run --graphs G1 G11 G14 --phase full --method all --workers 3 --baseline-root 'C:\path\to\original\results\gset\experiment025-gw16-bestcut-quality'
python scripts/experiment027_angular.py summary --phase full
```

Expand to all 71 `--graphs` explicitly once storage, factor paths, CPU and memory have been checked. Do not launch the full 71×3×16 workload until the small full-refinement pilot is verified. The current code's `--workers` parallelizes **graphs only**, to avoid concurrent writes for the same shared GW-direction folder or per-method seed. Running the same graph in two processes or two terminals is not supported.

## Analysis and decision criteria

**Primary:** Best of 16 independently refined final scores per graph, paired with Experiment025's verified GW-RR and Hyperplane Sweep results. The same original graph, factor hash, and 16 GW normals are required for each comparison. Report equal-graph mean relative improvement and graphwise win/tie/loss counts, with graph-cluster bootstrap confidence intervals or paired signed-rank tests. A published mean alone does not establish unique superiority.

**Stage ablations:** Compare raw 16-start cuts, Flip, KL, ILS trajectories, best-so-far profiles, and CPU time/evaluations. Report success rates on different graph topologies and signed versus nonnegative edge weights; selection by the best of many more candidates is a computational advantage, not automatically a smarter proposal policy. For fair cost comparisons, use matched exact-sweep counts and measured wall time. Early-stopping iterations must be reported, not padded as if completed.

**Failure criteria:** Input graph/factor checksum mismatch, changed Experiment025 GW normals, missing verified baseline, incomplete seed, or inconsistent journal/partition score must be reported; no substitution or silent recomputation. If old data are missing locally, run only the new method and mark historical pairing pending. Do not label a new feasible cut as a certified global optimum.

## Next steps / independent agent handoff

1. Review PR and run hosted toy-graph tests; separately validate one G1 pilot on the VM with verified source factors.
2. Inspect archives and compute original signed cuts independently; audit accuracy of accepted normals against labels.
3. Finish all 16 starts on three pilot graphs only, then examine whether full 71-graph execution is affordable, and confirm off-VM backup exists before heavy computation.
4. Publish compact analysis, with complete reproducibility metadata; archive large raw journals via Git LFS or documented external storage with checksums. Do not leave irreplaceable results solely on the VM.

**Implementation:** `scripts/experiment027_angular.py`; tests: `tests/test_experiment027_angular.py`. Historical code and Experiment026 PR #77 are unchanged.
