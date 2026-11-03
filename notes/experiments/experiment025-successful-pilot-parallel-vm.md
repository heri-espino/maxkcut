# Experiment025 — audited successful pilot and direct parallel Windows VM handoff

**Audit date:** 2026-11-03 UTC. **GitHub run:** https://github.com/heri-espino/maxkcut/actions/runs/41570037856; linked job https://github.com/heri-espino/maxkcut/actions/runs/41570037856/job/125701033595. **Source commit:** 98dfa134d4203af83d99009b562c1b91da408c4b. **Numerical source SHA256:** 99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad. **Results branch:** experiment025-results, successful pilot final head 9dafbf02fff8d6819275b043510ecff404a63dbe.

## Verified conclusion and actual coverage

The user-triggered workflow started 2026-11-03 03:05:08 UTC and completed with success at 03:53:42 UTC (48 minutes 34 seconds). The 63 successful jobs comprise one hosted validation, one university Windows preflight, six hosted matrix-preparation jobs, **54 university numerical jobs**, and one university report job. Each numerical job was an individual graph-method combination on G1 (random, positive), G14 (planar-generator family, positive) or G11 (toroidal, signed) for **only the first GW seed** (user-facing seed 1, internally index 0).

All 54 graph-method status files record status=partially_refined, starts_completed=1, completed_seed_indices=[0], and all_starts_independently_refined=false. The published pilot_seed_progress.json confirms 54/54 complete matched one-seed cells, 3/3 fully paired pilot graphs, zero two-seed prefixes, and final_full_coverage=false. No full/ output directory existed on the audited results branch. Thus the run was an **operational pilot**, not a complete 16-seed pilot, not the full-budget 71-graph benchmark, and not a new EVA solve.

The durable output lives on the **experiment025-results branch**, NOT in ephemeral Actions artifacts and NOT yet on main:
- results/gset/experiment025-gw16-bestcut-quality/analysis/pilot_seed_completion.csv (54 rows, raw signed best-ever scores)
- results/gset/experiment025-gw16-bestcut-quality/analysis/pilot_seed_progress.json and pilot_seed_progress.csv (paired-prefix reports)
- results/gset/experiment025-gw16-bestcut-quality/pilot/G*/METHOD/0000.json and 0000.npz (signed stage scores and verified partitions)
- pilot/G*/METHOD/progress/0000/events.jsonl (every meaningful observed iteration), best.json, best_000001.npz etc. (versioned strict-best improvements), identity.json; plus method/status.json, graph/statuses/METHOD.json and source/factor SHA provenance.

Pilot best-ever signed cuts, including each method's short-setting refinement:

| Graph | Refined GW-RR seed 1 | Best of 18 pilot methods | Pilot leaders |
|---|---:|---:|---|
| G1 | 11,543 | 11,621 | cutaware-kmeans, control-tabu |
| G14 | 3,023 | 3,050 | control-paired |
| G11 | 548 | 554 | control-paired, gw-threshold, multigroup-k8, control-tabu |

These **do not rank the full-budget methods**. Example G1/GW-RR seed 1: Round 11256 -> Flip 11452 -> KL 11533 -> BEST EVER 11543 (ILS cycle 1), with 17 journal events and nine persisted strictly improved NPZ partitions. The signed partition checksum and source/factor identity are in the individual JSON.

## Why 48 minutes? Repeated GitHub Actions orchestration

The reusable workflow experiment025-seed-shard.yml has strategy.max-parallel: 1; six method shards execute sequentially. For each of 54 jobs, the one physical PCAG103A-13 Windows runner repeatedly: checked out source, fetched main/results branches, switched from detached HEAD to the dedicated results branch, located Conda, ran source/reference preflights, executed one small numerical seed, created Git commit, uploaded Git LFS snapshots, pushed, and cleaned up checkout. G1/GW-RR's actual Python wall_seconds_this_invocation was ~1.25 s, while its complete Actions job occupied ~41 s. Other numerical job durations were generally ~37–41 s. Do not confuse job duration with numerical optimization duration. No direct parallel speed-up has yet been benchmarked, so do not claim linear 32x improvement.

**Verified university environment:** PCAG103A-13, Windows 11, Windows PowerShell 5.1, existing Conda environment maxkcut, installed Git and Git LFS 3.7.1. The tested job logged NumPy 2.5.4 and SciPy 1.18.1. PowerShell 7/pwsh was NOT on the Actions PATH. Do not install or upgrade institutional software without authorization.

## Research contract: preserved numerical experiment

This campaign solves **binary signed Max-Cut** on the 71 canonical original G-set graphs. Original signed edge objective:

    CUT_g(x) = sum_{(i,j) in E_g} w_ij * 1[x_i != x_j].

The original GW random hyperplane (gw-rr) is the **primary baseline**. For each method and each graph, share the same 16 source-pinned GW directions across methods, and separately refine **all 16** starts through each meaningful initializer iteration, Flip (up to 20 sweeps), KL (up to 30 passes), and ILS (up to 80 outer cycles with up to eight inner KL passes). Selection is the BEST VERIFIED feasible cut visited at ANY iteration, even if the final walking state is worse. Distinguish raw GW, raw method initialization, Flip, KL, and final best-ever. The legacy Final partition label denotes best-ever and not necessarily the last accepted iteration. Every meaningful iteration's signed cut score is appended to JSONL and each new strict best saves a versioned binary partition with SHA. Runtime is recorded for audit, not used in rankings.

The 18 protocols are gw-rr, control-paired, gw-threshold, cutaware-kmeans, hyperplane-sweep, multigroup-k4, multigroup-k8, em-hard, em-cem, em-gibbs, mst-l2-longest, mst-l2-balanced, mst-l2-diameter, mst-l2-bestcut, mst-l2-multicut, mst-l1-bestcut, mst-linf-bestcut, control-tabu. The deterministic MST protocols retain a GW starting incumbent and are correctly called GW-anchored hybrids, NOT 16 independent random MST searches. Matched GW starts do not mean equal computation budgets.

**No new SDP or EVA solves.** The completed Experiment021 archived usable EVA factors for 71/71 graphs (G1–G67, G70, G72, G77, G81). Experiment025's run_method calls select_factor and _factor_and_case to load frozen factors and original signed graph edges. EVA is an approximate nonconvex factorization, not a certified convex SDP optimum or upper bound. The full design is **71 x 18 x 16 = 20,448 independent full-quality seed refinements**.

## Direct Windows/Conda execution, with graph-parallel workers

The original source-preserving direct script is scripts/experiment025-vm.ps1. The companion scripts/experiment025-parallel.py adds 1..32 graph-level concurrent Python workers with **one serialized Git checkpoint publisher**. These two operational files are in scripts/, outside the src/maxkcut/*.py source fingerprint, and do not change mathematical search, GW directions or previous pilot archives. The existing serial behavior is unchanged for -Workers 1.

Start in your **interactive Windows PowerShell 5.1 console** in the local cloned repository. Ensure no other Actions workflows, agents, PCs, or processes are currently writing experiment025-results. Work from a clean checkout; Git credentials and Git LFS must be configured. The PowerShell wrapper discovers the existing Conda maxkcut environment, sets its Library/bin DLL search PATH, uses its precise python.exe, sets OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=OMP_NUM_THREADS=NUMBA_NUM_THREADS=1 for every worker, validates source SHA and data, and switches safely to experiment025-results.

    git fetch origin experiment025-results
    git switch experiment025-results
    git pull --ff-only origin experiment025-results
    git lfs install

    # Pilot only: G1, G14, G11, so at most 3 parallel graph workers.
    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase pilot -Workers 8

    # Full 71 graphs, 18 methods and 16 independent GW seeds each.
    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -AllGraphs -Workers 32

    # Lower RAM footprint, publish after every 4 fresh seeds/method:
    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -AllGraphs -Workers 8 -ChunkSize 4

    # AFTER all full methods are completed and verified:
    powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase coverage -SkipTests

    # No-compute planning only (from maxkcut Conda Python):
    python .\scripts\experiment025-parallel.py --phase full --all-graphs --workers 32 --dry-run

**Sync requirement:** Both the parallel script and PowerShell integration must be present on experiment025-results before the PowerShell wrapper switches there. They must be reviewed/ported as non-optimizer operational changes from the implementation PR; do NOT automatically merge whole branches or change any src/maxkcut files. Then verify numerical source SHA256 remains 99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad before reusing existing pilot archives.

### Worker/publisher rules

One worker owns one graph directory at a time; its methods and seeds execute sequentially, while **different graphs** can run concurrently. The parent serializes all Git operations with a lock and publishes only one completed graph's folder after each method seed checkpoint; no concurrent Git commits/pushes. -ChunkSize 16 (default) publishes after each method's up-to-16 new seeds. -ChunkSize 4 publishes in up to four blocks per method. A failure publishes any completed local checkpoint/log and aborts without changing other graph metadata. No forced push, rebase, or branch reset is performed. Remote concurrent changes cause a rejected push and fail-closed exit.

Each algorithm saves events and record-best partitions **immediately to local disk**. Remote Git publication occurs at checkpoint boundaries, not after each iteration. Thus the last active chunk may be local-only after a VM outage: inspect the worktree and recover it before switching machines. Existing completed seed archives are re-verified and reused on retry. If a crash leaves .git/experiment025-parallel.lock, first verify that no scheduler is active before removing a stale lock. Do not delete or overwrite archives, or solve EVA again. The existing src/maxkcut numerical source SHA and per-case fingerprint reject mixed code revisions.

**Memory/resource qualification:** 32 logical processors do NOT guarantee enough memory for 32 simultaneously loaded large EVA embeddings. Monitor RAM and paging, especially G77/G81. Start at 4 or 8 workers if uncertain, then increase to 32 only when measured memory headroom permits. Pilot has just 3 graph workers regardless of -Workers. This is a **parallel process scheduler**, not automatic single-algorithm multithreading; NumPy/Numba/BLAS threads are capped at one per Python worker.

## Final scientific reporting and agent handoff

Use matched **GW-RR after common refinement** as the primary reference; report raw GW separately. For each graph/method/seed preserve method/graph IDs, G-set source and graph type, signed original graph SHA, EVA factor SHA, shared normal index/hash, algorithm source SHA, original cut at all meaningful iterations, final best-seen cut, saved best partition SHA, stage cuts, last walking cut, failures/resumes, method wall time and Git checkpoint. Analyze per graph and 1..k matched seed prefixes. Don't count an incomplete arm as zero; don't pool raw cut magnitudes across different graph instances. Relative gains against GW should be interpreted carefully for signed weights, and observed best-known feasible graph cuts are not all proven optima. Distinguish method effect before/after Flip/KL/ILS, graph-topology heterogeneity, unequal computational work, and empirical performance from worst-case approximation guarantees.

Only declare complete after all **1,278** graph-method full cells have status complete_16_best_seen, 16 verified saved partitions with indices 0..15 and a full_coverage.json all_complete=true, plus independent original signed cut rescoring. Then preserve final reports, CSVs, all iterations, all improved partitions and logs on experiment025-results and only then propose publication/merge of the numerical results branch. If any coverage is missing, report it accurately.

## Previously solved operational incidents

- Original Actions job labels were wrong. Runner has only self-hosted, Windows, X64; check RUNNER_NAME=PCAG103A-13 (PR #62).
- pwsh missing on university runner; use shell: powershell / powershell.exe (PR #63).
- GitHub Actions checkout uses detached HEAD. PowerShell 5.1 returned null for git branch --show-current, causing .Trim() failures. Use scripts/experiment025-git-branch.ps1 and Get-Experiment025GitBranch; the regression was tested with an actual detached Windows PowerShell 5.1 runner (PRs #64 and #65).
- Canonicalize CRLF/LF when hashing optimizer source to match Windows and hosted Linux (PR #64).
- A green hosted test, a green Windows preflight, or a queued numerical job is **not** a successful experiment. Confirm saved, independently verifiable cut partitions and remote Git LFS pushes. Earlier failed runs #41526784028, #41534228271, #41538460202 never established completed numerical coverage.
- The successful pilot is run #41570037856. The previous successful Experiments020/021 and their 71 EVA sources remain untouched.

For original design details see notes/experiments/experiment025-seed-first.md, experiment025-vm-runbook.md, experiment025-gw16-bestcut-quality.md, AGENTS.md and RESEARCH_HANDOFF.md.
