# Experiment025 — GitHub Actions on PCAG103A-13

The scientific objective is the highest verified original signed Max-Cut ever
found at ANY initializer, Flip, KL or ILS iteration. All methods receive 16
common GW starting directions. Reuse archived EVA factors; no new EVA solves.

## One-time runner setup

GitHub automatically gives a self-hosted runner OS/architecture labels such as
self-hosted, Windows, X64. The runner NAME is not automatically a label.
Therefore, open the [runner settings](https://github.com/heri-espino/maxkcut/settings/actions/runners/21).
If missing, create/assign the custom label **PCAG103A-13** under Labels. It must
be assigned to this runner; otherwise targeted jobs remain queued even when
the runner status is Active. The workflow checks RUNNER_NAME and refuses to
compute on any other machine. The connected GitHub plugin cannot inspect or
change runner labels or dispatch new workflow runs.

Also verify Windows runner has the existing maxkcut Conda environment, Git,
Git LFS, reliable persistent disk and GitHub Actions token write access to
contents and LFS objects. The execution script automatically activates Conda,
configures native DLL paths, verifies reference table and source-code hash,
and commits results to the dedicated experiment025-results branch.

## Four workflow launch modes

Open [Experiment025 GitHub Actions](https://github.com/heri-espino/maxkcut/actions/workflows/experiment025-quality-first.yml).
Select branch main (not the historical Experiment025 branch), then:

1. Pilot: choose scope = pilot. The workflow creates 54 graph-method jobs
   (G1, G14, G11 times 18 methods). Each job independently refines all 16
   shared GW seed starts under pilot settings. Do not use pilot numerical
   cuts as the final full-budget scores.
2. Full: choose scope = full-batch, batch = 0. Each batch contains 71 graphs
   x 3 methods = 213 independent jobs, under GitHub's 256-job matrix limit.
   After it completes, run batches 1, 2, 3, 4, 5 in order, ONE AT A TIME.
   Six full batches cover 71 x 18 = 1278 graph-method cells and
   1278 x 16 = 20448 independently refined seed trajectories. Batch
   membership is recorded in scripts/experiment025_actions_matrix.py.
3. Resume a single method/graph: choose scope = single and specify graph,
   method and phase. Completed verified seeds are reused. A changed source
   hash aborts instead of mixing different algorithm versions.
4. Final audit: choose scope = coverage after all six batches pass. The
   command verifies 1278 complete method cells with 16 full refinements each;
   missing/failed cells are not assigned fake scores or included in winners.

The default checkpoint chunk_size is 4 completed seeds. Every meaningful
algorithm iteration is logged locally; on a strict new best, save a compressed
binary partition. Each finished four-seed block is committed to GitHub and
large NPZ/JSONL outputs use scoped Git LFS.

## Operational qualifications

- The actual runtime does not affect method ranking. A single Actions job
  nevertheless has a platform maximum of 360 minutes; a slow graph-method
  may need a retry after its most recent pushed seed checkpoint.
- The six workflows must be launched sequentially. The shared Actions
  concurrency group allows only one active run and can replace a pending
  workflow if too many are submitted together.
- The main GitHub commit SHA at dispatch is recorded as source identity,
  and every result is fingerprinted against all optimizer Python files.
- GitHub Actions must have sufficient LFS storage and permissions.
  A failed push stops the job rather than pretending remote preservation.
- The final result will be the BEST cut observed anywhere in each
  seed trajectory, compared to GW-RR and the archived feasible best-known
  references, grouped by graph topology. Runtimes are provenance only.
- Once the full dataset is verified, create and merge a separate results PR
  into main. The earlier EVA factors and Experiment021-024 remain unchanged.

## GitHub CLI alternatives

Run these commands only from a terminal with authenticated GitHub CLI:

    gh workflow run experiment025-quality-first.yml --ref main -f scope=pilot -f batch=0 -f graph=G11 -f method=gw-rr -f single_phase=full -f chunk_size=4

    gh workflow run experiment025-quality-first.yml --ref main -f scope=full-batch -f batch=0 -f graph=G11 -f method=gw-rr -f single_phase=full -f chunk_size=4

For the complete study, repeat full-batch with batches 1,2,3,4,5 only after
the previous batch has finished, then choose scope=coverage.
