# Experiment025 — One matched GW seed per complete G-set round

## Objective and paired design

Each Actions workflow dispatch selects ONE user-facing GW seed number
s = 1, 2, ..., 16. That round runs all 71 frozen G-set graphs with all
18 methods, using the SAME saved-embedding GW direction for seed s on
every method of the same graph. Graphs have different embeddings, so
the numerical random vector differs across graphs. EVA factors are
loaded from already completed experiments and never optimized again.

For graph g and method m, C[g,m,s] is the BEST ORIGINAL SIGNED Max-Cut
ever observed during the method initializer, Flip, KL, or ILS for seed s.
After k fully paired rounds, compare

  B[g,m,k] = max(C[g,m,s] for s=1,...,k)

against B[g,gw-rr,k]. This curve is monotonic in k. Only declare method
winners on a graph if all 18 methods have every seed 1..k. Never count an
incomplete method as a failure or give it a fabricated zero. The highest
intermediate cut, not the last iteration, is authoritative.

## One workflow run, six internal stages

GitHub Actions limits a single matrix to 256 jobs. The 71 x 18 = 1,278
graph-method cases for one seed are divided into six sequential internal
stages of 71 graphs x 3 methods = 213 jobs each. The GitHub Actions
interface shows ONE workflow run per seed; the user does not need to
manually dispatch six method batches.

Each job runs exactly one graph, one method and one GW seed. This
avoids the six-hour timeout being shared by multiple methods or seeds.
The same physical PCAG103A-13 self-hosted Windows X64 runner processes
its jobs sequentially. This runner reports only the three built-in labels `self-hosted`, `Windows`, and
`X64`. The workflow now matches those actual labels and verifies the exact
`RUNNER_NAME=PCAG103A-13` before beginning any numerical optimization.

The optimizer accepts a 0-based seed index 0..15 corresponding to the
1-based workflow input 1..16. Each job reads any existing archives,
verifies that exact seed partition and cut, logs every meaningful
iteration, saves every strictly improving best-so-far binary partition,
and commits/pushes the new seed archive using scoped Git LFS to
the dedicated experiment025-results branch. A later seed can complete
before an earlier one without resetting the global saved method status.

## GitHub checkout and cross-platform optimizer fingerprint

GitHub Actions generally checks out an exact commit into **detached HEAD** mode. `git branch --show-current` therefore returns an empty string; `experiment025-vm.ps1` treats it as empty (not null .Trim()) and then safely creates or resumes the existing dedicated `experiment025-results` branch. Always pin numerical operations to the dispatch source SHA before switching branches. The Linux-hosted validation stage and Windows-runner computation may check out Python files with LF and CRLF line endings respectively. The optimizer source hash canonicalizes CRLF to LF before SHA256 so a byte-level platform checkout difference does not spuriously invalidate the mathematical protocol. No best-cut computations completed in the earlier run #41534228271, although Windows Conda and dataset preflight passed.

## University Windows 11 shell/Conda preflight

PCAG103A-13 currently has built-in Windows PowerShell (`powershell.exe`) but its Actions environment does not provide `pwsh`. All Windows jobs therefore select `shell: powershell`. Before any seed matrix, a single `windows_preflight` job uses `scripts/experiment025-windows-preflight.ps1` to verify runner name, Git LFS, existing `maxkcut` Conda Python, NumPy/SciPy/Numba/pytest and 71 archived reference graphs. The numerical driver then invokes the precise `maxkcut/python.exe` and sets native Conda `Library/bin` directories in `PATH`. If Conda was installed under a nonstandard university path and the Actions account cannot discover it, set repository Actions variable `MAXKCUT_CONDA_PREFIX` to its exact environment directory (ending in `maxkcut`). This is optional when automatic discovery works. No environment installation is performed by the workflows.

## Recommended sequence

1. Open the runner settings:
   https://github.com/heri-espino/maxkcut/settings/actions/runners/21
   Verify it is active/Idle and has the labels self-hosted, Windows, X64.
2. Open the new seed-first workflow:
   https://github.com/heri-espino/maxkcut/actions/workflows/experiment025-seed-round.yml
   Select main, phase=pilot, seed=1. This runs 54 short-setting
   graph-method trajectories (G1, G14, G11 by 18 methods).
3. Once the pilot succeeds, select phase=full, seed=1. The workflow
   runs the full 1,278 graph-method cells for that one seed. Check the
   six stages and saved per-seed progress output.
4. Launch full seed=2, then 3, ..., finally 16. Each round uses
   the identical per-graph GW normal across all methods; each seed's
   normal is independently sampled. One full round must finish before
   the next for straightforward diagnosis. GitHub queue:max permits
   multiple queued rounds without replacing pending ones when desired.
5. After all 16 seeds, run the original Experiment025 manual workflow
   with scope=coverage and verify every one of 71 x 18 methods has
   all 16 saved and independently auditable seeds.

The six methods batches per workflow are chained through reusable
workflow experiment025-seed-shard.yml. The prepare stage validates
the numerical integration tests and pins the code source hash.
Each graph-method seed uses the active repository history and the
existing Conda environment, without any new EVA solves. GitHub Actions
still imposes per-job and whole-run platform time limits. If one case
fails, rerun the same seed round; saved cases are reused.

After a successful round, the report stage writes descriptive
seed-prefix curves and completeness under the Experiment025 analysis
folder, including full_seed_progress.csv, full_seed_completion.csv and
full_seed_progress.json (pilot equivalents use pilot prefix). Each
report tracks the original signed best-ever cut, its gain relative to
GW-RR with matched seed counts, and the generator's topology family.
These reports are not a substitute for the final verification of
stored full partition archives.

## Reproducibility and publication rules

The new workflow is manual-only and source-pinned. Each seed is saved
with its exact GW direction index, graph and EVA factor provenance and
full optimizer source-tree fingerprint. Non-improving iterations retain
score/diagnostic entries, and each strict new best retains a versioned
binary partition. Incomplete or timed-out cases never enter the final
paired rankings. One seed round is 1,278 graph-method jobs; 16 rounds
total 20,448 full-quality trajectories. This is deliberately more job
overhead but substantially easier to recover than 16 seeds together.

Results remain on experiment025-results while the study is incomplete.
After verification, create and merge a separate numerical-results PR
into main; never overwrite prior EVA factors or Experiments021-024.
