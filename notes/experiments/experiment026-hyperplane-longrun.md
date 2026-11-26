# Experiment026 — extended Hyperplane Sweep search

Status: code, references and small-graph regression tests prepared. Production computation has NOT been started on the university VM.

## Objective and mathematical mechanism

Improve feasible binary, original signed Max-Cut on selected canonical G-set instances. Use EXISTING frozen low-rank EVA factors only; DO NOT solve SDP or EVA again.

For fixed embedding vectors Y_i and a unit hyperplane normal r, labels are the signs of Y_i dot r. The inherited exact great-semicircle routine optimizes the actual signed graph objective over all realizable sign changes on an arc. It chooses among cyclic random, gain-guided, and boundary-guided tangents and never accepts a worse realized hyperplane cut.

One *batch* has 60 successive semicircle directions, 10 batches per independent restart (600 directions), then a seeded new Gaussian normal. Each graph can run for four hours, stopping only after a completed batch. Original random generation is determined by SeedSequence([2026, 26, graph_number, restart]). Users can extend an already complete run by invoking the exact same configuration again with a new --hours budget.

Optional comparison to Experiment025 uses the exact original Flip (20 sweeps), KL (30 passes), ILS-KL (80 cycles) refiner every fifth batch. This is the same downstream refinement of Hyperplane Sweep candidates, NOT an independently tested pure-hyperplane cut. The --refine-every 0 setting runs a pure hyperplane-only ablation; the default 5 prepares the quality-first search. Runtime is not equalized against other methods.

## Run on the university Windows 11 VM

First confirm that Experiment025 is idle and its full raw archive has a separate verified backup. Do not checkout, clean, reset or pull over the existing active Experiment025 worktree. Open PowerShell with the existing maxkcut Conda environment from an IDLE, clean local Git worktree:

    git fetch origin experiment026-hyperplane-longrun
    git worktree add "$env:USERPROFILE\maxkcut-exp026" -b exp026-local origin/experiment026-hyperplane-longrun
    Set-Location "$env:USERPROFILE\maxkcut-exp026"
    conda activate maxkcut
    $env:PYTHONPATH = (Resolve-Path .\src).Path
    $env:NUMBA_NUM_THREADS = "1"
    $env:OMP_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"

    python scripts/experiment026_hyperplane_sweep.py --dry-run
    python -m pytest -q tests/test_experiment026_hyperplane_sweep.py

    # Short, recoverable one-graph test before the long production run:
    python scripts/experiment026_hyperplane_sweep.py --graphs G1 --workers 1 --max-batches 1 --hours 0.25

    # Eight independent per-graph workers; each gets up to four hours:
    python scripts/experiment026_hyperplane_sweep.py --graphs G1 G9 G28 G36 G40 G42 G44 G53 --workers 8 --hours 4

If archived factors or graph data are available only in the original worktree, add the read-only options --factor-dir "C:\path\to\original\results\gset\experiment021-complete-71-gset\eva-numba" and --data-dir "C:\path\to\original\data" to *both* the pilot and the full command. The chosen factor directory must contain existing cases/ and factors/ for every requested graph. Neither path is ever written by Experiment026.

Do not start this from a GitHub Actions numerical job on the university machine: the Windows VM is shared, and long-running direct PowerShell with explicit thread controls is preferable. The hosted Actions workflow runs ONLY a tiny toy-graph smoke test, not the numerical campaign. There is no automatic background job created by committing this PR.

## Storage, crash safety and reporting

Results go exclusively under results/gset/experiment026-hyperplane-longrun in the NEW worktree. Each graph has its own exclusive active.lock (do not remove a lock unless the owner process is confirmed dead). The runner fails closed on a fingerprint mismatch or orphaned output.

The per-graph identity.json pins original graph and EVA factor hashes, the existing optimizer source hash, the new runner hash, the updated 71-reference CSV hash, random seed and configuration. checkpoint.json holds the next batch/restart, active hyperplane normal, RNG state, and an independently checked best partition pointer. Each complete batch produces traces/batch_*.jsonl with all 60 sweep outcomes and reports/batch_*.json with scores and gap-to-literature. The best labelings are saved in immutable, SHA-checked, versioned partitions/best_batch_*.npz. Optional downstream refinement uses refiner's existing per-iteration journals, best-so-far events, and exact signed-graph score checks.

Because checkpoints are finalized after all new batch files, an abrupt power outage during a write may leave orphaned files. Such files are preserved and must be manually audited, never overwritten or silently deleted. A normal command restart resumes precisely at the last completed checkpoint. Failed batches can be retried after review.

Before concluding the experiment, back up the complete Experiment026 folder to approved persistent off-VM storage and verify SHA256. Publish compact, independently rescored results, learning curves, manifests, and their reproducing scripts in the repository PR. Use Git LFS or documented storage for raw high-volume iterations and NPZ partitions. Do not leave irreplaceable raw results only on the university computer.

## Evaluation

Separate raw Hyperplane Sweep scores from those improved by Flip/KL/ILS. Compare to the new published feasible values in references/gset-literature-2026-11.csv and to the frozen Experiment025 best-of-16 scores. A higher result is a candidate new record, not proof of optimality. Report number of completed directions, restarts, wall time, the best-observed score after each batch, graph and factor hashes, and whether any candidate survives independent reference and partition validation.

The project deliberately preserves Experiment025 (71 graphs × 18 protocols × 16 directions) as a historical comparison and reuses its solved EVA factors without recomputing them.
