# Experiment009 — Multiple longest MST edges + two-stage Max-Cut refinement

## Question

Could the poor G63 one-edge MST scores (best 21,055) be mainly caused
by requiring each Max-Cut group to be an entire connected subtree?
We remove several of the **longest** edges from the same original
full-dimensional EVA-derived MST, allow the disconnected components
to join EITHER of two binary Max-Cut groups, and optimize the original
signed-weight Max-Cut again.

Current G63 benchmark to beat: **26,682**, observed from the user's
1,000-restart K-MeansNM-Flip_EVA experiment008 (20 flip sweeps).
This is a feasible observed score, not a proven graph optimum.

## Three stages and mathematical invariants

1. Reuse experiment005's EXACT complete-graph MST on the original
   full p-dimensional EVA factor, with **no PCA**. Do not rerun EVA,
   the convex SDP or the geometric MST.
2. For q in [1,2,4,8,16,32,64,128], remove the q longest geometric
   edges, breaking distance ties by tree child vertex ID.
   This creates q+1 forest components F_1,...,F_{q+1}. It does **NOT**
   imply q+1 Max-Cut labels. Assign these components to only TWO
   labels by optimizing the **contracted signed weighted graph**:
       W_ab = sum_{u in F_a,v in F_b} w_uv   for a != b
       max_{z_a in {0,1}} sum_{a<b} W_ab * 1[z_a != z_b].
   Within-component signed edges never cross before vertex refinement.
   Scoring a contracted partition equals the ORIGINAL graph Max-Cut
   objective EXACTLY.
3. Once per (graph, metric, q), refine the winning coarse partition
   using the repository's signed-weight greedy vertex-flip search on
   the original graph. Now individual vertices may change labels,
   and the final labels need **not** be connected MST subtrees.

At each new q, the first coarse trial inherits the previous q's best
coarse vertex partition. Because removing more tree edges only splits
existing components, this remains an exactly feasible coarse partition.
Additional independent random component assignments are optimized
with signed-weight sparse positive-gain component flips.

    best_coarse_cut(q_next) >= best_coarse_cut(q_previous)
    refined_cut(q) >= best_coarse_cut(q)

The finalized refined scores need not be monotone across q, because
each q runs its own vertex-level local search. They are feasible
heuristic cuts, not certified Max-Cut optima.

## Experimental design and controls

- Three metrics: L1, L2, Linf, each with ONE fixed saved original-space
  MST; one MST construction is *not* repeated 1,000 times.
- q schedule default 1,2,4,8,16,32,64,128; q must not exceed n-1.
- q=1 is exactly the longest-edge baseline (one deterministic coarse cut).
- q>1 uses the previous coarse best plus 20 independently seeded random
  binary component assignments, each improved by up to 20 coarse
  signed-weight flips sweeps.
- Retain the best *actual signed graph cut*, then perform one 20-sweep
  vertex-level refinement. Its visitation order is seeded, so unlike
  the tree construction, final local search can vary with seed.
- Every coarse trial partition and final refined partition is archived
  and independently rescored. Each q has its own immutable archive,
  settings fingerprint, trial JSONL, case JSON and verifiable resumability.
- Trees are read and verified from experiment005/006, not recalculated.
  Missing or ambiguous factor archives are explicit errors.
- q is a hyperparameter: select by a predeclared training graph set or
  evaluate all values descriptively. Do not report the best q over 71
  G-set graphs as an unbiased generalization estimate.
- Compare the same signed G-set graph against the 26,682 G63
  K-MeansNM-Flip_EVA feasible cut. For fair comparisons on more graphs
  align optimization/rounding compute budgets and seeds, reporting
  full distributions where random starts differ.

## Windows PowerShell pilot, G63

First preserve experiment008 and existing MST result files on your
currently active branch, and ensure the Git push completes:

    git add .
    git commit -m "save: experiment008 rounding and experiment006 MST results"
    git push

Then switch after any active experiment process has finished:

    git fetch origin
    git switch --track origin/experiment009-mst-multiedge-refine

If the branch already exists locally, use:

    git switch experiment009-mst-multiedge-refine

If earlier G63 result files are absent after the switch, the results
were committed to a different branch. Merge that branch into this
experiment009 working branch BEFORE running, without overwriting
existing user files or rerunning costly EVA/MST computations.

Run fast correctness and archive tests:

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_mst_multicut.py -q

Run the full G63 pilot using all three exact, already saved MSTs:

    python -m maxkcut.eva_mst_multicut_cli --graphs G63 --metrics l1 l2 linf --q-values 1 2 4 8 16 32 64 128 --coarse-restarts 20 --coarse-sweeps 20 --vertex-sweeps 20 --seed 2026 --mst-dir results/gset/experiment005-eva-mst --output-dir results/gset/experiment009-mst-multiedge

The command prints all 24 cuts (3 metrics x 8 q choices) and stores
summaries at results/gset/experiment009-mst-multiedge/summary.csv.

For a quick check with fewer trials:

    python -m maxkcut.eva_mst_multicut_cli --graphs G63 --metrics l2 --q-values 1 2 4 8 16 --coarse-restarts 5 --coarse-sweeps 20 --vertex-sweeps 20 --seed 2026 --mst-dir results/gset/experiment005-eva-mst --output-dir results/gset/experiment009-mst-multiedge

Distinct settings do not overwrite previous attempts; settings
fingerprints identify immutable results. The larger run with 20
restarts and the smaller 5-restart pilot create separate archives.

After G63, expand to any graphs already having saved MSTs:

    python -m maxkcut.eva_mst_multicut_cli --graphs ALL --metrics l1 l2 linf --q-values 1 2 4 8 16 32 64 128 --coarse-restarts 20 --coarse-sweeps 20 --vertex-sweeps 20 --mst-dir results/gset/experiment005-eva-mst --output-dir results/gset/experiment009-mst-multiedge --continue-on-error --push --push-branch experiment009-mst-multiedge-refine

The --push option triggers ONE Git commit/push after all graphs.
Errors, including missing MST trees, are saved under errors/.
All results remain in the repo, so ephemeral VMs can be discarded
only once the final Git push has succeeded.

## Interpretation

If multi-edge forest starts plus vertex flips approach or exceed
26,682 on G63, this would suggest that the **one-tree-edge partition
restriction**, not geometric distances alone, was the main limitation.
But any successful final cut might be due mostly to unconstrained
local vertex flips. Compare:

- stage-1 one longest-edge cut,
- best contracted component cut at each q,
- final refined cut after vertex flips,
- original GW-Flip and K-MeansNM-Flip baselines.

Only a controlled comparison can identify the effect of MST geometry.

No new numerical G-set results are claimed in this note until the
user runs the experiment and saves its verified artifacts to GitHub.
