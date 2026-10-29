# Experiment017 — Multi-group K-MeansNM, exact binary coarsening

## Motivation and experiment objective

The user's prior best independently verified G63 Max-Cut is 26,917
from the Experiment013 K-MeansNM + Flips + KL + ILS pipeline.
27,047 is an unverified target, not a certified optimum.

Hypothesis: **Two-group K-MeansNM may converge too quickly and
miss useful multigroup structure**. Instead initialize 4, 6, 8,
10 spherical centers (optionally 12, 14, 16), cluster the FIXED
full-dimensional G63 EVA row vectors, and then merge these
small groups into exactly TWO sides to solve the original
weighted signed Max-Cut. Do NOT mistake a k-group cut for
a 2-group Max-Cut: the final solution must be binary.

No new EVA optimization, no PCA, no additional SDP solve.

## The exact group-merging combinatorial subproblem

Given fixed k-way labels g(i) in 0..k-1, aggregate signed
weights of the original undirected G-set edges:

    W_ab = sum_{i<j: {g(i),g(j)}={a,b}} w_ij , for a<b.

For a merging mask z in {0,1}^k, the original binary cut is:

    C(z) = sum_{a<b} W_ab * 1[z_a != z_b].

This identity is **exact**, even for negative G-set weights.
Edges inside one k-way group cannot contribute to the binary cut
and are therefore omitted. Because z and 1-z describe the same
cut, fix z_0=0 and enumerate all 2^(k-1) assignments, including
the trivial all-same cut. The number of NONTRIVIAL assignments:

| k groups | Distinct nontrivial binary mergers |
|---|---:|
| 2 | 1 |
| 4 | 7 |
| 6 | 31 |
| 8 | 127 |
| 10 | 511 |
| 12 | 2,047 |
| 14 | 8,191 |
| 16 | 32,767 |

Thus the k=10 coarsening is cheap enough to do EXACTLY after every
Lloyd spherical K-means iteration. Ranking a merge is cheap; the
bulk of work is clustering and optional postprocessing KL.

The group-merging optimum is **conditional on the current k-way
clustering**. It does not globally solve the 7,000-vertex Max-Cut.

## Generalizing spherical K-MeansNM

The existing K-MeansNM k=2 starts with c0 = u, c1 = -u. We
reproduce its exact RNG draw schedule, assignments, centroid
updates and best-iteration cut for k=2.

For even k>2, propose k/2 *antipodal center PAIRS* at the
first step:

    c0=u0,c1=-u0; c2=u1,c3=-u1; ...

All initial centers are unit vectors. Subsequently, every center
is updated independently by the normalized sum of its assigned
EVA vectors. Pairs are not forced to remain opposite, which would
change the standard Lloyd process. Optional 'independent' mode
draws k independent initial unit directions as a control.

This is a NEW generalization: the existing repository's
round_spherical_kmeans(..., antipodal=True) accepts only k=2.
It does NOT claim that k-way clustering itself optimizes Max-Cut.

At every Lloyd iteration:
1. Assign every full-dimensional EVA row to its highest-dot center.
2. Aggregate the original signed G-set graph into a k-vertex
   signed weighted graph by the current group assignment.
3. Exactly enumerate ALL 2^(k-1) binary merges.
4. Retain up to the top-merges best DISTINCT binary coarsenings
   ACROSS all Lloyd iterations, not only the final clustering.
5. Continue Lloyd until assignments stabilize or max_iter.
6. Run independently seeded compiled generalized KL on each
   retained coarsening; keep the best verified original-graph cut.

The source of the winning KL result may have a LOWER raw cut
than the top raw merge. Preserve separate Raw, BeforeKL, AfterKL
partitions, and save ALL top-merge CandidatesRaw/CandidatesKL
partitions so the comparison remains scientifically auditable.

For k=2, paired, top_merges=1 the best raw output must reproduce
the original K-MeansNM (without Flips/KL). The regression test
runs the two algorithms with the SAME initial NumPy RNG and graph.

## Windows PowerShell

### Preserve and switch

If running on the Experiment016 branch, FIRST save all existing
completed results, and ensure they have been pushed:

    git add .
    git commit -m "experiment016: preserve soft-EM G63 results"
    git push

Then switch to the Experiment017 branch (based on the saved
Experiment016 code):

    git fetch origin
    git switch --track origin/experiment017-multigroup-kmeans-coarsening

If you have already created its local branch:

    git switch experiment017-multigroup-kmeans-coarsening

If Experiment016 results were pushed AFTER this new branch was
created, merge the source branch so their archives remain visible:

    git merge --no-edit origin/experiment016-soft-em-bernoulli

The input source factor remains
results/gset/experiment003-eva-numba/factors/G63.npz,
alongside its completed original G63 EVA case JSON.
DO NOT regenerate EVA or alter its checkpoint.

### First, tests

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_multigroup_kmeans.py -q

Tests verify signed negative edge weights, exact enumeration
against brute force, the zero-cut trivial merge, bit-by-bit k=2
reproduction of the original K-MeansNM for matching RNG, group
mask reconstruction, repeated deterministic multi-group Lloyd
starts, and Windows-spawn parallel archive/resume.

### Pilot: 10 paired restarts across five k settings

    python -m maxkcut.eva_multigroup_kmeans_cli --graphs G63 --ks 2 4 6 8 10 --inits paired --restarts 10 --chunk-size 2 --workers 4 --max-iter 60 --top-merges 3 --kl-passes 30 --target 27047

This is 50 paired (k,restart) runs, k=10 evaluating 511
nontrivial group unions PER Lloyd iteration. Run after tests.
The top three unique coarse binary candidates of each restart
also receive KL. k=2 is a true original-method raw control,
not an arbitrarily initialized unrelated clustering baseline.

### Expanded experiment: 100 restarts and both initialization modes

    python -m maxkcut.eva_multigroup_kmeans_cli --graphs G63 --ks 2 4 6 8 10 --inits paired independent --restarts 100 --chunk-size 2 --workers 4 --max-iter 100 --top-merges 5 --kl-passes 45 --target 27047

This evaluates 1000 independently seeded clustering runs,
including the ordinary independent-centroid control, with up
to 5 distinct pre-KL candidate mergers per restart. Many
restart IDs/initial hyperplanes are paired across the k settings;
the same random seed gives the same initial first two Gaussian
rows, but the subsequent clustering is not otherwise coupled.

A fixed command can be repeated to verify and skip completed
chunks. Expanding restarts from 10 to 100 with the SAME settings,
chunk size and start index can reuse its original complete chunks;
changing max_iter, top_merges or kl_passes creates distinct
configuration fingerprints and preserves both experiments.

### Push complete results (repo-first)

    git add .
    git commit -m "experiment017: G63 multigroup KMeansNM signed coarsening"
    git push

Outputs in results/gset/experiment017-multigroup-coarsening/:

- factor-snapshots/: original EVA factor SHA-checked exact copy.
- groups/: original full k-way group assignment for every retained
  coarse candidate, integer side-merging masks, restart IDs,
  Lloyd iteration IDs and NPZ SHA256.
- archives/: independently verified original signed edge snapshot,
  all binary Raw / BeforeKL / AfterKL / CandidatesRaw /
  CandidatesKL partitions and original-graph cut scores.
- cases/: graph/factor SHA fingerprints, k, initialization,
  settings, all detailed Lloyd traces and candidate KL scores.
- trials/: complete per-restart signed cut and group-merging data
  as line-delimited JSON.
- summary.csv: per-restart, per-k raw and KL scores, selected mask,
  gap to 27,047 and trace/case paths.
- leaderboard.json: deterministic score-only ranking of k and
  initialization by best signed final cut and then mean final cut.
- errors/: failures; finished chunks remain independently verifiable.

Only independently verified original-graph binary cuts above 26,917
would supersede our best user-reported feasible G63 result. A score
of 27,047 would not certify global optimality.
