# Experiment024 — 16 matched GW directions for all 18 method protocols

**State:** new prospective comparison. Code, pilot and full results are independent from Experiments021–023 and archived to results/gset/experiment024-paired-gw16/. Completion must be established from GitHub Actions and saved hashes, not assumed from committing source.

## Hypothesis and paired design

The older GW-screen16 policy evaluated 16 unit normals while a K-MeansNM restart used one. Experiment024 fixes the STARTING OPPORTUNITY disparity by giving every method **the SAME 16 Gaussian directions** on a given graph.

For original signed graph G and frozen nonconvex EVA rows Y in R^(n x p), let

    z_{g,s} ~ N(0,I_p)
    r_{g,s} = z_{g,s}/||z_{g,s}||
    GW_{g,s,i} = 1[(Y_i dot r_{g,s}) > 0],  s = 0,...,15.

SeedSequence([2026, graph_number, 2401, s]) is fixed and independent across s; graph and EVA factor SHA256 are validated. The 16 normals and binary GW partitions are archived once per graph, shared exactly by all methods. For each s, a candidate method receives both r_s and GW_s, and retains GW_s when its own initial candidate is weaker on the original SIGNED cut objective. The method may use independently seeded auxiliary proposals, but never an unrelated hyperplane as its first initialization.

We save **all 16 raw candidate partitions, all 16 raw signed cut values, the winning seed index, the shared-vector hashes, time and cost diagnostics**. The best-of-16 raw partition receives exactly **ONE** common Flip -> KL -> ILS refinement, and its signed cut and four-stage partitions are reverified and archived. This is NOT 16 independent complete ILS searches.

## Mapping for all 18 original Experiment023 protocols

| Algorithm family | Exact common-GW input | Subsequent operation |
| --- | --- | --- |
| GW-RR primary control | r_s and its sign partition GW_s | No candidate changes |
| K-MeansNM and separate Tabu-KL control | First centers -r_s and +r_s | Spherical Lloyd and common refinement (Tabu separate refiner) |
| GW threshold | Projections Y r_s | Signed-cut threshold optimization along same direction |
| Cut-aware K-means | First centers -r_s and +r_s | Lloyd, threshold and extra specified cut-aware proposals |
| Direct hyperplane | Exact starting normal r_s | Great-circle searches from that normal |
| Multi-group k=4 and k=8 | First center pair -r_s and +r_s | Further independent pairs and binary merges |
| Soft EM, EM+CEM, EM+Gibbs | GW_s as probability/membership anchor; no historical elite cut | Soft probabilistic updates then CEM or Gibbs |
| Seven MST edge/multicut choices | GW_s as incumbent and exact EVA-derived MST as candidate | Explicitly labeled **GW-anchored MST hybrids**, NOT independently initialized original MSTs |

**Important scientific limit:** A deterministic MST does not start from a random hyperplane. The tree itself is unchanged across the 16 seeds, so its fusion with the GW incumbent is a new hybrid algorithm. Do not report it as 16 independent randomized MST restarts.

Sixteen equal *starting directions* do NOT imply equal total computation. GW-threshold, cut-aware K-means, hyperplane searches, EM and group merges evaluate additional candidates. Cached MST preprocessing costs O(n² p). Report wall time, preprocessing and extra proposals; do not claim a controlled equal-evaluation benchmark.

## Primary comparisons and topology

1. Original signed raw cut best-of-16 and mean-of-16; preserve matched seed index s and paired raw differences against the same GW_s. By construction GW retention makes raw gains nonnegative; this is not proof of a generally superior algorithm.
2. Best raw candidate refined once with the same Flip/KL/ILS (Tabu refiner reported separately); compare final cuts against refined GW-RR and all 71 reference best-known feasible values, plus certified upper bounds where recorded.
3. Compare by 30 random, 24 planar-family/almost-planar and 17 toroidal generator classes, plus signed weight fraction, degree CV, clustering and cycle features. Use each graph as one experimental unit, not every seed as a separate graph. No new GW approximation ratio is claimed for these nonconvex EVA-derived heuristics.
4. Record incomplete, timed-out and failed graph-method cases explicitly. **Only complete 16-of-16 rows qualify for primary winner counts**.

## Reproducibility

Sources:
- src/maxkcut/experiment024_paired_gw16.py — deterministic shared starts; seeded candidates; score and partition archives; resumable graph-method runs.
- src/maxkcut/experiment024_analysis.py — complete-cell-only topology summaries.
- tests/test_experiment024_paired_gw16.py — source-independent synthetic tests.
- scripts/experiment024-windows.ps1 — Windows Conda execution, safe git checkpointing and failure logs.
- .github/workflows/experiment024-paired-gw16.yml — three-family pilot followed by full 71-graph matrix only if the pilot succeeds.

From repository root:

    python -m pytest tests/test_experiment024_paired_gw16.py -q
    python -m maxkcut.experiment024_paired_gw16 batch --graph G11 --phase pilot
    python -m maxkcut.experiment024_paired_gw16 batch --graph G11 --phase full
    python -m maxkcut.experiment024_analysis --phase full

Pilot: G1 random, G14 planar-family, G11 toroidal, all 16 starting directions, shortened EM/ILS, 120-second soft method budget, 180-second hard timeout. Full: all 71 canonical instances, all 18 methods, 16 matched directions each (20,448 raw graph-method-seed candidates maximum), one common refinement per method, 300-second soft and 420-second hard timeouts. Complete primary coverage means **1,278 verified graph-method cells**, each with 16 raw candidates and validated final partition. Missing or budget-limited cases remain visible; do not silently impute them.

The workflow uses the existing self-hosted Windows runner and the same concurrency group as Experiment023, so the studies are not launched simultaneously. Result commits do not retrigger the workflow. Merge completed PR implementation per user authorization, but any generated data arriving after merging require a separate results PR into main. No force pushes or overwriting immutable EVA factors/history.
