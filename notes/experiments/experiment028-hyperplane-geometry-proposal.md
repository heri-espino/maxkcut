# Experiment028 proposal — the geometry and combinatorics of hyperplane rounding (NO local refinement)

**Status:** Research proposal only; **NO numerical experiment run**, no new algorithm implementation, no claims of measured improvements. Created 9 November 2026. Keep the name \`experiment028\` unless an existing independent branch/workstream has reserved it; verify before executing. This is a scientifically different estimand from Experiment025/027.

## Research question

> Given a **fixed**, SHA-pinned EVA/SDP vertex embedding, what can rotating and translating a hyperplane achieve **by itself**, and what do changes in its cut-score landscape reveal about the **original graph's signed, local, and global structure**?

The important experiment design decision: **NO Flip, NO KL, NO ILS, NO Tabu, NO vertex-label correction of any kind** in the primary study. Only partitions realizable by a single hyperplane in the *fixed embedding* are admissible. Recorded cuts are scored directly on the canonical **original signed weights**. If a separate optional downstream-refinement ablation is ever added, it is strictly segregated from all geometric claims and primary rankings.

**Why:** In the full 71-graph published Experiment027 audit, each of three new angular methods improved best-of-16 post-refinement quality relative to refined GW by about +0.28–0.32% across equal-weight graphs, but did not clearly outperform Hyperplane Sweep. The matched-seed **raw** angular results were on average ~16–17 signed cut units *below* historical HP raw rounding, while the common refiner nearly erased this gap. The follow-up must expose—not hide—what the rounder itself contributes. These E027 numbers are **descriptive measured facts from the archived data**, not predictions for Experiment028.

## Notation and theoretical problem

- \`G=(V,E,w)\`: original undirected, possibly signed graph, \`n=|V|\`, \`m=|E|\`; \`W\` its symmetric adjacency with \`W_ii=0\`.
- \`Y in R^(n×d)\`: fixed EVA factors, row \`y_i\` per vertex; **never resolve the SDP**. Record actual algebraic and numerical rank.
- \`r in S^(d−1)\`: rotating unit normal; \`b in R\`: signed offset along \`r\`.
- \`s_i(r,b)=+1\` if \`y_i·r>b\`, otherwise \`−1\`. At equality, implement a documented deterministic tie rule and test boundaries carefully.
- \`C(s)=sum_{i<j} w_ij (1−s_i s_j)/2\`: **original signed Max-Cut score**.

Origin-constrained hyperplane search:
\`F_0(r)=C(s(r,0))\`.
Affine search:
\`F(r,b)=C(s(r,b))\`.
The desired geometric optima are \`max_{||r||=1} F_0(r)\` and \`max_{||r||=1,b in R} F(r,b)\`. These are not generally tractable in unrestricted/high-rank dimension, and **they are not solved globally by greedy angular search**.

GW, as classically analyzed for nonnegative weights, samples a random \`r\` with \`b=0\`. Moving \`b\` away from zero changes the rounding family, so GW's 0.87856 expected-approximation result **does not carry over by default** (and does not apply directly to arbitrary signed weights).

## Lemma 1 — Exact 1D translation for fixed orientation

Fix \`r\`, let \`z_i=y_i·r\`. Except at tied values, the partition only changes when the threshold crosses one of the sorted \`z_i\`. Thus the global optimum over \`b\` for this **fixed \`r\`** is attained in one of the \`n+1\` event-free intervals. Let \`s\` be the current spin vector. Flipping vertex \`i\` changes the **original signed cut** by

\`Delta_i = C(s with s_i flipped) − C(s) = s_i * sum_j W_ij s_j\`.

This follows because only terms incident to \`i\` change, each by \`w_ij s_i s_j\`. With adjacency lists, a sweep of sorted vertices costs \`O(n log n + m)\` **after** projection (plus \`O(nd)\` projection). Tied \`z_i\` must be flipped **simultaneously** as one event group; never select a mathematically unattainable cut between coincident crossings. Verify every purported best cut from the actual binary assignment.

**Repository already has this:** \`src/maxkcut/eva_cutaware_kmeans.py::optimal_threshold\` and the historical \`gw-threshold\` method. The method should be **reused and isolated**, not presented as a novel algorithm. Check the existing exact tie handling and add explicit degeneracy tests if needed.

## Lemma 2 — Exact 1D rotation along a specified arc

Fix a unit \`r\` and tangent \`u perpendicular r\`, \`||u||=1\`. Rotate
\`r(theta)=r cos(theta) + u sin(theta)\`, within a specified great semicircle.

At zero offset, a vertex changes side only when

\`(y_i·r) cos(theta) + (y_i·u) sin(theta) = 0\`.

The event roots can be sorted, yielding finitely many constant-cut intervals. Evaluate incremental signed-cut changes exactly using Lemma 1's flip gains, grouping simultaneous crossings. **Exactness applies to this one chosen arc only**, not the entire orientation sphere or arbitrary partitions.

At fixed nonzero offset \`b\`, replace the right-hand side of this equation by \`b\` and solve each trigonometric event equation over the selected angular interval; there may be two roots per vertex on a full cycle. Prefer the existing robust origin-arc routine through Lemma 3 rather than implementing a new unvalidated root solver.

**Repository already has:** \`src/maxkcut/eva_direct_rounding.py::optimize_semicircle\` (original-signed-score event sweep) and \`optimize_hyperplane\` (repeated exact arc search). Experiment027 additionally ranks tangents by angular margins, one-flip incentives and target pairs, but **then applies Flip/KL/ILS** in its official final ranking. For Experiment028, archive the **raw** rounder output as the main outcome; do not invoke those refinement routines.

## Lemma 3 — An affine hyperplane is an origin hyperplane in one more dimension

Let \`tilde Y = [Y | 1_n]\` and \`tilde r=(r,−b)\`. Then

\`sign(Yr − b*1_n) = sign(tilde Y * tilde r)\`.

Therefore **a rotating *and translating* hyperplane in the original \`d\`-dimensional embedding is equivalent to a through-origin rotating normal in augmented \`d+1\` dimensions**. Since multiplying \`tilde r\` by a positive scalar does not change signs, \`tilde r\` can be normalized to the sphere for existing great-circle sweep code. A tangent step in this augmented space may change both orientation and translation. This is the most direct algorithmic bridge to our existing exact arc-sweep kernel.

**Important scientific distinction:** The augmentation is a *rounding search transformation*, **not** a new feasible GW SDP solution, not an unchanged GW distribution, and not a claim that the same 0.87856 guarantee persists. Normalize/scale offsets consistently, especially if original factor rows are not perfectly unit length.

## Proposition — Full-rank embeddings remove hyperplane expressivity restrictions

If \`rank(Y)=n\` (which requires \`d>=n\`), for *any* desired spin vector \`s in {−1,+1}^n\` there exists \`r\` with \`Y r=s\`, hence \`sign(Yr)=s\`. Thus the unrestricted *homogeneous hyperplane family* already contains **every possible Max-Cut partition**. Similarly, affine hyperplanes can express all sign assignments whenever \`rank([Y|1_n])=n\`.

**Consequences:** A claim that a “best linear separator” is an intrinsically simpler problem is false at full row rank: optimizing it is then as hard as unrestricted Max-Cut. An apparent graph-structure story can also be vacuous if every partition is linearly representable. Determine factor rank / effective rank and compare **rank-truncated fixed factors** (orthogonal low-rank projection, not a new SDP solve), holding cut scoring on the original graph. Analyze reachability, angular stability and complexity versus retained dimension \`d_eff\`. Report which limits preserve the original gram relationships and which are altered by truncation.

## Primary experimental arms (unrefined roundings only)

All methods use existing canonical G-set graphs, exact frozen archived EVA factors, and **the identical 16 saved GW starting normals** per graph, with matched index and verifiable SHA. Never recompute EVA.

| Arm | Allowed search | What it tests |
|---|---|---|
| A — GW origin | Single random \`r_0\`, \`b=0\` | Unrefined baseline |
| B — exact threshold | Same \`r_0\`, \`b=argmax_b F(r_0,b)\` | Pure *translation* gain, exactly optimal conditionally |
| C — origin rotation | Repeated exact semicircles in \`R^d\`, \`b=0\` | Pure *rotation* gain; reuse historical raw HP/E027 where seed identities match |
| D — alternating | Rotate \`r\`, then re-optimize \`b\` exactly; accept only better true cuts | Conditional blockwise optimization; not global |
| E — augmented rotation | Exact semicircle search using \`tilde Y=[Y|1]\`, \`tilde r=(r,−b)\` | Joint rotation+translation in lifted space |
| F — matched random search | Matched count of random orientations, each with exact threshold sweep | Tests whether greedy guidance, not simply more candidates, matters |

A no-search \`b=0\` always serves as feasible starting point, so an **exact threshold on that same \`r\` cannot reduce the cut**. Affine optimization has more expressivity than origin-constrained rounding by construction, but empirical **equal-budget**, **held-out-graph** benefit is not automatic. Compare CPU and **number of graph-cut event evaluations / exact arc sweeps**; pure translation is much cheaper than a fresh 250-arc search. Keep both *best-of-16* and *mean-of-16* raw values; the latter characterizes typical paired rounding behavior.

**Do not call C/D/E “analytic global best”**: they only optimize along selected paths / blocks. One could later examine exact global arrangement enumeration at *fixed small rank d* on small graphs, but this is a separate algorithmic experiment with potentially combinatorial \`n^d\` cost.

## The interesting mathematics is the *landscape*, not only the winning cut

The function \`F(r,b)\` is **piecewise constant** over cells of a hyperplane arrangement in the parameter domain. Its ordinary gradient is zero almost everywhere; classical differentiable gradient descent on the hard cut objective is unsuitable without a surrogate. At boundaries \`y_i·r=b\`, crossing one vertex changes the score by the explicit signed neighborhood gain \`Delta_i\`; simultaneous events require grouping. The event ordering depends on the geometry of the embedding, while each jump depends on the graph's signed adjacency. **This is a measurable bridge between geometric topology and network topology.**

Pre-register landscape outputs such as:

1. **Boundary event score:** \`Delta_i\`, \`degree_i\`, weighted degree, sign imbalance, local triangles, angular margin \`abs(y_i·r-b)\`, and distribution of single/pair critical events.
2. **Basin/plateau stability:** angular/offset volume of constant-cut cells, jump-size distribution, number of accepted strict improvements, number of promising but *geometrically inaccessible* single-vertex flips.
3. **Partition-size spectrum:** size of either side as \`b\` varies; weighted cut quality conditional on balance. Max-Cut does not impose balance by default; only an optional explicitly named secondary analysis should add it.
4. **Graph-level covariates:** Experiment022 family, negative-edge fraction, degree, transitivity, triangles, and components; control for \`n,m\`, factor rank/eigenvalue concentration, and generator confounding.
5. **Reproducibility / robustness:** repeated matched GW seeds, perturbations of \`r\` and \`b\`, same graph's Gram-eigenbasis changes (orthogonal equivariance), numerical tie tolerances, and low-rank truncation.
6. **Scientific null controls:** distinguish fixed-original-embedding experiments from graph rewiring/sign shuffling controls (the latter perturb the relationship between embedding and graph; interpret carefully).

No claimed association is causal just because “the best” plane was selected on the same graphs. Use a *held-out graph-group* protocol for any predictor of a high-quality hyperplane or of which arm to choose. For purely descriptive per-graph results, give bootstrap uncertainty clustered by graph, not by correlated GW normal.

## Reproducible implementation sequence; future work, NOT executed

**Proposed module:** \`scripts/experiment028_hyperplane_geometry.py\`; output namespace \`results/gset/experiment028-hyperplane-geometry/\`; **separate PR/branch after verifying no collision with other agents**. Keep \`scripts/experiment027_angular.py\` and \`src/maxkcut/*.py\` frozen unless a separately reviewed compatibility change is necessary.

1. Verify saved original graph SHA, factor SHA and the 16 identical GW normals before any trial; test on a toy \`n=4\` signed graph first.
2. Reuse and independently test \`optimal_threshold\` for pure translation, including exact projection ties and endpoints; check all returned cuts with the original \`cut_weight\`.
3. Reuse and independently test \`optimize_semicircle\` in origin coordinates and in the lifted affine space; separately assert that the final labels **are realizable by the final stored hyperplane**.
4. Compare arms A–F with common starts, graph-scoped paired seeds and strict budget accounting. Source-hash executable/settings as in E027.
5. For **every** trial, save the starting \`r_0\`, current/accepted \`(r,b)\`, raw cut, best cut, candidate events, grouped tie crossings, thresholds, graph and factor IDs, CPU time, attempted directions, and intermediate strictly improving partitions. **No FLIP, KL, ILS, or other label-space postprocessor.**
6. Preserve per-iteration JSONL/NPZ histories in the VM source folder and publish content-addressed Release archives and small Git analysis tables (no Git LFS required).
7. Compute controlled single-arm quality, matching equal computational work, family/rank stratified hypotheses, graph-cluster CIs and honest failure modes; report both negative and positive findings. Update AGENTS/handoff and do not merge without authorization.

## Hypotheses and falsifiable tests

- **H1 (translation helps):** \`C_B>=C_A\` *necessarily for the same start*, but the *magnitude* of its gain and its conditional balance profile vary by graph/rank.
- **H2 (joint improves search at a cost):** \`C_E\` may exceed \`C_C\` at equal CPU budget, but this is NOT guaranteed for finite-budget heuristics. Pre-register comparison.
- **H3 (geometry exposes structure):** low margins alone do **not** predict improvements; signed neighborhood gains jointly with geometry predict which critical crossing arcs yield positive delta. Test held-out graphs and sign/degree controls.
- **H4 (rank effect):** as effective rank increases, reachable cuts and event complexity increase; unconstrained full rank may erase meaningful hyperplane representational restrictions.
- **H5 (refinement masking):** unrefined rankings/orderings differ materially from historical post-Flip/KL/ILS rankings; use archived original E025/E027 *raw* outcomes as available, without rerunning numerical work.

## Related work (novelty boundary)

- M. X. Goemans and D. P. Williamson, “Improved Approximation Algorithms for Maximum Cut and Satisfiability Problems Using Semidefinite Programming,” *JACM*, **1995**. Origin-constrained random hyperplane; classical guarantee for standard nonnegative Max-Cut weights: https://research.ibm.com/publications/improved-approximation-algorithms-for-maximum-cut-and-satisflability-problems-using-semidefinite-programming.
- **K. Lang**, “Finding Good Nearly Balanced Cuts in Power Law Graphs,” **2004**, §4.1: *swept random hyperplanes* deliberately translate GW normal directions and exactly scan all induced threshold cuts. **Translation sweep is existing prior art**, not novel here: https://studylib.net/doc/14933844/finding-good-nearly-balanced-cuts-in-power-law-graphs-kev...
- **G. Maliakal et al.**, “A Dataless Reinforcement Learning Approach to Rounding Hyperplane Optimization for Max-Cut,” arXiv:2505.13405, **2025**. Directly searches stronger GW rounding normals with an online RL policy; our potential contribution must differ in analytic geometry/finite arc exactness, interpretability, graph-family structure, or principled rank control: https://arxiv.org/abs/2505.13405.
- **Existing repository methods:** \`gw-threshold\` and \`optimal_threshold\` (fixed normal, affine offset), \`hyperplane-sweep\` and \`optimize_semicircle\` (great-circle directional search), and E027's three guided tangents. Any follow-up must acknowledge these mechanisms and their previous numerical outcomes.

## Deliverables and next handoff

At this stage this note **documents the hypothesis and design**, not an empirical discovery. Before coding, check \`AGENTS.md\`, \`RESEARCH_HANDOFF.md\`, E027 source SHA and exact archived factor identity, all existing Experiment028 reservations and current branches. The next agent should first implement a tiny exact/degenerate 4-vertex proof-of-correctness fixture, then a raw-only 3-graph pilot, then full 71 only if a scientifically justified pilot and reproducibility gate pass. No new hyperplane experiment has been run in this note.
