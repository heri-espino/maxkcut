# Proposed reproduction protocol for Chapter 7 (not yet executed)

**Source baseline:** [TH21, §§7.3–7.4 and Chapter 8] and [RF20]. **Everything below is a proposed experimental design**, unless a statement is explicitly attributed to those sources. See [source index](README.md).

## 1. Questions and primary endpoint

**Primary question:** when both methods use the **same solved Frieze–Jerrum SDP**, does spherical $k$-means rounding produce larger objective values than FJ randomized rounding?

For a graph instance $g$, let $F_{g,\mathrm{KM}}$ and $F_{g,\mathrm{FJ}}$ be the best cut weights found within **identical, specified rounding budgets**. Define

$
\Delta_g=F_{g,\mathrm{KM}}-F_{g,\mathrm{FJ}}.

$
The primary endpoint is a **paired comparison** of $\Delta_g$ across independent graph instances. For a relative quantity on nonnegative instances, report $\Delta_g/F_{g,\mathrm{FJ}}$ only if $F_{g,\mathrm{FJ}}>0$. For negative weights or zero baselines, report differences plus separately defined shifted ratios.

Secondary endpoints: AIG to the reported SDP objective (provided it is a validated upper bound), CG with the proper signed shift, wall-clock time, rounding CPU time, objective evaluations, actual number of **nonempty classes**, failure rate, and per-instance variability across seeds.

## 2. Dataset construction and provenance

First reproduce the **four A/B/C/D generators**, not the G-set graphs already in the repository:

```text
for family in {A,B,C,D}:
  for n, number_of_graphs in {(50,50),(100,20),(200,5)}:
    independently generate number_of_graphs undirected graphs
    for k in {3,4,5}:
      solve the Frieze–Jerrum SDP once per graph/k configuration
      apply both algorithms to exactly that SDP
```

Generator definitions are in [experiment notes](03-thesis-experiments.md#2-the-four-graph-generators-table-1-pdf-p-97). Preserve edge symmetry, diagonal zeros, signed integer weights for B, and the exact type-D two-half probabilities. Save **independent graph-generation seeds** separately from **rounding seeds**. The initial study is $300$ graphs and $900$ graph-$k$ configurations, although additional repeated rounding runs are required for robust comparisons.

A **separate** future experiment may use the existing `G-Set/*.mat` files (primarily a Max-Cut benchmark) or planted $k$-way stochastic block models. Keep those experimental families separately labeled; they are not thesis Figure 26 replications.

## 3. Mathematical SDP specification and audit

Construct symmetric $W$ and Laplacian $L=D-W$ with $D_{ii}=\sum_j w_{ij}$. Solve, independently for each graph and $k$,

$
\begin{aligned}
\max_X\quad &\frac{k-1}{2k}\operatorname{Tr}(LX)\\
\text{subject to}\quad &X=X^\top\succeq0,\ X_{ii}=1,\ 
X_{ij}\ge -1/(k-1)\quad(i\neq j).
\end{aligned}

$
Log solver, version, termination status, time, primal/dual objective, primal/dual residuals, tolerance, and minimum eigenvalue. In exact mathematics, $S_{\mathrm{SDP}}\ge F^\star$. **Numerical caution:** an approximate *primal feasible point* alone does not certify an upper bound on a maximization problem; obtain a suitable **dual bound** or document tolerance limitations before describing an empirical ratio as a certified integrality gap.

Extract a vector factor from $X$:

1. Symmetrize $X$ numerically.
2. Inspect its eigenvalues $\lambda_i$.
3. For **tiny negative eigenvalues caused by numerical error**, document an explicit PSD correction/eigenvalue clipping and record its Frobenius distance to the solver output.
4. Factor the PSD surrogate $\bar X=V^\top V$; extract one vector **per vertex**.
5. Check $\|v_i\|$, resulting inner products, and whether repair/normalization altered the SDP objective and pairwise bounds.

Do **not** silently claim that “clip negative eigenvalues” is the exact Higham nearest-correlation-matrix projection or that it preserves all k-way elliptope constraints. Report if the repaired matrix violates $X_{ii}=1$ or $X_{ij}\ge-1/(k-1)$.

## 4. Rounding baselines: specify the random source

### Frieze–Jerrum (TH21 Algorithm 11)

```text
given V (rows or columns: record convention), k, seed:
    sample R: k independent Gaussian N(0, I_d) vectors
    for vertex i:
        label[i] = argmax_a dot(v[i], R[a])
    return labels, exact_weighted_cut(labels)
```

**Do not normalize each Gaussian direction as an unannounced change.** The source's Algorithm 11 uses independent Gaussian coordinates.

### Spherical $k$-means (TH21 Algorithm 12 with §7.4 initialization)

```text
given the *same* V, k, seed:
    sample k independent Gaussian vectors
    normalize each to unit norm -> centroid directions
    repeat:
        assign each vertex vector to centroid with largest dot product
        recompute each centroid by normalizing the sum of assigned vectors
        apply a documented empty-cluster and zero-centroid policy
        log geometric distortion AND graph cut weight at each iteration
    until assignments converge, or a stated stopping criterion applies
    return final labels, exact_weighted_cut(labels)
```

The update rule is a concrete implementation of **spherical** $k$-means, not pseudocode directly reproduced from the thesis. Record maximum iterations, tolerance, treatment of ties/empty clusters, and **whether the best intermediate cut or only the final cut is retained**. For historical-faithful reproduction, retain final cut as one endpoint; analyze trajectory-best as a distinct proposed variant.

**Independent restarts** matter: select a stated set of seeds (e.g. 50 per method as a prospective experiment design) and predeclare either best-of-$R$ or a fixed-time policy. The exact number “50” here is **a proposal**, not attributed to the Chapter 7 experiment. Do not compare 50 k-means restarts against one Gaussian FJ draw and present it as a controlled rounding comparison.

### Optional hybrids and refinement

- **Hybrid:** evaluate FJ and $k$-means outputs on the original $W$, return the higher cut. This trivially dominates the realized FJ cut and therefore retains its *expected bound* under the corresponding FJ theorem assumptions, but proves no stronger universal factor.
- **One-vertex local search:** for vertex $i$ currently in class $a$, moving to class $b\ne a$ changes the cut by
  $
  \Delta_{i:a\to b}
  =\sum_{j:c(j)=a}w_{ij}
  -\sum_{j:c(j)=b}w_{ij}.
  
$$
  Move only if $\Delta>0$; stop when no improving move remains. Log performance **before and after** local search to avoid attributing all gains to clustering.

The source [RF20] explores local search for $k=2$. Generalizing one-vertex moves to $k>2$ is the proposed protocol here.

## 5. Reporting template

For every graph-$k$ pair save a machine-readable record with:

```text
family, n, graph_id, graph_seed, k,
m, weight_min, weight_max, negative_weight_fraction,
solver, solver_version, status, runtime_sdp_s,
sdp_primal, sdp_dual_bound, min_eigenvalue,
diag_max_error, kway_bound_max_violation,
repair_method, repair_frobenius_distance,
rounding_method, rounding_seed, restart_id,
rounding_time_s, iteration, cluster_distortion,
num_nonempty_clusters, cut_weight,
AIG_if_valid, CG_if_defined, local_search_applied
```

For plots reproduce the **aggregate form** of Figure 26 with explicit means/error bars; do not invent source-exact points. Also show **paired $\Delta_g$** distributions and comparisons by $k$, graph size, family, and time. Prespecify treatment of missing/failed SDP solves; never drop hard cases only for one rounding method.

Useful inference: bootstrap **over independent graph instances**, not over correlated repeated rounding runs on the same graph. Report 95% intervals for average and median paired differences, sign proportions, and multiple-testing context across graph families and $k$. Include failed/timeout counts for each group.

For small $n$ where feasible, add a separate exact MILP or combinatorial optimizer to estimate $F^\star$; then report actual $F/F^\star$, which is distinct from AIG $F/S_{\mathrm{SDP}}$.

## 6. Preregistered diagnostic hypotheses

1. **H1 (paired quality):** spherical $k$-means has higher average cut weight than FJ at a matched budget for at least one of A/C/D, but might not dominate for all $k$ and signed B.
2. **H2 (geometry):** benefit correlates with effective rank/eigenspectrum of the optimized Gram matrix, beyond density and size.
3. **H3 (trajectory):** lower spherical inertia may be associated with higher cut on average, **not necessarily monotonically at every step**. Log both; search for counterexamples.
4. **H4 (benchmark structure):** the source's type D favors two classes; a planted *$k$-class* graph generator will reveal qualitatively different rounding behavior.
5. **H5 (cost):** improvements in best cut may shrink or disappear when one allows FJ as many draws as k-means uses in time.

All hypotheses require tests; none follows as a theorem from the supplied materials.

## 7. Acceptance criteria for an actual “reproduced thesis result”

- The PDF chapter's SDP equation, graph generators, $k$ range, and rounding distributions are implemented *as written* or every deviation is disclosed.
- Each graph-$k$ instance uses **the same SDP factor** for the two rounding alternatives.
- Every rounded partition is checked for membership and its cut score recomputed directly from the graph.
- For the signed B graphs, ratios are computed with the source's shifts $W_-$ and $W_+$.
- Figure 26-like summaries are regenerated from saved per-instance data, with uncertainty and provenance.
- Numerical PSD repair is explicitly recorded and feasibility diagnostics are preserved.
- The report distinguishes **reproduced observations**, **new experimental findings**, and **unproved approximation claims**.
- Heavy generation of figures/SDP results should be **explicitly launched** rather than accidentally run as automatic repository CI.

**Status as of writing:** this file is a documentation and experiment-design artifact; **no reproduction has been executed**.
