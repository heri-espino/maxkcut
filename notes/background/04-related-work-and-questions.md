# Related work, graph coloring, and unresolved research questions

**Sources:** [TH21, §§7.5, 8; PDF pp. 102, 105–107], [RF20], [FKP22]. See [source index](README.md).

## 1. Chronology and what is *already* known

| Work | Combinatorial problem | Relaxation / rounding idea | Distinguishing claim |
| --- | --- | --- | --- |
| **Goemans–Williamson (1995)**, as recounted in TH21 and RF20 | Max-Cut, $k=2$ | SDP vectors; one random hyperplane | Expected approximation bound about 0.87856 |
| **Frieze–Jerrum (1997)**, as recounted in TH21 §7.3.3 | Max-$k$-Cut, $k\ge2$ | $k$-way SDP; $k$ Gaussian directions | Expected $\alpha_k$ approximation factors |
| **de Klerk, Pasechnik & Warners (2004)**, as recounted in TH21 §7.3.2 | Coloring-related / Max-$k$-Cut links | Lovász-$\vartheta$-related SDP and different rounding | Thesis discusses equivalence and guarantees |
| **Rodríguez Fernández et al. (2020)** [RF20] | Max-Cut, $k=2$ | Keep SDP; replace random-hyperplane-only rounding with spherical 2-means or other clustering | G-set improvements; e.g., mean +1.541% for K-MeansNM over baseline |
| **Rodríguez Fernández (2021)** [TH21, ch. 7] | **Max-$k$-Cut for $k=3,4,5$** | Keep the Frieze–Jerrum SDP; replace Gaussian direction rounding with **spherical $k$-means** on SDP vectors | Better **reported benchmark** cuts, no proved universal approximation factor for clustering |
| **Felzenszwalb, Klivans & Paul (2022)** [FKP22] | Max-$k$-Cut SDP used as a **clustering formulation** | **Iterated linear optimization / fixed-point** SDP rounding | Geometric partition-matrix theorem, convergence framework, improved empirical clusterings versus random rounding |

**Research integrity:** An empirical extension of “use $k$-means instead of Gaussian rounding for Max-$k$-Cut” is **not a new proposal**: it appears explicitly as Algorithm 12 in [TH21, §7.3.4]. A claim that general-$k$ SDP rounding by structured postprocessing has never been studied would ignore [FKP22]. A new project must define a narrower contribution (proof, scalability, benchmark regime, algorithmic modification, or new rigorous experiment).

## 2. The 2020 article versus the 2021 dissertation

[RF20, *Computation* 8(3):75, DOI:10.3390/computation8030075]:

- Works with **$k=2$**, Goemans–Williamson SDP and the G-set graph corpus.
- Compares fuzzy c-means, 2-means, 2-medoids, MST-based rounding, and GW, including multiple initializations.
- Observes negative correlation of *cluster distortion* with *cut weight* over parts of the 2-means trajectory (Figure 1), but not an exact objective identity. Some graphs reach their best cut **before** reaching minimum distortion.
- Reports mean relative cut gains of 1.541% for K-MeansNM and 1.533% for K-Means2N; local search further improves results.
- Shows that **keeping the best cut along a trajectory initialized by GW rounding** preserves the GW expected guarantee for that hybrid, but does **not** improve the universal ratio.

[TH21, Chapter 7] **extends the idea to $k>2$** by directly clustering the vectors from the Frieze–Jerrum relaxation; the benchmark changes to 300 small A/B/C/D random graphs and $k=3,4,5$. A new paper should not conflate the 2020 large G-set results with the 2021 Max-$k$-Cut experiment.

## 3. The 2022 JMLR paper is related but *not* the same algorithm

[FKP22, “Clustering with Semidefinite Programming and Fixed Point Iteration,” *JMLR* 23 (2022), paper 21-0402](https://jmlr.org/papers/v23/21-0402.html).

Let

$
\mathcal L_{n,k}=
\{X=X^\top\succeq0:
X_{ii}=1,\ X_{ij}\ge-1/(k-1)\}.

$
FKP22 calls this the **$k$-way elliptope**. Its partition matrices have pairwise entries

$
X_{ij}\in\left\{1,-\frac1{k-1}\right\};

$
**Theorem 7 (FKP22, §4)** identifies the vertices of the $k$-way elliptope as $k$-partition matrices (partitions into **at most** $k$ nonempty sets).

Instead of directly passing the SDP factor through $k$-means, FKP22 constructs a sequence of SDPs. With

$
a=\frac{1-k/2}{k-1},\qquad A=a\,\mathbf1\mathbf1^\top,

$
the update in **FKP22, §5, Eq. (fixed-point iteration)** is

$
\boxed{
X_{t+1}\in\operatorname*{arg\,max}_{Y\in\mathcal L_{n,k}}
\langle X_t+A,Y\rangle_F.
}

$
This repeatedly solves a *new linear objective* over the same convex feasible body. [FKP22] proves partition matrices are **attractive fixed points** and describes convergence; it reports experiments where iterates reach partition matrices. **Theoretical convergence to a fixed point should not be overstated as a theorem that all starting states must reach a discrete partition**: the paper notes possible other fixed points.

### Different use of Max-$k$-Cut as a clustering objective

FKP22 often starts with *data points* $x_i\in\mathbb R^d$ and sets **graph weights** $M_{ij}=\|x_i-x_j\|^2$. Maximizing separated-pair distance is the same as minimizing within-cluster pairwise distance. The identity

$
\sum_{i,j\in C}\|x_i-x_j\|^2
=2|C|\sum_{i\in C}\|x_i-\bar x_C\|^2

$
shows a **cluster-size-weighted** $k$-means-type objective, not exactly ordinary $k$-means. This is **clustering data by solving Max-$k$-Cut**, whereas TH21 is **solving an externally given Max-$k$-Cut by clustering SDP vectors**.

The FKP22 experiments include synthetic point clouds and an MNIST subset. In their Gaussian-mixture experiment (10 datasets), fixed-point versus best-of-50 random rounding had a mean cut-weight ratio of **1.014**; reported mean Rand indices were **0.972 ± 0.006** (fixed-point) and **0.935 ± 0.018** (random). These numbers belong to **FKP22’s data-clustering setup**, not the 2021 thesis's graph benchmark. See FKP22 §6, Table 1 (PDF p. 21).

**Practical consequence:** include fixed-point rounding as an optional *strong prior-art baseline* if resources allow, but account for its multiple SDP solves. It is not automatically equivalent to spherical $k$-means and should not be misattributed to the dissertation.

## 4. Connection with $k$-coloring and Lovász $\vartheta$

**Source:** [TH21, §7.5, PDF p. 102 / printed p. 92].

For an unweighted graph and any $k\ge2$:

- The existence of a proper $k$-coloring is equivalent to the existence of a $k$-partition that cuts **every edge** (cut weight $|E|$), permitting empty classes in the “at most $k$” convention.
- The smallest such $k$ is the chromatic number $\chi(G)$.
- With $\omega(G)$ the clique number and $\overline G$ the graph complement, the thesis quotes Lovász's sandwich inequality:
  $
  \omega(G)\le\vartheta(\overline G)\le\chi(G).
  
$$
- Vector $k$-coloring asks **on graph edges** $v_i^\top v_j\le -1/(k-1)$, with equality for *strict* vector coloring (TH21 Eq. 7.11).

**Do not mix up signs.** The **Max-$k$-Cut relaxation** imposes $v_i^\top v_j\ge-1/(k-1)$ on all distinct pairs, while the **vector-coloring** condition just quoted has a $\le$ on edges. The goals/feasible sets differ even though both involve simplex vectors and SDP machinery.

For signed weights, a proper coloring need not maximize the signed cut; the “all edges cut = optimum” statement relies on unweighted/nonnegative edges.

## 5. Thesis Chapter 8: source-stated future work

**Source:** [TH21, Chapter 8, printed pp. 96–97 / PDF pp. 106–107].

- Establish a theoretical approximation guarantee for clustering rounding; the chapter's **data suggest** one, but supply no proof.
- Extend the random-graph benchmark to **planted $k$-way** graphs: current type D favors **two** groups.
- Test larger values of $k$, including cases with negative edge weights where increasing $k$ worsened AIG.
- Develop more scalable SDP solvers; the thesis identifies around $n=200$ as a practical bottleneck.
- Investigate whether Gram and graph-Laplacian **spectra** predict problem difficulty and help select rounding methods; chapter 6 explores this for Max-Cut.
- Explore improved similarity metrics for spectral clustering and more complete search through the cut-versus-inertia landscape (chapter 6).

These are **2021 proposals**, not claims that they remain open in the entire 2026 literature. [FKP22] is evidence that related structured rounding has advanced since the dissertation.

## 6. Proposed falsifiable project questions (not source results)

**Q1 — Geometric predictors.** Does the SDP spectrum (effective rank, eigenvalue decay, leading eigengap, clustering margin) predict paired $\Delta F=F_{\mathrm{KM}}-F_{\mathrm{FJ}}$ after controlling for $n$, $k$, graph family, and time budget?

**Q2 — Matched computational work.** At equal *wall-clock rounding* budgets, does many-start spherical $k$-means outperform many-draw FJ? What about both with identical single-vertex local search?

**Q3 — Planted $k$-way structures.** How does each rounding strategy behave when the generator actually favors $k$ groups, rather than the D-family's planted 2-way structure?

**Q4 — Signed graphs.** On type B, does using a shift-aware objective/diagnostic change which rounded solutions appear best? Optimize **unshifted cut weight** in either case; use shifts only for interpretable ratios.

**Q5 — Geometry/objective conflict.** Can we construct small explicit graphs where one spherical $k$-means update **reduces spherical distortion but lowers cut weight**? A counterexample would disprove any naive monotonic-improvement claim.

**Q6 — Guarantees.** Investigate whether a *specific* randomized initialization+refinement has a bound without fallback. In parallel, use the [FJ + clustering hybrid](02-rounding-methods.md#6-a-safe-hybrid-baseline-derived-proposal-not-a-thesis-theorem) to retain an established baseline bound **without** claiming a stronger $\alpha_k$.

**Q7 — Strong baselines.** How much of the advantage persists against [FKP22] fixed-point rounding under matched solver budgets and scalability constraints?

No priority ordering here implies novelty. Before presenting a publication claim, conduct a newer literature review; these are candidate questions for tests and proofs.

**Next:** [Reproduction protocol](05-reproduction-protocol.md).
