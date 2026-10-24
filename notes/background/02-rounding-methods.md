# Rounding an SDP solution: random directions versus $k$-clustering

**Source:** [TH21, §7.3, PDF pp. 92–96 / printed pp. 82–86], with [RF20] for the $k=2$ predecessor and [FKP22] for a distinct later approach. See [source index](README.md).

## 1. Why rounding exists

The SDP provides a unit-vector embedding $v_1,\dots,v_n$ whose Gram matrix approximately solves the *relaxed* maximum-cut objective. It does **not** directly specify one of the $k$ simplex labels. A rounding rule must turn vectors into graph classes $A_1,\ldots,A_k$.

There are two independent objectives:
- **Geometric coherence:** group nearby SDP vectors.
- **Cut quality:** maximize $F(c)=\sum_{i<j}w_{ij}\mathbf1[c(i)\ne c(j)]$.

[TH21] hypothesizes that, because positive-weight edges pull vectors apart in the SDP, geometric clustering is an effective surrogate for a good graph cut. It **does not prove** that reducing $k$-means inertia monotonically increases cut weight.

## 2. Algorithm 11: Frieze–Jerrum randomized rounding

[TH21, §7.3.3 and Algorithm 11, PDF pp. 95–96]. Given $v_i\in\mathbb R^d$:

1. Draw $k$ **independent Gaussian vectors** $r_1,\ldots,r_k$, with each coordinate $\sim N(0,1)$.
2. For each graph vertex $i$, choose
   $$

   c_{\mathrm{FJ}}(i)=\operatorname*{arg\,max}_{1\le a\le k}\langle v_i,r_a\rangle,
   
$$
   using a reproducible tie-breaking rule.
3. Return the partition corresponding to $c_{\mathrm{FJ}}$.

The Gaussian vectors need **not** be normalized: Algorithm 11 explicitly samples raw Gaussian coordinates. **Normalizing the $r_a$ individually can change the maximizer**; do not silently change the baseline to nearest *unit* random centroid.

The guarantee quoted in [TH21, §7.3.3] is of the form

$$

\mathbb E[F(c_{\mathrm{FJ}})]\ge\alpha_k F^\star

$$

under the usual conditions of the cited approximation setting (in particular distinguish nonnegative from signed weights). Values reproduced as reported in the thesis:

| $k$ | Lower bound on $\alpha_k$ quoted in TH21 |
| ---: | ---: |
| 2 | 0.878567 |
| 3 | 0.800217 |
| 4 | 0.850304 |
| 5 | 0.874243 |
| 10 | 0.926642 |
| 100 | 0.990625 |

The thesis also states $\alpha_k>1-1/k$, with $\alpha_k-(1-1/k)\sim 2\log(k)/k^2$ for large $k$. These are **bounds on expectation**, not promises for every draw, and not derived anew here.

## 3. Algorithm 12: clustering-based rounding

[TH21, §7.3.4 and Algorithm 12, PDF p. 96].

1. Solve the **same** SDP/vector relaxation as in Algorithm 11; obtain $v_1,\ldots,v_n$.
2. Cluster these **SDP vectors**, not original graph vertices or adjacency rows, into $C_1,\ldots,C_k$.
3. Convert membership into labels $c_{\mathrm{KM}}(i)=a$ if $v_i\in C_a$, hence graph blocks $A_a=\{i:c_{\mathrm{KM}}(i)=a\}$.

This is the *actual* Max-$k$-Cut generalization proposed in Chapter 7 — not merely a speculative extension of the earlier Max-Cut work.

### Which clustering algorithm was actually tested?

[TH21, §7.4, PDF p. 97] says the experiment used **spherical $k$-means**, initializing its $k$ centroids as independent random unit vectors: sample each coordinate from $N(0,1)$, then normalize each centroid to unit length.

For unit data $v_i$ and unit centroid directions $\mu_a$, an implementation consistent with the spherical $k$-means description is:

$$

c(i)\leftarrow\operatorname*{arg\,max}_a v_i^\top \mu_a,
\quad
s_a\leftarrow\sum_{i:c(i)=a}v_i,
\quad
\mu_a\leftarrow s_a/\|s_a\| \quad(\|s_a\|>0).

$$

Repeat assignment and update until stable (or a tolerance / iteration limit); handle empty classes and $s_a=0$ explicitly. This is **implementation-level pseudocode**, not a verbatim listing from Algorithm 12, which specifies the clustering step abstractly. The thesis does not document every stopping criterion, initialization seed, or empty-cluster policy.

Spherical $k$-means seeks to maximize $\sum_i v_i^\top\mu_{c(i)}$, or equivalently minimize $\sum_i\|v_i-\mu_{c(i)}\|^2$ with unit centroids. Neither criterion directly contains $w_{ij}$, except insofar as the SDP embedding encodes those weights.

### Why “closer” can be useful but is not a proof

If two SDP vectors have large positive inner product, geometric rounding tends to assign them together. If they are widely separated, it tends to place them in different classes. This behavior is compatible with the SDP objective for many positive-weight graphs. However:

- A local decrease in geometric distortion **need not** increase $F$ for an arbitrary graph.
- Signed weights and frustrated edge preferences can make the geometric surrogate inaccurate.
- Random initialization, local optima, and empty clusters create variability.
- No universal $\alpha_k$ for the **standalone** Algorithm 12 is proved in [TH21].

The thesis reports an empirical improvement, not a stronger worst-case theorem.

## 4. Other rounding formulations discussed in §7.3

**§7.3.1 — Goemans–Williamson (2004), three colors.** The thesis discusses a complex-Hermitian SDP via Max-2-Lin-Mod-3, then phase-based randomized rounding into three angular sectors (Eq. 7.7 and its rounding rule). This is distinct from the original 1995 binary hyperplane algorithm.

**§7.3.2 — de Klerk, Pasechnik & Warners (2004).** The thesis introduces a formulation related to graph coloring and the Lovász $\vartheta$-function, including a higher-dimensional matrix/Kronecker-product rounding procedure (Eqs. 7.8–7.9, Algorithm 10). This is not directly interchangeable with the weighted maximization SDP (7.5); feasibility restrictions and dimension differ. The text notes equivalence connections to the Frieze–Jerrum analysis.

**§7.3.3 — Frieze–Jerrum (1997).** A maximization SDP allowing any integer $k\ge2$, with independent Gaussian directions (Algorithm 11).

**§7.3.4 — Rodríguez Fernández's proposal.** The **same** maximization SDP as Frieze–Jerrum, but round the *solution vectors* via clustering (Algorithm 12).

For controlled experiments, **Algorithms 11 and 12 must share the same solved SDP instance**; otherwise differences in solver quality confound rounding comparisons.

## 5. Connection to the 2020 Max-Cut paper

[RF20, §§2.2–3.4] studied $k=2$ and compared:
- Goemans–Williamson random-hyperplane rounding;
- spherical 2-means with several seeds, notably a random vector and its negative (**K-MeansNM**), and two independent random seeds (**K-Means2N**);
- k-medoids, fuzzy c-means, and MST-based methods;
- optional 1-spin-flip local improvement.

Why K-MeansNM is special at $k=2$: antipodal centroid initialization gives **the same initial bipartition** as a random hyperplane. Recording the **best cut over the entire clustering trajectory, including iteration zero**, therefore preserves the GW baseline solution and its guarantee. The paper reports mean relative improvements of **1.541%** for K-MeansNM and **1.533%** for K-Means2N on its benchmark (and larger improvements on some individual graphs). [RF20, §3.3, printed p. 9.]

**Do not automatically extrapolate this proof to $k>2$.** In [TH21, Algorithm 11] FJ uses **raw** Gaussian directions, whereas §7.4 initializes k-means with **normalized** random centroid directions; the initial assignments can differ. The chapter does **not** prove an analogous bound for Algorithm 12.

## 6. A safe hybrid baseline (derived proposal, not a thesis theorem)

A **new comparison method** can always evaluate both candidate partitions on the exact original weights:

$$

c_{\mathrm{hybrid}}=\operatorname*{arg\,max}_{c\in\{c_{\mathrm{FJ}},c_{\mathrm{KM}}\}}F(c).

$$

Then **pointwise** $F(c_{\mathrm{hybrid}})\ge F(c_{\mathrm{FJ}})$. Under the nonnegative-weight assumptions where the cited FJ bound applies, taking expectations yields

$$

\mathbb E F(c_{\mathrm{hybrid}})
\ge\mathbb E F(c_{\mathrm{FJ}})
\ge \alpha_k F^\star.

$$

This preserves the *known* bound but proves **no strictly larger** worst-case factor. It remains valid as a pointwise cut comparison with negative weights, although signed-weight approximation-ratio conventions require a shift and separate justification.

A further safe postprocessor is monotonic local search: move a vertex to a different class **only when its exact weighted cut value strictly increases**. This cannot worsen its starting cut, but has its own time/optimality limits.

## 7. Algorithmic comparison at a glance

| Method | Rounds the original SDP? | Random source | Proven bound as stated in sources |
| --- | --- | --- | --- |
| Frieze–Jerrum (Algorithm 11) | Yes | Independent Gaussian directions | Yes, expected $\alpha_k$ for its setting |
| Thesis $k$-clustering (Algorithm 12) | Yes | Random centroid initialization for spherical $k$-means | **Not established** in the thesis |
| Hybrid FJ + clustering (proposed here) | Yes | At least the FJ draws | **At least baseline FJ** by pointwise retention |
| Iterated SDP/fixed-point rounding [FKP22] | Yes, by a sequence of new SDP optimizations | Deterministic update after initial SDP | Convergence and geometric fixed-point results; **not** a demonstrated new universal FJ cut factor |

**Next:** [Thesis experiments](03-thesis-experiments.md).
