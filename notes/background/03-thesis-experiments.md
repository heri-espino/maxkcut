# Thesis Chapter 7: benchmark design, metrics, findings, and numerical repair

**Source:** [TH21, §7.4 and §7.4.2, PDF pp. 96–101 / printed pp. 86–91], particularly **Table 1, Figure 26, Figure 27**. See [source index](README.md).

> **Reading discipline:** The following are *reported experimental observations*, not results generated in this repository. The thesis provides plotted aggregates but no per-instance result CSV in the supplied chapter. We do not infer invented exact numbers from its figures.

## 1. What was compared

- **Algorithm 11 (“RandRound”):** Frieze–Jerrum Gaussian randomized rounding.
- **Algorithm 12 (“K-Means”):** spherical \(k\)-means on the **same kind of SDP solution vectors**, initialized with \(k\) independent Gaussian vectors normalized to unit length.
- Values of \(k\): **3, 4, 5**.
- The SDP: **thesis Eq. 7.6**, solved with Mathematica \`SemidefiniteOptimization\`. Vectors \(v_i\) obtained using a factorization \(X=V^\top V\), taking columns of \(V\).
- This is a **rounding** comparison, not a comparison of different graph objectives or different SDP formulations.

The 2020 G-set results for **Max-Cut, \(k=2\)** [RF20] are a separate experiment. Do not combine their improvements or graph sizes into the figures below.

## 2. The four graph generators (Table 1, PDF p. 97)

Let \(n\in\{50,100,200\}\), with edges independently sampled for each unordered vertex pair unless otherwise specified.

| Type | Edge generation | Edge weights | Purpose / source observation |
| --- | --- | --- | --- |
| **A** | \(G(n,1/2)\) | \(+1\) | Dense Erdős–Rényi; little planted partition structure |
| **B** | \(G(n,1/2)\) | Independent discrete uniform integers in \([-50,50]\) | Signed weights; challenge conventional cut-gap interpretation |
| **C** | \(G(n,10/n)\) | \(+1\) | Sparse with expected degree \(10(n-1)/n\), approximately 10 |
| **D** | If \(i\) and \(j\) are in opposite predetermined halves: probability \(0.1\); otherwise \(0.05\) | \(+1\) | Favors a **two-way** partition even when rounding into \(k=3,4,5\) |

Thesis Table 1 reports these expected edge counts:

| \(n\) | Type A and B | Type C | Type D |
| ---: | ---: | ---: | ---: |
| 50 | 612.5 | 245–250* | 92.5 |
| 100 | 2475 | 495–500* | 372.5 |
| 200 | 9950 | 995–1000* | 1495 |

\`*\` The **source table itself** lists \(250,500,1000\) (using \(10n/2\)); exact \(G(n,10/n)\) with **no self-loops** gives \(5(n-1)=245,495,995\). The distinction is a finite-\(n\) expectation, not a claim about the plotted experiment. Replication should use the explicitly stated **probability \(10/n\)**.

For each graph type the thesis generated:
- **50 instances** with \(n=50\);
- **20 instances** with \(n=100\);
- **5 instances** with \(n=200\).

Thus \(4\times(50+20+5)=\mathbf{300}\) *graph instances*. Each is evaluated for the stated \(k\) values. The source does **not** give a complete rounding-seed/repetition and error-bar protocol for each instance in this chapter. Do not confuse 300 graphs with 300 independent random clustering runs.

## 3. Quality measures — Eq. 7.10 and “Cut Gap”

Define \(F=F(c)\), the cut returned by a rounding method, and \(S=S_{\mathrm{SDP}}\), the optimum value of the relaxation as used in the thesis.

For **nonnegative weights**:

\[
\operatorname{AIG}(c)=\frac{F}{S},\qquad
\operatorname{CG}(c)=\frac{F}{\sum_{i<j}w_{ij}}.
\]

TH21 calls the first quantity **Average Integer Gap (AIG)**, even though it is expressed as a ratio rather than the usual absolute or relative “integrality gap.” Larger is better here; if the SDP is a reliable upper bound, \(0\le F/S\le1\) for nonnegative-weight instances. CG expresses the fraction of total edge weight cut, **not** a certified ratio to the unknown integer optimum.

For **signed weights** the thesis uses

\[
W_-=\sum_{i<j}\min(w_{ij},0),\qquad
W_+=\sum_{i<j}\max(w_{ij},0)
\]

and shifts the ratios:

\[
\boxed{
\operatorname{AIG}_{\pm}
=\frac{F-W_-}{S-W_-},\qquad
\operatorname{CG}_{\pm}
=\frac{F-W_-}{W_+-W_-}.
}
\]

These require nonzero denominators. \(W_-\) is a **lower bound**, not an achievable cut value in general; \(W_+\) is an **upper bound** for a signed-weight cut (in a simple graph, choose at most all positive edges). Therefore large or small CG values cannot be interpreted as approximation guarantees. For signed instances the source specifically notes that CG can be misleading even if a rounded solution is close to the SDP relaxation.

### An important logical distinction

\[
F(c)\ \le\ F^\star\ \le\ S_{\mathrm{SDP}}
\]

when the relaxation is valid and accurately solved. AIG near 1 indicates a **small observed relaxation-to-candidate gap**. It does not by itself establish that the same bound will hold on other graphs, or that the rounding method has a worst-case approximation factor.

## 4. What Figure 26 actually reports (PDF p. 99)

**Figure 26a:** AIG for the four graph families, three sizes, and \(k=3,4,5\). **Figure 26b:** CG for the same settings. Cyan indicates RandRound; magenta shows the higher bar attributable to K-Means. The author concludes in §7.4.1 (printed p. 90) that clustering improves the **reported aggregate values** across the displayed configurations.

Source-reported qualitative details:

- **A:** K-Means AIG approaches **about 0.97** and remains similar across the tested \(k\); the randomized baseline shows worsening rounding efficiency as \(k\) grows. Cut value increases with \(k\).
- **B:** Increasing \(k\) **decreases AIG** for both methods while the cut value stays approximately constant. Signed weights prevent interpreting “more classes” as automatically better. CG is particularly low/misleading here.
- **C:** With expected degree approximately 10, clustering at \(k=5\) reaches an AIG **near 0.98**, fairly stable across \(n\). The thesis observes a roughly **0.04** AIG difference versus randomized rounding in the discussed setting. It **speculates**, without testing, that extending to \(k=10\) might nearly close the gap.
- **D:** For \(n=50\), K-Means obtains cuts nearly equal to the SDP optimum; at \(k=5\) it visually closes the measured AIG gap. For larger \(n\), AIG falls and the gap no longer closes.
- **Overall:** The *plotted averages* favor clustering. Individual-run superiority and guaranteed approximation ratios are **not demonstrated** by Figure 26.

The displayed figure has no table of precise AIG/CG numbers, confidence intervals, paired differences, or p-values. Avoid presenting visual readings as exact measurements. For a new study, archive per-instance values and statistical uncertainty.

## 5. Numerical PSD problem and Figure 27 (PDF pp. 100–101)

The thesis found that Mathematica could return nominally PSD SDP matrices \(X\) with eigenvalues as low as about **\(-10^{-9}\)**. A strict Cholesky factorization can then fail.

The reported fix follows **Higham (1988)**: replace the computed \(X\) with a **nearby PSD matrix** \(\bar X\), then factor \(\bar X\) rather than \(X\). Figure 27 compares AIG obtained from unmodified vs corrected matrices in the subset of examples where the original \(X\) was already PSD; the general pattern was retained. The notable reported exception is **C50 at \(k=3\)**, whose AIG increased after correction.

Important implementation distinction, **derived from feasibility requirements**:

1. PSD projection alone does **not** necessarily preserve **diag\((X)=1\)** or **\(X_{ij}\ge -1/(k-1)\)**.
2. A repaired matrix used **only to extract rounded vectors** may be operationally useful while no longer being a feasible *SDP solution*. Label it as a rounding factorization surrogate, not a new certified optimum.
3. For meaningful, comparable SDP bounds, also log eigenvalue minima, diagonal residuals, lower-bound violations, primal/dual gaps, and any objective change after correction.
4. When \(X\) is PSD but singular, a **spectral factorization** with clipped tiny negatives is often computationally more straightforward than strict Cholesky. This is a proposed implementation choice, **not** the algorithm the thesis reports.

## 6. Limits and documentation ambiguities to preserve

- **Scaling:** the reported benchmark goes up to \(n=200\). Chapter 8 (printed p. 97) says its SDP software stops working for \(n\ge200\); interpret that as a practical scaling barrier, **not** a rigorous statement that none of the \(n=200\) cases ran.
- **Negative weights:** do not compare unshifted \(F/S\) to positive-only scores.
- **Generator D:** favors a **bipartition**, not a planted \(k\)-way partition. Chapter 8 explicitly recommends benchmarking planted \(k\)-way examples.
- **Seed protocol:** the source text specifies random initialization distribution but not enough details here to reproduce exactly the full set of runs; choose and report a protocol rather than invent one.
- **No significance tests shown** in this chapter for the Fig. 26 differences; per-graph variability is not quantified.
- **Comparison fairness:** random directions vs \(k\)-means has different computational budgets; better cut weight need not imply greater speed.
- **PDF figure interpretation:** the manuscript uses “all cases” to describe the outcomes in the plotted settings. The claim should **not** be enlarged into “every input graph always improves.”

## 7. Research implications

The most immediate empirical question is not “Does clustering beat FJ on this same aggregate plot?” (the thesis says yes), but:

> Under a fixed, reproducible SDP solution and matched computational budgets, **when** does spherical \(k\)-means rounding improve cuts, by **how much**, and **which graph/SDP spectral properties predict the gain?**

Chapter 8 names larger \(k\), planted \(k\)-way structure, better SDP scaling, and graph-type variation as future work. See [related questions](04-related-work-and-questions.md) and [protocol](05-reproduction-protocol.md).
