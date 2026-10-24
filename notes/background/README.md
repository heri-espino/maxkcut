# Research background: Max-$k$-Cut and SDP rounding

This directory is a **source-grounded reading notebook**, not a claim of new results or an implementation. Its primary source is Chapter 7 of Rodríguez Fernández's 2021 dissertation. We separate **reported results**, **mathematical deductions**, and **proposed experiments**.

## Reading order

1. [Problem, simplex encoding, and semidefinite relaxation](01-problem-and-sdp.md) — Thesis §§7.1–7.2, equations (7.1)–(7.6), Lemma 5.
2. [Rounding algorithms and geometric clustering](02-rounding-methods.md) — Thesis §7.3, especially Algorithms 11–12; connections to the 2020 Max-Cut study.
3. [Benchmarks, results, and numerical limitations](03-thesis-experiments.md) — Thesis §7.4, Table 1, Figure 26, Figure 27.
4. [Prior work, distinctions, and research questions](04-related-work-and-questions.md) — Thesis §§7.5 and 8, the 2020 paper, and the 2022 JMLR paper.
5. [Reproduction and evaluation protocol](05-reproduction-protocol.md) — A proposed protocol; **not** something reported as executed.

## Primary sources and page conventions

| Key | Reference | Location in supplied sources |
| --- | --- | --- |
| **[TH21]** | Ángel Eduardo Rodríguez Fernández (2021), *On the Study of Fitness Landscapes and the Max-Cut Problem*, doctoral dissertation, Universität Leipzig. [Persistent library identifier](https://nbn-resolving.org/urn:nbn:de:bsz:15-qucosa2-769667); [MPI publication record](https://www.mis.mpg.de/publications/2021). | Supplied PDF `2021_Fernandez_OnTheStudyOfFitnessLandscapesA_Qucosa-UBL.pdf`; **chapter 7: PDF pages 87–102 = printed pp. 77–92**. Chapter 8: PDF pages 105–107 = printed pp. 95–97. |
| **[RF20]** | Rodríguez-Fernández, González-Torres, Menchaca-Méndez, and Stadler (2020), “Clustering Improves the Goemans–Williamson Approximation for the Max-Cut Problem,” *Computation* **8**(3), 75. DOI: [10.3390/computation8030075](https://doi.org/10.3390/computation8030075). | Supplied PDF `computation-08-00075-v2.pdf`; 12 pages. The experiments are for **Max-Cut, $k=2$**. |
| **[FKP22]** | Felzenszwalb, Klivans, and Paul (2022), “Clustering with Semidefinite Programming and Fixed Point Iteration,” *JMLR* **23**, paper 21-0402. [Paper](https://jmlr.org/papers/v23/21-0402.html). | Supplied PDF `21-0402.pdf`; 23 pages. A **different SDP-rounding algorithm** relevant to general $k$. |

Page numbers below distinguish **PDF pages** (1-based viewer position) from the **printed page labels** in [TH21]. For example, §7.3.4 and Algorithm 12 are on **PDF p. 96 / printed p. 86**.

## The central claim, carefully delimited

[TH21, §7.3.4, §7.4, Fig. 26] compares **Frieze–Jerrum Gaussian randomized rounding** with **spherical $k$-means of the optimized SDP vectors** for $k=3,4,5$. In the reported random-graph benchmark, the clustering method gives larger cuts across the configurations shown in Figure 26. **The author does not establish a universal approximation ratio for Algorithm 12.** Empirical cut improvements and theoretical approximation guarantees are distinct.

The supplied repository initially had a short `README.md` and `G-Set/` MATLAB-format files. **G-Set is not the 300-instance A/B/C/D random-graph benchmark of chapter 7.** The G-set corpus belongs to the Max-Cut benchmark used in [RF20] and thesis chapter 6; do not present it as having reproduced chapter 7.

## Conventions for the notes

- $G=(V,E,w)$: undirected weighted graph; $n=|V|$; $k\ge 2$.
- $c:V\to\{1,\dots,k\}$: a (possibly not surjective) partition labeling, unless explicitly stated “exactly $k$ nonempty sets.”
- $F(c)=\sum_{i<j}w_{ij}\mathbf 1\{c(i)\ne c(j)\}$: cut objective.
- $X\succeq0$: positive-semidefinite matrix; $X_{ij}=v_i^\top v_j$.
- $S_{\mathrm{SDP}}$: **upper bound** supplied by a feasible, adequately solved maximization SDP (subject to verified numerical tolerances).
- $F^\star$: unknown integer optimum; $F(c)\le F^\star\le S_{\mathrm{SDP}}$ when the SDP is valid and solved to the required accuracy.
- Equations prefixed “(7.x)” and “Algorithm 11/12” are numbered as in [TH21]. New deductions are labeled as such.
- The original PDFs are **not** duplicated in this repository. Please use the supplied source files or the publication links above.

## Important unresolved distinctions

1. The thesis uses clustering **to round the SDP for Max-$k$-Cut**; [FKP22] uses a different iterative SDP procedure that can itself be interpreted as clustering. These should not be conflated.
2. The guarantee of Frieze–Jerrum [TH21, §7.3.3] applies to its **randomized algorithm in expectation**, not automatically to $k$-means rounding.
3. For negative weights, the “cut gap” and approximation ratios need an explicit weight shift; comparing raw ratios may mislead.
4. Figure 26 gives aggregate plotted results, not a machine-readable table of per-instance outcomes. The documentation does not pretend that its points were digitized or independently reproduced.

## Scope of this documentation change

The notes reconstruct the mathematics and the published experimental claims, mark open proofs and implementation decisions, and provide a falsifiable experiment plan. **No SDP, clustering, solver benchmark, or numerical experiment was run as part of producing these notes.**
